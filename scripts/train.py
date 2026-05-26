#!/usr/bin/env python
"""Train DGCNN correspondence model."""

from __future__ import annotations

import argparse
import copy
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fragreg.data import build_dataloader
from fragreg.evaluation import Evaluator
from fragreg.registry import build_loss, build_model
from fragreg.training import save_checkpoint, train_one_epoch
from fragreg.training.logger import JSONLLogger, TensorBoardLogger, setup_logger
from fragreg.utils import apply_units_to_loss_cfg, get_units_cfg, load_config, seed_everything
from fragreg.utils.io import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="Path to Python config.")
    parser.add_argument("--resume", default=None, help="Optional checkpoint to resume from.")
    parser.add_argument("--device", default=None, help="Override device, e.g. cuda or cpu.")
    parser.add_argument("--max-epochs", type=int, default=None, help="Runtime override for train_cfg.max_epochs.")
    parser.add_argument("--limit-train-batches", type=int, default=None, help="Smoke-test limit for train batches.")
    parser.add_argument("--limit-val-batches", type=int, default=None, help="Smoke-test limit for val batches.")
    parser.add_argument("--batch-size", type=int, default=None, help="Runtime dataloader batch-size override.")
    parser.add_argument("--num-workers", type=int, default=None, help="Runtime dataloader worker override.")
    parser.add_argument("--num-points", type=int, default=None, help="Runtime dataset num_points override.")
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=None,
        help="Runtime override for train_cfg.gradient_accumulation_steps.",
    )
    return parser.parse_args()


def build_optimizer(model: torch.nn.Module, cfg: dict) -> torch.optim.Optimizer:
    cfg = dict(cfg)
    optim_type = cfg.pop("type", "AdamW")
    if optim_type != "AdamW":
        raise ValueError(f"Only AdamW is supported in the first version, got {optim_type!r}.")
    return torch.optim.AdamW(model.parameters(), **cfg)


def log_model_summary(model: torch.nn.Module, logger) -> None:
    total_params = sum(param.numel() for param in model.parameters())
    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    logger.info("Model: %s", model.__class__.__name__)
    logger.info("Model parameters: total=%s, trainable=%s", f"{total_params:,}", f"{trainable_params:,}")
    logger.info("Learnable layers:")
    for name, module in model.named_modules():
        if not name:
            continue
        direct_params = sum(param.numel() for param in module.parameters(recurse=False))
        trainable_direct = sum(param.numel() for param in module.parameters(recurse=False) if param.requires_grad)
        if direct_params == 0:
            continue
        logger.info(
            "  %-36s %-18s params=%s trainable=%s",
            name,
            module.__class__.__name__,
            f"{direct_params:,}",
            f"{trainable_direct:,}",
        )


def apply_loader_overrides(loader_cfg: dict, args: argparse.Namespace) -> dict:
    loader_cfg = copy.deepcopy(loader_cfg)
    if args.batch_size is not None:
        loader_cfg["batch_size"] = args.batch_size
    if args.num_workers is not None:
        loader_cfg["num_workers"] = args.num_workers
        if args.num_workers == 0:
            loader_cfg["persistent_workers"] = False
    if args.num_points is not None:
        loader_cfg["dataset"]["num_points"] = args.num_points
    return loader_cfg


def resolve_output_path(value: str | Path | None, default: Path, work_dir: Path) -> Path:
    if value is None:
        return default
    path = Path(value)
    if path.is_absolute():
        return path
    if len(path.parts) == 1:
        return work_dir / path
    return path


def resolve_checkpoint_dir(train_cfg: dict, work_dir: Path) -> Path:
    value = train_cfg.get("checkpoint_dir", train_cfg.get("weights_dir"))
    return resolve_output_path(value, default=work_dir, work_dir=work_dir)


def should_use_date_subdir(train_cfg: dict) -> bool:
    return bool(train_cfg.get("date_subdir", True))


def make_date_subdir(train_cfg: dict) -> str:
    date_format = str(train_cfg.get("date_subdir_format", "%d_%m_%Y"))
    return datetime.now().strftime(date_format)


def infer_work_dir_from_resume(resume_path: str | Path, train_cfg: dict) -> Path:
    """Infer the run work_dir from a checkpoint path.

    For checkpoints saved into a short relative checkpoint dir, e.g.
    ``work_dir/checkpoints/latest.pth``, return ``work_dir``. Otherwise return
    the checkpoint parent directory.
    """

    checkpoint_path = Path(resume_path).expanduser().resolve()
    checkpoint_parent = checkpoint_path.parent
    checkpoint_dir_value = train_cfg.get("checkpoint_dir", train_cfg.get("weights_dir"))
    if checkpoint_dir_value is None:
        return checkpoint_parent

    configured_checkpoint_dir = Path(checkpoint_dir_value)
    is_short_relative_name = (
        not configured_checkpoint_dir.is_absolute()
        and len(configured_checkpoint_dir.parts) == 1
    )
    if is_short_relative_name and checkpoint_parent.name == configured_checkpoint_dir.name:
        return checkpoint_parent.parent
    return checkpoint_parent


def resolve_work_dir(
    train_cfg: dict,
    config_name: str,
    resume_path: str | Path | None = None,
) -> tuple[Path, Path, str | None]:
    base_work_dir = Path(train_cfg.get("work_dir", f"work_dirs/{config_name}"))
    if resume_path is not None and bool(train_cfg.get("resume_work_dir", True)):
        work_dir = infer_work_dir_from_resume(resume_path, train_cfg)
        if should_use_date_subdir(train_cfg) and work_dir.parent.resolve() == base_work_dir.resolve():
            return base_work_dir, work_dir, work_dir.name
        return base_work_dir, work_dir, None
    if not should_use_date_subdir(train_cfg):
        return base_work_dir, base_work_dir, None
    run_date = make_date_subdir(train_cfg)
    return base_work_dir, base_work_dir / run_date, run_date


def dataset_scene_ids(loader_cfg: dict) -> Any:
    return copy.deepcopy(loader_cfg.get("dataset", {}).get("scene_ids"))


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    seed_everything(int(cfg.get("seed", 42)))
    units_cfg = get_units_cfg(cfg)

    train_cfg = cfg.train_cfg
    base_work_dir, work_dir, run_date = resolve_work_dir(train_cfg, cfg._config_name, resume_path=args.resume)
    checkpoint_dir = resolve_checkpoint_dir(train_cfg, work_dir)
    logger = setup_logger(work_dir)
    json_log = bool(train_cfg.get("json_log", True))
    tensorboard_log = bool(train_cfg.get("tensorboard_log", False))
    tensorboard_dir = resolve_output_path(train_cfg.get("tensorboard_dir"), default=work_dir / "tensorboard", work_dir=work_dir)
    metrics_logger = JSONLLogger(work_dir / "metrics.jsonl", enabled=json_log)
    tb_logger = TensorBoardLogger(tensorboard_dir, enabled=tensorboard_log)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    logger.info("Using device: %s", device)
    logger.info("Torch: %s, CUDA available: %s", torch.__version__, torch.cuda.is_available())
    logger.info("Base work dir: %s", base_work_dir)
    logger.info("Date subdir: %s", run_date or "<disabled>")
    logger.info("Work dir: %s", work_dir)
    logger.info("Checkpoint dir: %s", checkpoint_dir)
    logger.info(
        "Coordinate units: dataset=%s, log/loss=%s, coord_scale=%s",
        units_cfg["dataset_coord_unit"],
        units_cfg["coord_unit"],
        units_cfg["coord_scale"],
    )
    logger.info("JSON logging: %s", json_log)
    logger.info("TensorBoard logging: %s", tb_logger.available)
    logger.info("Overfit mode: %s", bool(cfg.get("overfit_mode", False)))
    logger.info("Train scene_ids: %s", dataset_scene_ids(cfg.train_dataloader))
    if "val_dataloader" in cfg:
        logger.info("Val scene_ids: %s", dataset_scene_ids(cfg.val_dataloader))
    if tensorboard_log and not tb_logger.available:
        logger.warning("TensorBoard logging requested, but tensorboard is not installed.")

    train_loader = build_dataloader(apply_loader_overrides(cfg.train_dataloader, args), is_train=True)
    val_loader = (
        build_dataloader(apply_loader_overrides(cfg.val_dataloader, args), is_train=False)
        if "val_dataloader" in cfg
        else None
    )
    logger.info("Train samples: %d", len(train_loader.dataset))
    if hasattr(train_loader.dataset, "get_summary"):
        logger.info("Train dataset summary: %s", train_loader.dataset.get_summary())
    if val_loader is not None:
        logger.info("Val samples: %d", len(val_loader.dataset))
        if hasattr(val_loader.dataset, "get_summary"):
            logger.info("Val dataset summary: %s", val_loader.dataset.get_summary())

    model = build_model(cfg.model).to(device)
    log_model_summary(model, logger)
    loss_fn = build_loss(apply_units_to_loss_cfg(cfg.loss, units_cfg)).to(device)
    optimizer = build_optimizer(model, cfg.optim)

    start_epoch = 1
    best_score = float("inf")
    metric_for_best = train_cfg.get("metric_for_best", "residual_rmse")
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["state_dict"])
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        checkpoint_metrics = checkpoint.get("metrics", {})
        same_units = (
            checkpoint_metrics.get("coord_scale") == units_cfg["coord_scale"]
            and checkpoint_metrics.get("coord_unit") == units_cfg["coord_unit"]
        )
        if same_units and metric_for_best:
            best_score = float(checkpoint_metrics.get(metric_for_best, best_score))
        elif checkpoint_metrics:
            logger.warning("Checkpoint metric units differ from current config; best score will be reset.")
        logger.info("Resumed from %s at epoch %d", args.resume, start_epoch)

    max_epochs = int(args.max_epochs or train_cfg.get("max_epochs", 100))
    val_interval = int(train_cfg.get("val_interval", 1))
    checkpoint_interval = int(train_cfg.get("checkpoint_interval", 1))
    save_latest = bool(train_cfg.get("save_latest", True))
    save_best = bool(train_cfg.get("save_best", True))
    gradient_accumulation_steps = int(
        args.gradient_accumulation_steps
        or train_cfg.get("gradient_accumulation_steps", train_cfg.get("grad_accum_steps", 1))
    )
    if gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be >= 1.")
    logger.info("Gradient accumulation steps: %d", gradient_accumulation_steps)
    eval_cfg = cfg.get("eval_cfg", {})
    axis = eval_cfg.get("axis", cfg.loss.get("axis", "z"))
    solver_cfg = eval_cfg.get("solver", {})
    return_per_sample = bool(eval_cfg.get("return_per_sample", False))

    try:
        for epoch in range(start_epoch, max_epochs + 1):
            train_metrics = train_one_epoch(
                model,
                train_loader,
                loss_fn,
                optimizer,
                device,
                epoch,
                max_batches=args.limit_train_batches,
                gradient_accumulation_steps=gradient_accumulation_steps,
            )
            payload = {
                "epoch": epoch,
                "split": "train",
                "duration_sec": train_metrics.get("epoch_time_sec"),
                **{f"train/{k}": v for k, v in train_metrics.items()},
            }
            metrics_logger.log(payload)
            tb_logger.log_scalars(train_metrics, step=epoch, prefix="train")
            logger.info("Epoch %d train: %s", epoch, train_metrics)

            val_metrics = {}
            if val_loader is not None and val_interval > 0 and epoch % val_interval == 0:
                val_start = time.perf_counter()
                evaluator = Evaluator(
                    model,
                    val_loader,
                    device=device,
                    loss_fn=loss_fn,
                    axis=axis,
                    max_batches=args.limit_val_batches,
                    coord_scale=units_cfg["coord_scale"],
                    coord_unit=units_cfg["coord_unit"],
                    solver_cfg=solver_cfg,
                    profile_eval_modes=eval_cfg.get("profile_eval_modes"),
                    return_per_sample=return_per_sample,
                )
                val_metrics = evaluator.evaluate(desc=f"val epoch {epoch}")
                val_time_sec = time.perf_counter() - val_start
                val_batches = (
                    len(val_loader)
                    if args.limit_val_batches is None
                    else min(args.limit_val_batches, len(val_loader))
                )
                val_metrics["eval_time_sec"] = val_time_sec
                val_metrics["seconds_per_batch"] = val_time_sec / max(val_batches, 1)
                per_sample_metrics = val_metrics.pop("per_sample", None)
                if per_sample_metrics is not None:
                    per_sample_path = work_dir / "per_sample_metrics_val.json"
                    for item in per_sample_metrics:
                        item["epoch"] = epoch
                        item["split"] = "val"
                    write_json(per_sample_path, per_sample_metrics)
                    logger.info("Saved per-sample val metrics: %s", per_sample_path)
                metrics_logger.log(
                    {
                        "epoch": epoch,
                        "split": "val",
                        "duration_sec": val_time_sec,
                        **{f"val/{k}": v for k, v in val_metrics.items()},
                    }
                )
                tb_logger.log_scalars(val_metrics, step=epoch, prefix="val")
                logger.info("Epoch %d val: %s", epoch, val_metrics)

            checkpoint_metrics = val_metrics or train_metrics
            if save_latest:
                save_checkpoint(
                    checkpoint_dir / "latest.pth",
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    metrics=checkpoint_metrics,
                    cfg=cfg.to_dict(),
                )

            if checkpoint_interval > 0 and epoch % checkpoint_interval == 0:
                save_checkpoint(
                    checkpoint_dir / f"epoch_{epoch:04d}.pth",
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    metrics=checkpoint_metrics,
                    cfg=cfg.to_dict(),
                )
                logger.info("Saved periodic checkpoint for epoch %d", epoch)

            score = val_metrics.get(metric_for_best)
            if save_best and score is not None and score < best_score:
                best_score = float(score)
                save_checkpoint(
                    checkpoint_dir / "best.pth",
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    metrics=val_metrics,
                    cfg=cfg.to_dict(),
                )
                logger.info("Saved new best checkpoint: %s=%.6g", metric_for_best, best_score)
    finally:
        tb_logger.close()


if __name__ == "__main__":
    main()
