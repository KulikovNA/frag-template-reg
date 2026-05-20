#!/usr/bin/env python
"""Evaluate a trained checkpoint."""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fragreg.data import build_dataloader
from fragreg.evaluation import Evaluator
from fragreg.registry import build_loss, build_model
from fragreg.training import load_checkpoint
from fragreg.utils import apply_units_to_loss_cfg, get_units_cfg, load_config
from fragreg.utils.io import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="Path to Python config.")
    parser.add_argument("checkpoint", help="Path to .pth checkpoint.")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"], help="Dataset split to evaluate.")
    parser.add_argument("--device", default=None, help="Override device, e.g. cuda or cpu.")
    parser.add_argument("--out", default=None, help="Optional output metrics JSON path.")
    parser.add_argument("--limit-batches", type=int, default=None, help="Smoke-test limit for eval batches.")
    parser.add_argument("--batch-size", type=int, default=None, help="Runtime dataloader batch-size override.")
    parser.add_argument("--num-workers", type=int, default=None, help="Runtime dataloader worker override.")
    parser.add_argument("--num-points", type=int, default=None, help="Runtime dataset num_points override.")
    return parser.parse_args()


def dataloader_cfg_for_split(cfg, split: str) -> dict:
    key = f"{split}_dataloader"
    if key in cfg:
        return cfg[key]
    base = copy.deepcopy(cfg.val_dataloader if "val_dataloader" in cfg else cfg.train_dataloader)
    base["dataset"]["split"] = split
    base["dataset"]["random_sample"] = False
    return base


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


def dataset_scene_ids(loader_cfg: dict):
    return copy.deepcopy(loader_cfg.get("dataset", {}).get("scene_ids"))


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    units_cfg = get_units_cfg(cfg)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    model = build_model(cfg.model).to(device)
    load_checkpoint(args.checkpoint, model, map_location=device)
    loss_fn = build_loss(apply_units_to_loss_cfg(cfg.loss, units_cfg)).to(device)

    loader_cfg = apply_loader_overrides(dataloader_cfg_for_split(cfg, args.split), args)
    loader = build_dataloader(loader_cfg, is_train=False)
    print(f"overfit_mode: {bool(cfg.get('overfit_mode', False))}")
    print(f"{args.split}_scene_ids: {dataset_scene_ids(loader_cfg)}")
    eval_cfg = cfg.get("eval_cfg", {})
    evaluator = Evaluator(
        model,
        loader,
        device=device,
        loss_fn=loss_fn,
        axis=eval_cfg.get("axis", cfg.loss.get("axis", "z")),
        max_batches=args.limit_batches,
        coord_scale=units_cfg["coord_scale"],
        coord_unit=units_cfg["coord_unit"],
        solver_cfg=eval_cfg.get("solver", {}),
        profile_eval_modes=eval_cfg.get("profile_eval_modes"),
        return_per_sample=bool(eval_cfg.get("return_per_sample", False)),
    )
    metrics = evaluator.evaluate(desc=f"evaluate {args.split}")
    per_sample_metrics = metrics.pop("per_sample", None)
    metrics["split"] = args.split
    metrics["num_samples"] = len(loader.dataset)

    out_path = Path(args.out) if args.out else Path(args.checkpoint).resolve().parent / f"metrics_{args.split}.json"
    write_json(out_path, metrics)
    per_sample_path = None
    if per_sample_metrics is not None:
        per_sample_path = out_path.parent / f"per_sample_metrics_{args.split}.json"
        write_json(per_sample_path, per_sample_metrics)
    for key in sorted(metrics):
        value = metrics[key]
        if isinstance(value, list):
            print(f"{key}: <{len(value)} items>")
        else:
            print(f"{key}: {value}")
    print(f"saved: {out_path}")
    if per_sample_path is not None:
        print(f"saved per-sample: {per_sample_path}")


if __name__ == "__main__":
    main()
