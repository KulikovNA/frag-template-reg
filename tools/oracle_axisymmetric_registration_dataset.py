#!/usr/bin/env python
"""Validate axisymmetric registration from GT profile coordinates."""

from __future__ import annotations

import argparse
import copy
import sys
from itertools import islice
from pathlib import Path

import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fragreg.data import build_dataloader
from fragreg.evaluation.metrics import compute_profile_batch_metrics, summarize_metric_dicts
from fragreg.geometry.axisymmetric_solver import estimate_axisymmetric_pose
from fragreg.utils import get_units_cfg, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="Path to Python config.")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--init-mode", default=None, choices=["auto", "gt", "gt_perturbed"])
    parser.add_argument("--limit-batches", type=int, default=None)
    return parser.parse_args()


def dataloader_cfg_for_split(cfg, split: str) -> dict:
    key = f"{split}_dataloader"
    if key in cfg:
        loader_cfg = copy.deepcopy(cfg[key])
    else:
        loader_cfg = copy.deepcopy(cfg.train_dataloader)
        loader_cfg["dataset"]["split"] = split
    loader_cfg["shuffle"] = False
    loader_cfg["drop_last"] = False
    loader_cfg["num_workers"] = 0
    loader_cfg["persistent_workers"] = False
    loader_cfg["dataset"]["random_sample"] = False
    loader_cfg["dataset"]["return_profile"] = True
    return loader_cfg


def make_init(T_gt: torch.Tensor, init_mode: str) -> torch.Tensor | None:
    if init_mode == "auto":
        return None
    init = T_gt.clone()
    if init_mode == "gt_perturbed":
        delta = torch.tensor([1e-4, -1e-4, 1e-4], dtype=init.dtype, device=init.device)
        init[:, :3, 3] = init[:, :3, 3] + delta
    return init


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    units_cfg = get_units_cfg(cfg)
    eval_cfg = cfg.get("eval_cfg", {})
    solver_cfg = dict(eval_cfg.get("solver", {}))
    init_mode = args.init_mode or solver_cfg.pop("init_mode", "auto")
    axis = eval_cfg.get("axis", cfg.loss.get("axis", "y"))
    device = torch.device(args.device)

    loader = build_dataloader(dataloader_cfg_for_split(cfg, args.split), is_train=False)
    if len(loader.dataset) == 0:
        raise RuntimeError("Dataset contains no valid samples.")

    iterable = loader if args.limit_batches is None else islice(loader, args.limit_batches)
    total = len(loader) if args.limit_batches is None else args.limit_batches
    all_metrics = []
    solver_losses = []
    for batch in tqdm(iterable, total=total, desc=f"axisym oracle {args.split}"):
        points_C_orig = batch["points_C_orig"].to(device)
        profile_O = batch["profile_O"].to(device)
        valid_mask = batch["valid_mask"].to(device)
        T_gt = batch["T_C_from_O"].to(device)
        init = make_init(T_gt, init_mode)

        T_pred, diagnostics = estimate_axisymmetric_pose(
            points_C_orig,
            profile_O,
            weights=valid_mask.float(),
            axis=axis,
            init=init,
            num_iters=int(solver_cfg.get("num_iters", 200)),
            lr=float(solver_cfg.get("lr", 1e-2)),
            huber_beta=float(solver_cfg.get("huber_beta", 0.01)),
        )
        if isinstance(diagnostics, dict) and "final_loss" in diagnostics:
            solver_losses.append(float(diagnostics["final_loss"]))

        batch_metrics = compute_profile_batch_metrics(
            pred_profile_O=profile_O,
            target_profile_O=profile_O,
            points_C_orig=points_C_orig,
            T_pred=T_pred,
            T_gt=T_gt,
            valid_mask=valid_mask,
            axis=axis,
            coord_scale=units_cfg["coord_scale"],
            coord_unit=units_cfg["coord_unit"],
            residual_profile_O=profile_O,
        )
        all_metrics.extend(batch_metrics)

    summary = summarize_metric_dicts(all_metrics)
    summary["num_samples"] = len(all_metrics)
    summary["coord_scale"] = units_cfg["coord_scale"]
    summary["coord_unit"] = units_cfg["coord_unit"]
    summary["axis"] = axis
    summary["init_mode"] = init_mode
    if solver_losses:
        summary["solver_final_loss"] = float(torch.tensor(solver_losses, dtype=torch.float64).mean().item())
    for key in sorted(summary):
        print(f"{key}: {summary[key]}")


if __name__ == "__main__":
    main()
