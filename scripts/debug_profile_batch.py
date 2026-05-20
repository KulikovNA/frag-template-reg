#!/usr/bin/env python
"""Inspect one profile batch and optionally run a checkpoint."""

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
from fragreg.evaluation.metrics import compute_profile_batch_metrics, summarize_metric_dicts
from fragreg.geometry.axisymmetric_solver import estimate_axisymmetric_pose
from fragreg.registry import build_model
from fragreg.training import load_checkpoint
from fragreg.utils import get_units_cfg, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="Path to Python config.")
    parser.add_argument("--checkpoint", default=None, help="Optional model checkpoint.")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    units_cfg = get_units_cfg(cfg)
    eval_cfg = cfg.get("eval_cfg", {})
    solver_cfg = dict(eval_cfg.get("solver", {}))
    axis = eval_cfg.get("axis", cfg.loss.get("axis", "y"))
    loader_cfg = copy.deepcopy(cfg[f"{args.split}_dataloader"] if f"{args.split}_dataloader" in cfg else cfg.train_dataloader)
    loader_cfg["num_workers"] = 0
    loader_cfg["persistent_workers"] = False
    loader_cfg["dataset"]["return_profile"] = True
    loader = build_dataloader(loader_cfg, is_train=args.split == "train")
    batch = next(iter(loader))
    device = torch.device(args.device)

    print(f"dataset samples: {len(loader.dataset)}")
    for key in ("points_C_orig", "points_C", "points_O", "profile_O", "valid_mask", "T_C_from_O"):
        value = batch[key]
        print(f"{key}: shape={tuple(value.shape)} dtype={value.dtype}")
    print("first profile_O[0, :5]:")
    print(batch["profile_O"][0, :5])

    points_C_orig = batch["points_C_orig"].to(device)
    profile_O = batch["profile_O"].to(device)
    valid_mask = batch["valid_mask"].to(device)
    T_gt = batch["T_C_from_O"].to(device)
    T_oracle, diagnostics = estimate_axisymmetric_pose(
        points_C_orig,
        profile_O,
        weights=valid_mask.float(),
        axis=axis,
        num_iters=int(solver_cfg.get("num_iters", 200)),
        lr=float(solver_cfg.get("lr", 1e-2)),
        huber_beta=float(solver_cfg.get("huber_beta", 0.01)),
    )
    oracle_metrics = compute_profile_batch_metrics(
        pred_profile_O=profile_O,
        target_profile_O=profile_O,
        points_C_orig=points_C_orig,
        T_pred=T_oracle,
        T_gt=T_gt,
        valid_mask=valid_mask,
        axis=axis,
        coord_scale=units_cfg["coord_scale"],
        coord_unit=units_cfg["coord_unit"],
        residual_profile_O=profile_O,
    )
    print("oracle axisymmetric metrics:")
    print(summarize_metric_dicts(oracle_metrics))
    print(f"solver diagnostics: {diagnostics}")

    if args.checkpoint is not None:
        model = build_model(cfg.model).to(device)
        load_checkpoint(args.checkpoint, model, map_location=device)
        model.eval()
        with torch.no_grad():
            outputs = model(batch["points_C"].to(device))
        print("first pred_profile_O[0, :5]:")
        print(outputs["pred_profile_O"][0, :5].detach().cpu())


if __name__ == "__main__":
    main()
