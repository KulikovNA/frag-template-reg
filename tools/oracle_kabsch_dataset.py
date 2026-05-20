#!/usr/bin/env python
"""Validate GT correspondences by estimating pose from GT points_O."""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fragreg.data import build_dataloader
from fragreg.evaluation.metrics import compute_batch_metrics, summarize_metric_dicts
from fragreg.geometry.kabsch import batch_kabsch
from fragreg.utils import get_units_cfg, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="Path to Python config.")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--device", default="cpu")
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
    return loader_cfg


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    units_cfg = get_units_cfg(cfg)
    device = torch.device(args.device)
    loader = build_dataloader(dataloader_cfg_for_split(cfg, args.split), is_train=False)
    if len(loader.dataset) == 0:
        raise RuntimeError("Dataset contains no valid samples.")

    all_metrics = []
    for batch in tqdm(loader, desc=f"oracle {args.split}"):
        points_O = batch["points_O"].to(device)
        points_C_orig = batch["points_C_orig"].to(device)
        valid_mask = batch["valid_mask"].to(device)
        T_gt = batch["T_C_from_O"].to(device)
        T_oracle = batch_kabsch(points_O, points_C_orig, weights=valid_mask.float())
        batch_metrics = compute_batch_metrics(
            pred_points_O=points_O,
            target_points_O=points_O,
            points_C_orig=points_C_orig,
            T_pred=T_oracle,
            T_gt=T_gt,
            valid_mask=valid_mask,
            axis=cfg.loss.get("axis", "z"),
            coord_scale=units_cfg["coord_scale"],
        )
        all_metrics.extend(batch_metrics)

    summary = summarize_metric_dicts(all_metrics)
    summary["num_samples"] = len(all_metrics)
    summary["coord_scale"] = units_cfg["coord_scale"]
    summary["coord_unit"] = units_cfg["coord_unit"]
    for key in sorted(summary):
        print(f"{key}: {summary[key]}")


if __name__ == "__main__":
    main()
