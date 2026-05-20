#!/usr/bin/env python
"""Inspect one dataloader batch and run oracle Kabsch on it."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fragreg.data import build_dataloader
from fragreg.geometry.kabsch import batch_kabsch
from fragreg.geometry.transforms import apply_transform
from fragreg.utils import get_units_cfg, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="Path to Python config.")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    units_cfg = get_units_cfg(cfg)
    loader_cfg = cfg[f"{args.split}_dataloader"] if f"{args.split}_dataloader" in cfg else cfg.train_dataloader
    loader_cfg["num_workers"] = 0
    loader_cfg["persistent_workers"] = False
    loader = build_dataloader(loader_cfg, is_train=args.split == "train")
    batch = next(iter(loader))

    print(f"dataset samples: {len(loader.dataset)}")
    for key, value in batch.items():
        if torch.is_tensor(value):
            print(f"{key}: shape={tuple(value.shape)} dtype={value.dtype}")
        else:
            print(f"{key}: {value}")

    device = torch.device(args.device)
    points_O = batch["points_O"].to(device)
    points_C_orig = batch["points_C_orig"].to(device)
    valid_mask = batch["valid_mask"].to(device)
    T_oracle = batch_kabsch(points_O, points_C_orig, weights=valid_mask.float())
    transformed = apply_transform(points_O, T_oracle)
    residual = torch.linalg.norm(transformed - points_C_orig, dim=-1) * units_cfg["coord_scale"]
    residual = residual[valid_mask]
    print(f"coordinate unit: {units_cfg['coord_unit']} (scale={units_cfg['coord_scale']})")
    print(f"oracle residual mean: {residual.mean().item():.8f} {units_cfg['coord_unit']}")
    print(f"oracle residual rmse: {torch.sqrt((residual.square()).mean()).item():.8f} {units_cfg['coord_unit']}")
    print(f"oracle residual max: {residual.max().item():.8f} {units_cfg['coord_unit']}")
    print("first points_C_orig[0, :5]:")
    print(batch["points_C_orig"][0, :5])
    print("first points_O[0, :5]:")
    print(batch["points_O"][0, :5])


if __name__ == "__main__":
    main()
