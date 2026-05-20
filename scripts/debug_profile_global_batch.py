#!/usr/bin/env python
"""Inspect one global-profile batch, forward pass, axis head, and loss."""

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
from fragreg.evaluation.metrics import axis_vector_error_deg
from fragreg.registry import build_loss, build_model
from fragreg.training import load_checkpoint
from fragreg.utils import apply_units_to_loss_cfg, get_units_cfg, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="Path to Python config.")
    parser.add_argument("--checkpoint", default=None, help="Optional model checkpoint.")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--device", default=None, help="Override device, e.g. cuda or cpu.")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-points", type=int, default=None)
    return parser.parse_args()


def dataloader_cfg_for_split(cfg, split: str) -> dict:
    key = f"{split}_dataloader"
    if key in cfg:
        return copy.deepcopy(cfg[key])
    loader_cfg = copy.deepcopy(cfg.val_dataloader if "val_dataloader" in cfg else cfg.train_dataloader)
    loader_cfg["dataset"]["split"] = split
    loader_cfg["dataset"]["random_sample"] = False
    return loader_cfg


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    units_cfg = get_units_cfg(cfg)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    loader_cfg = dataloader_cfg_for_split(cfg, args.split)
    loader_cfg["num_workers"] = 0
    loader_cfg["persistent_workers"] = False
    loader_cfg["dataset"]["return_profile"] = True
    if args.batch_size is not None:
        loader_cfg["batch_size"] = args.batch_size
    if args.num_points is not None:
        loader_cfg["dataset"]["num_points"] = args.num_points
    loader = build_dataloader(loader_cfg, is_train=args.split == "train")
    batch = next(iter(loader))

    print(f"dataset samples: {len(loader.dataset)}")
    for key in ("points_C_orig", "points_C", "profile_O", "gt_axis_C", "valid_mask", "T_C_from_O"):
        value = batch[key]
        print(f"{key}: shape={tuple(value.shape)} dtype={value.dtype}")

    model = build_model(cfg.model).to(device)
    if args.checkpoint is not None:
        load_checkpoint(args.checkpoint, model, map_location=device)
        print(f"checkpoint: {args.checkpoint}")
    model.eval()

    model_batch = {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}
    with torch.no_grad():
        outputs = model(model_batch["points_C"])

    print("outputs:")
    for key, value in outputs.items():
        if isinstance(value, dict):
            shapes = {sub_key: tuple(sub_value.shape) for sub_key, sub_value in value.items()}
            print(f"  {key}: {shapes}")
        else:
            print(f"  {key}: shape={tuple(value.shape)} dtype={value.dtype}")

    print("first target profile_O[0, :5]:")
    print(model_batch["profile_O"][0, :5].detach().cpu())
    print("first pred_profile_O[0, :5]:")
    print(outputs["pred_profile_O"][0, :5].detach().cpu())

    if "pred_axis_C" in outputs:
        axis_error = axis_vector_error_deg(outputs["pred_axis_C"], model_batch["gt_axis_C"])
        print("pred_axis_C[0]:")
        print(outputs["pred_axis_C"][0].detach().cpu())
        print("gt_axis_C[0]:")
        print(model_batch["gt_axis_C"][0].detach().cpu())
        print(f"pred_axis_error_deg mean: {float(axis_error.mean().detach().cpu()):.6f}")

    loss_fn = build_loss(apply_units_to_loss_cfg(cfg.loss, units_cfg)).to(device)
    loss_dict = loss_fn(outputs, model_batch)
    print("loss:")
    for key, value in loss_dict.items():
        print(f"  {key}: {float(value.detach().cpu()):.6f}")


if __name__ == "__main__":
    main()
