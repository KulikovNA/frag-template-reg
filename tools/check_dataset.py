#!/usr/bin/env python
"""Print dataset statistics and inspect a few samples."""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fragreg.registry import build_dataset
from fragreg.utils import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="Path to Python config.")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--examples", type=int, default=3)
    return parser.parse_args()


def dataset_cfg_for_split(cfg, split: str) -> dict:
    key = f"{split}_dataloader"
    if key in cfg:
        return copy.deepcopy(cfg[key]["dataset"])
    dataset_cfg = copy.deepcopy(cfg.train_dataloader["dataset"])
    dataset_cfg["split"] = split
    dataset_cfg["random_sample"] = False
    return dataset_cfg


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    dataset = build_dataset(dataset_cfg_for_split(cfg, args.split))
    summary = dataset.get_summary() if hasattr(dataset, "get_summary") else {}

    print("Dataset summary")
    for key, value in summary.items():
        print(f"  {key}: {value}")

    if len(dataset) == 0:
        raise RuntimeError("Dataset contains no valid samples.")

    print("\nExamples")
    for i in range(min(args.examples, len(dataset))):
        sample = dataset[i]
        print(
            f"  #{i}: scene={sample['scene_id']} frame={sample['frame_id']} "
            f"fragment={sample['fragment_id']} shell_points={sample['num_shell_points']}"
        )
        for key in ["points_C", "points_C_orig", "points_O", "profile_O", "T_C_from_O", "gt_axis_C", "valid_mask"]:
            if key not in sample:
                continue
            value = sample[key]
            if torch.is_tensor(value):
                print(f"    {key}: shape={tuple(value.shape)} dtype={value.dtype}")
        if sample["T_C_from_O"].shape != (4, 4):
            raise RuntimeError("T_C_from_O has invalid shape.")
        if sample["points_C"].shape[-1] != 3 or sample["points_O"].shape[-1] != 3:
            raise RuntimeError("Point tensors must have 3 coordinates.")
    print("\nOK")


if __name__ == "__main__":
    main()
