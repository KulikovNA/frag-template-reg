"""Dataloader builders."""

from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import DataLoader

from fragreg.registry import build_dataset


def build_dataloader(cfg: dict[str, Any], is_train: bool = False) -> DataLoader:
    cfg = dict(cfg)
    dataset_cfg = cfg.pop("dataset")
    dataset = build_dataset(dataset_cfg)
    batch_size = int(cfg.pop("batch_size", 1))
    num_workers = int(cfg.pop("num_workers", 0))
    shuffle = bool(cfg.pop("shuffle", is_train))
    pin_memory = bool(cfg.pop("pin_memory", torch.cuda.is_available()))
    drop_last = bool(cfg.pop("drop_last", is_train))
    persistent_workers = bool(cfg.pop("persistent_workers", False))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=persistent_workers and num_workers > 0,
        **cfg,
    )
