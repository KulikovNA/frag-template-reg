"""Rigid transform helpers using column-vector transform matrices."""

from __future__ import annotations

import torch


def apply_transform(points: torch.Tensor, transform: torch.Tensor) -> torch.Tensor:
    """Apply T to points with shape [N, 3] or [B, N, 3]."""

    if points.ndim == 2:
        rotation = transform[:3, :3]
        translation = transform[:3, 3]
        return points @ rotation.transpose(0, 1) + translation
    if points.ndim == 3:
        rotation = transform[:, :3, :3]
        translation = transform[:, :3, 3]
        return points @ rotation.transpose(1, 2) + translation.unsqueeze(1)
    raise ValueError(f"Expected points with shape [N, 3] or [B, N, 3], got {tuple(points.shape)}")


def invert_transform(transform: torch.Tensor) -> torch.Tensor:
    if transform.ndim == 2:
        rotation = transform[:3, :3]
        translation = transform[:3, 3]
        inv = torch.eye(4, dtype=transform.dtype, device=transform.device)
        inv[:3, :3] = rotation.transpose(0, 1)
        inv[:3, 3] = -(rotation.transpose(0, 1) @ translation)
        return inv
    rotation = transform[:, :3, :3]
    translation = transform[:, :3, 3]
    inv = torch.eye(4, dtype=transform.dtype, device=transform.device).unsqueeze(0).repeat(transform.shape[0], 1, 1)
    inv[:, :3, :3] = rotation.transpose(1, 2)
    inv[:, :3, 3] = -(rotation.transpose(1, 2) @ translation.unsqueeze(-1)).squeeze(-1)
    return inv


def compose_transform(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a @ b

