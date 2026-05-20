"""Small symmetry loss helpers kept separate for future extensions."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from fragreg.geometry.symmetry import points_to_rz


def rz_smooth_l1_loss(
    pred_points: torch.Tensor,
    target_points: torch.Tensor,
    axis: str = "z",
    beta: float = 0.01,
    reduction: str = "mean",
) -> torch.Tensor:
    pred_rz = points_to_rz(pred_points, axis=axis)
    target_rz = points_to_rz(target_points, axis=axis)
    return F.smooth_l1_loss(pred_rz, target_rz, beta=beta, reduction=reduction)

