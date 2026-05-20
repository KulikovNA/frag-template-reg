"""Correspondence losses for canonical-coordinate regression."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from fragreg.geometry.symmetry import points_to_rz
from fragreg.registry import LOSSES


def _masked_mean(values: torch.Tensor, valid_mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    mask = valid_mask.to(dtype=values.dtype, device=values.device)
    while mask.ndim < values.ndim:
        mask = mask.unsqueeze(-1)
    values = values * mask
    denom = mask.expand_as(values).sum().clamp_min(eps)
    return values.sum() / denom


@LOSSES.register_module()
class CorrespondenceLoss(nn.Module):
    """SmoothL1 loss in xyz and/or symmetry-aware rz profile coordinates."""

    def __init__(
        self,
        mode: str = "xyz",
        xyz_weight: float | None = None,
        rz_weight: float | None = None,
        smooth_l1_beta: float = 0.01,
        axis: str = "z",
        coord_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.mode = mode.lower()
        if self.mode not in {"xyz", "rz", "mixed"}:
            raise ValueError(f"Unsupported loss mode {mode!r}; expected 'xyz', 'rz' or 'mixed'.")
        if self.mode == "xyz":
            self.xyz_weight = 1.0
            self.rz_weight = 0.0
        elif self.mode == "rz":
            self.xyz_weight = 0.0
            self.rz_weight = 1.0
        else:
            self.xyz_weight = 1.0 if xyz_weight is None else float(xyz_weight)
            self.rz_weight = 1.0 if rz_weight is None else float(rz_weight)
        self.smooth_l1_beta = float(smooth_l1_beta)
        self.axis = axis
        self.coord_scale = float(coord_scale)

    def forward(self, outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        pred = outputs["pred_points_O"]
        target = batch["points_O"].to(device=pred.device, dtype=pred.dtype)
        if self.coord_scale != 1.0:
            pred_for_loss = pred * self.coord_scale
            target_for_loss = target * self.coord_scale
        else:
            pred_for_loss = pred
            target_for_loss = target
        valid_mask = batch.get("valid_mask")
        if valid_mask is None:
            valid_mask = torch.ones(pred.shape[:2], dtype=torch.bool, device=pred.device)
        else:
            valid_mask = valid_mask.to(device=pred.device)

        loss_xyz_raw = F.smooth_l1_loss(
            pred_for_loss,
            target_for_loss,
            beta=self.smooth_l1_beta,
            reduction="none",
        )
        loss_xyz = _masked_mean(loss_xyz_raw, valid_mask)

        pred_rz = points_to_rz(pred_for_loss, axis=self.axis)
        target_rz = points_to_rz(target_for_loss, axis=self.axis)
        loss_rz_raw = F.smooth_l1_loss(
            pred_rz,
            target_rz,
            beta=self.smooth_l1_beta,
            reduction="none",
        )
        loss_rz = _masked_mean(loss_rz_raw, valid_mask)

        if self.mode == "xyz":
            total = loss_xyz
        elif self.mode == "rz":
            total = loss_rz
        else:
            total = self.xyz_weight * loss_xyz + self.rz_weight * loss_rz
        return {
            "loss": total,
            "loss_xyz": loss_xyz.detach(),
            "loss_rz": loss_rz.detach(),
        }
