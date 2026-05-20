"""Profile-coordinate losses for axisymmetric registration."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from fragreg.geometry.symmetry import axis_to_vector
from fragreg.registry import LOSSES


def _masked_mean(values: torch.Tensor, valid_mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    mask = valid_mask.to(dtype=values.dtype, device=values.device)
    while mask.ndim < values.ndim:
        mask = mask.unsqueeze(-1)
    values = values * mask
    denom = mask.expand_as(values).sum().clamp_min(eps)
    return values.sum() / denom


def _masked_mean_per_sample(values: torch.Tensor, valid_mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    mask = valid_mask.to(dtype=values.dtype, device=values.device)
    return (values * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(eps)


def _masked_std_per_sample(values: torch.Tensor, valid_mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    mean = _masked_mean_per_sample(values, valid_mask, eps=eps)
    centered = (values - mean.unsqueeze(-1)).square()
    return torch.sqrt(_masked_mean_per_sample(centered, valid_mask, eps=eps).clamp_min(eps))


def _masked_span_per_sample(values: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    min_values = values.masked_fill(~valid_mask, float("inf")).min(dim=1).values
    max_values = values.masked_fill(~valid_mask, -float("inf")).max(dim=1).values
    span = max_values - min_values
    return torch.where(torch.isfinite(span), span, torch.zeros_like(span))


def _mean_smooth_l1_to_zero(values: torch.Tensor, beta: float) -> torch.Tensor:
    return F.smooth_l1_loss(values, torch.zeros_like(values), beta=beta, reduction="mean")


@LOSSES.register_module()
class ProfileLoss(nn.Module):
    """SmoothL1 loss between predicted and target profile coordinates."""

    def __init__(
        self,
        smooth_l1_beta: float = 10.0,
        coord_scale: float = 1.0,
        axis: str = "y",
        radial_weight: float = 1.0,
        axial_weight: float = 1.0,
        axial_mean_weight: float = 0.0,
        axial_std_weight: float = 0.0,
        axial_range_weight: float = 0.0,
        axial_pairwise_weight: float = 0.0,
        axis_loss_weight: float = 0.0,
    ) -> None:
        super().__init__()
        self.smooth_l1_beta = float(smooth_l1_beta)
        self.coord_scale = float(coord_scale)
        self.axis = axis
        self.radial_weight = float(radial_weight)
        self.axial_weight = float(axial_weight)
        self.axial_mean_weight = float(axial_mean_weight)
        self.axial_std_weight = float(axial_std_weight)
        self.axial_range_weight = float(axial_range_weight)
        self.axial_pairwise_weight = float(axial_pairwise_weight)
        self.axis_loss_weight = float(axis_loss_weight)

    def forward(self, outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        pred = outputs["pred_profile_O"]
        target = batch["profile_O"].to(device=pred.device, dtype=pred.dtype)
        valid_mask = batch.get("valid_mask")
        if valid_mask is None:
            valid_mask = torch.ones(pred.shape[:2], dtype=torch.bool, device=pred.device)
        else:
            valid_mask = valid_mask.to(device=pred.device)

        pred_for_loss = pred * self.coord_scale
        target_for_loss = target * self.coord_scale
        raw = F.smooth_l1_loss(
            pred_for_loss,
            target_for_loss,
            beta=self.smooth_l1_beta,
            reduction="none",
        )
        loss_profile = _masked_mean(raw, valid_mask)
        loss_r = _masked_mean(raw[..., 0], valid_mask)
        loss_axial = _masked_mean(raw[..., 1], valid_mask)
        weighted_profile = self.radial_weight * loss_r + self.axial_weight * loss_axial

        pred_axial = pred_for_loss[..., 1]
        target_axial = target_for_loss[..., 1]
        axial_error = pred_axial - target_axial

        target_axial_mean = _masked_mean_per_sample(target_axial, valid_mask)
        target_axial_std = _masked_std_per_sample(target_axial, valid_mask)
        target_axial_range = _masked_span_per_sample(target_axial, valid_mask)
        pred_axial_stats = outputs.get("pred_axial_stats")
        if isinstance(pred_axial_stats, dict):
            pred_mean = pred_axial_stats["mean"].to(device=pred.device, dtype=pred.dtype) * self.coord_scale
            pred_std = pred_axial_stats["std"].to(device=pred.device, dtype=pred.dtype) * self.coord_scale
            pred_range = pred_axial_stats["span"].to(device=pred.device, dtype=pred.dtype) * self.coord_scale
            axial_mean = F.smooth_l1_loss(
                pred_mean,
                target_axial_mean,
                beta=self.smooth_l1_beta,
                reduction="mean",
            )
            axial_std = F.smooth_l1_loss(
                pred_std,
                target_axial_std,
                beta=self.smooth_l1_beta,
                reduction="mean",
            )
            axial_range = F.smooth_l1_loss(
                pred_range,
                target_axial_range,
                beta=self.smooth_l1_beta,
                reduction="mean",
            )
        else:
            axial_mean = _mean_smooth_l1_to_zero(
                _masked_mean_per_sample(axial_error, valid_mask),
                beta=self.smooth_l1_beta,
            )
            axial_std = _mean_smooth_l1_to_zero(
                _masked_std_per_sample(pred_axial, valid_mask) - target_axial_std,
                beta=self.smooth_l1_beta,
            )
            axial_range = _mean_smooth_l1_to_zero(
                _masked_span_per_sample(pred_axial, valid_mask) - target_axial_range,
                beta=self.smooth_l1_beta,
            )

        if self.axial_pairwise_weight != 0.0:
            pred_pairwise = pred_axial.unsqueeze(2) - pred_axial.unsqueeze(1)
            target_pairwise = target_axial.unsqueeze(2) - target_axial.unsqueeze(1)
            pairwise_mask = valid_mask.unsqueeze(2) & valid_mask.unsqueeze(1)
            pairwise_raw = F.smooth_l1_loss(
                pred_pairwise,
                target_pairwise,
                beta=self.smooth_l1_beta,
                reduction="none",
            )
            axial_pairwise = _masked_mean(pairwise_raw, pairwise_mask)
        else:
            axial_pairwise = pred.new_zeros(())

        if self.axis_loss_weight != 0.0 and "pred_axis_C" in outputs:
            pred_axis = F.normalize(outputs["pred_axis_C"], dim=-1)
            if "gt_axis_C" in batch:
                target_axis = batch["gt_axis_C"].to(device=pred.device, dtype=pred.dtype)
                target_axis = F.normalize(target_axis, dim=-1)
            else:
                T_C_from_O = batch["T_C_from_O"].to(device=pred.device, dtype=pred.dtype)
                axis_O = axis_to_vector(self.axis, dtype=pred.dtype, device=pred.device)
                target_axis = F.normalize(T_C_from_O[:, :3, :3] @ axis_O, dim=-1)
            axis_alignment = (pred_axis * target_axis).sum(dim=-1).abs().clamp(0.0, 1.0)
            axis_loss = (1.0 - axis_alignment).mean()
        else:
            axis_loss = pred.new_zeros(())

        total = (
            weighted_profile
            + self.axial_mean_weight * axial_mean
            + self.axial_std_weight * axial_std
            + self.axial_range_weight * axial_range
            + self.axial_pairwise_weight * axial_pairwise
            + self.axis_loss_weight * axis_loss
        )
        return {
            "loss": total,
            "loss_profile": loss_profile.detach(),
            "loss_r": loss_r.detach(),
            "loss_axial": loss_axial.detach(),
            "loss_profile_weighted": weighted_profile.detach(),
            "loss_axial_mean": axial_mean.detach(),
            "loss_axial_std": axial_std.detach(),
            "loss_axial_range": axial_range.detach(),
            "loss_axial_pairwise": axial_pairwise.detach(),
            "loss_axis": axis_loss.detach(),
        }
