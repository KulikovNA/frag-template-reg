"""Evaluation metrics for correspondence and pose recovery."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import torch
import torch.nn.functional as F

from fragreg.geometry.symmetry import axis_error_deg, points_to_profile, points_to_rz, rotation_error_deg, translation_error_m
from fragreg.geometry.transforms import apply_transform, invert_transform


def _masked_mean(values: torch.Tensor, mask: torch.Tensor, dim: int | tuple[int, ...] | None = None) -> torch.Tensor:
    mask = mask.to(dtype=values.dtype, device=values.device)
    while mask.ndim < values.ndim:
        mask = mask.unsqueeze(-1)
    masked = values * mask
    if dim is None:
        return masked.sum() / mask.expand_as(values).sum().clamp_min(1e-8)
    return masked.sum(dim=dim) / mask.expand_as(values).sum(dim=dim).clamp_min(1e-8)


def _masked_max(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    values = values.masked_fill(~mask.to(device=values.device), 0.0)
    return values.max(dim=1).values


def _masked_min(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    values = values.masked_fill(~mask.to(device=values.device), float("inf"))
    min_values = values.min(dim=1).values
    return torch.where(torch.isfinite(min_values), min_values, torch.zeros_like(min_values))


def _masked_span(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return _masked_max(values, mask) - _masked_min(values, mask)


def _masked_std(values: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    mean = _masked_mean(values, mask, dim=1)
    centered = (values - mean.unsqueeze(-1)).square()
    return torch.sqrt(_masked_mean(centered, mask, dim=1).clamp_min(eps))


def _masked_corr(x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    x_mean = _masked_mean(x, mask, dim=1)
    y_mean = _masked_mean(y, mask, dim=1)
    x_centered = x - x_mean.unsqueeze(-1)
    y_centered = y - y_mean.unsqueeze(-1)
    cov = _masked_mean(x_centered * y_centered, mask, dim=1)
    x_std = torch.sqrt(_masked_mean(x_centered.square(), mask, dim=1).clamp_min(eps))
    y_std = torch.sqrt(_masked_mean(y_centered.square(), mask, dim=1).clamp_min(eps))
    return (cov / (x_std * y_std).clamp_min(eps)).clamp(-1.0, 1.0)


def axis_vector_error_deg(pred_axis_C: torch.Tensor, gt_axis_C: torch.Tensor) -> torch.Tensor:
    """Axis-direction error in degrees, sign-invariant."""

    pred_axis_C = F.normalize(pred_axis_C, dim=-1)
    gt_axis_C = F.normalize(gt_axis_C.to(device=pred_axis_C.device, dtype=pred_axis_C.dtype), dim=-1)
    cos_angle = (pred_axis_C * gt_axis_C).sum(dim=-1).abs().clamp(-1.0, 1.0)
    return torch.rad2deg(torch.acos(cos_angle))


def compute_batch_metrics(
    pred_points_O: torch.Tensor,
    target_points_O: torch.Tensor,
    points_C_orig: torch.Tensor,
    T_pred: torch.Tensor,
    T_gt: torch.Tensor,
    valid_mask: torch.Tensor,
    axis: str = "z",
    coord_scale: float = 1.0,
) -> list[dict[str, float]]:
    """Compute per-sample metrics for one batch."""

    valid_mask = valid_mask.to(device=pred_points_O.device, dtype=torch.bool)
    target_points_O = target_points_O.to(device=pred_points_O.device, dtype=pred_points_O.dtype)
    points_C_orig = points_C_orig.to(device=pred_points_O.device, dtype=pred_points_O.dtype)
    T_gt = T_gt.to(device=pred_points_O.device, dtype=pred_points_O.dtype)

    coord_scale = float(coord_scale)
    coord_abs = (pred_points_O - target_points_O).abs() * coord_scale
    coord_l1 = _masked_mean(coord_abs, valid_mask, dim=(1, 2))

    pred_rz = points_to_rz(pred_points_O * coord_scale, axis=axis)
    target_rz = points_to_rz(target_points_O * coord_scale, axis=axis)
    rz_l1 = _masked_mean((pred_rz - target_rz).abs(), valid_mask, dim=(1, 2))

    transformed = apply_transform(pred_points_O, T_pred)
    residual = torch.linalg.norm(transformed - points_C_orig, dim=-1) * coord_scale
    residual_mean = _masked_mean(residual, valid_mask, dim=1)
    residual_rmse = torch.sqrt(_masked_mean(residual.square(), valid_mask, dim=1))
    residual_max = _masked_max(residual, valid_mask)

    R_pred = T_pred[:, :3, :3]
    R_gt = T_gt[:, :3, :3]
    t_pred = T_pred[:, :3, 3]
    t_gt = T_gt[:, :3, 3]
    translation_error = translation_error_m(t_pred, t_gt) * coord_scale
    rot_error = rotation_error_deg(R_pred, R_gt)
    ax_error = axis_error_deg(R_pred, R_gt, axis_O=axis)

    metrics: list[dict[str, float]] = []
    for i in range(pred_points_O.shape[0]):
        metrics.append(
            {
                "coord_l1": float(coord_l1[i].detach().cpu()),
                "rz_l1": float(rz_l1[i].detach().cpu()),
                "residual_mean": float(residual_mean[i].detach().cpu()),
                "residual_rmse": float(residual_rmse[i].detach().cpu()),
                "residual_max": float(residual_max[i].detach().cpu()),
                "translation_error": float(translation_error[i].detach().cpu()),
                "rotation_error_deg": float(rot_error[i].detach().cpu()),
                "axis_error_deg": float(ax_error[i].detach().cpu()),
            }
        )
    return metrics


def _unit_suffix(coord_unit: str) -> str:
    unit = coord_unit.strip().lower()
    return f"_{unit}" if unit and unit != "dataset" else ""


def compute_profile_batch_metrics(
    pred_profile_O: torch.Tensor,
    target_profile_O: torch.Tensor,
    points_C_orig: torch.Tensor,
    T_pred: torch.Tensor,
    T_gt: torch.Tensor,
    valid_mask: torch.Tensor,
    axis: str = "y",
    coord_scale: float = 1.0,
    coord_unit: str = "dataset",
    residual_profile_O: torch.Tensor | None = None,
) -> list[dict[str, float]]:
    """Compute profile-model metrics for one batch."""

    valid_mask = valid_mask.to(device=pred_profile_O.device, dtype=torch.bool)
    target_profile_O = target_profile_O.to(device=pred_profile_O.device, dtype=pred_profile_O.dtype)
    if residual_profile_O is None:
        residual_profile_O = pred_profile_O
    else:
        residual_profile_O = residual_profile_O.to(device=pred_profile_O.device, dtype=pred_profile_O.dtype)
    points_C_orig = points_C_orig.to(device=pred_profile_O.device, dtype=pred_profile_O.dtype)
    T_gt = T_gt.to(device=pred_profile_O.device, dtype=pred_profile_O.dtype)

    scale = float(coord_scale)
    suffix = _unit_suffix(coord_unit)
    diff = (pred_profile_O - target_profile_O).abs() * scale
    profile_r_l1 = _masked_mean(diff[..., 0], valid_mask, dim=1)
    profile_axial_l1 = _masked_mean(diff[..., 1], valid_mask, dim=1)
    profile_l1 = _masked_mean(diff, valid_mask, dim=(1, 2))
    signed_axial_error = (pred_profile_O[..., 1] - target_profile_O[..., 1]) * scale
    axial_error_mean = _masked_mean(signed_axial_error, valid_mask, dim=1)
    axial_error_std = _masked_std(signed_axial_error, valid_mask)
    axial_corr = _masked_corr(pred_profile_O[..., 1], target_profile_O[..., 1], valid_mask)
    gt_axial_span = _masked_span(target_profile_O[..., 1], valid_mask) * scale
    pred_axial_span = _masked_span(pred_profile_O[..., 1], valid_mask) * scale
    radial_span = _masked_span(target_profile_O[..., 0], valid_mask) * scale
    num_points = valid_mask.sum(dim=1)

    T_O_from_C = invert_transform(T_pred)
    points_O_from_pose = apply_transform(points_C_orig, T_O_from_C)
    profile_from_pose = points_to_profile(points_O_from_pose, axis=axis)
    residual_vec = (profile_from_pose - residual_profile_O) * scale
    residual = torch.linalg.norm(residual_vec, dim=-1)
    residual_mean = _masked_mean(residual, valid_mask, dim=1)
    residual_rmse = torch.sqrt(_masked_mean(residual.square(), valid_mask, dim=1))
    residual_max = _masked_max(residual, valid_mask)

    R_pred = T_pred[:, :3, :3]
    R_gt = T_gt[:, :3, :3]
    t_pred = T_pred[:, :3, 3]
    t_gt = T_gt[:, :3, 3]
    translation_error = translation_error_m(t_pred, t_gt) * scale
    rot_error = rotation_error_deg(R_pred, R_gt)
    ax_error = axis_error_deg(R_pred, R_gt, axis_O=axis)

    metrics: list[dict[str, float]] = []
    for i in range(pred_profile_O.shape[0]):
        metrics.append(
            {
                f"profile_r_l1{suffix}": float(profile_r_l1[i].detach().cpu()),
                f"profile_axial_l1{suffix}": float(profile_axial_l1[i].detach().cpu()),
                f"profile_l1{suffix}": float(profile_l1[i].detach().cpu()),
                f"profile_residual_mean{suffix}": float(residual_mean[i].detach().cpu()),
                f"profile_residual_rmse{suffix}": float(residual_rmse[i].detach().cpu()),
                f"profile_residual_max{suffix}": float(residual_max[i].detach().cpu()),
                f"translation_error{suffix}": float(translation_error[i].detach().cpu()),
                f"axial_error_mean{suffix}": float(axial_error_mean[i].detach().cpu()),
                f"axial_error_std{suffix}": float(axial_error_std[i].detach().cpu()),
                "axial_corr": float(axial_corr[i].detach().cpu()),
                f"gt_axial_span{suffix}": float(gt_axial_span[i].detach().cpu()),
                f"pred_axial_span{suffix}": float(pred_axial_span[i].detach().cpu()),
                f"radial_span{suffix}": float(radial_span[i].detach().cpu()),
                "num_points": float(num_points[i].detach().cpu()),
                "axis_error_deg": float(ax_error[i].detach().cpu()),
                "rotation_error_deg_diagnostic": float(rot_error[i].detach().cpu()),
            }
        )
    return metrics


def summarize_metric_dicts(metrics: list[dict[str, Any]]) -> dict[str, float]:
    if not metrics:
        return {}
    values: dict[str, list[float]] = defaultdict(list)
    metadata_keys = {"frame_id", "fragment_id", "sample_index"}
    for item in metrics:
        for key, value in item.items():
            if key in metadata_keys:
                continue
            if isinstance(value, (int, float)):
                values[key].append(float(value))
    return {key: float(torch.tensor(vals, dtype=torch.float64).mean().item()) for key, vals in values.items()}


def add_axial_span_groups(
    metrics: list[dict[str, Any]],
    span_key: str = "gt_axial_span_mm",
    group_key: str = "axial_span_group",
) -> list[dict[str, Any]]:
    """Assign small/medium/large labels by tertiles of GT axial span."""

    if not metrics or any(span_key not in item for item in metrics):
        return metrics
    spans = torch.tensor([float(item[span_key]) for item in metrics], dtype=torch.float64)
    if spans.numel() < 3:
        for item in metrics:
            item[group_key] = "medium"
        return metrics
    q1 = torch.quantile(spans, 1.0 / 3.0).item()
    q2 = torch.quantile(spans, 2.0 / 3.0).item()
    for item in metrics:
        span = float(item[span_key])
        if span <= q1:
            label = "small"
        elif span <= q2:
            label = "medium"
        else:
            label = "large"
        item[group_key] = label
    return metrics


def summarize_metric_groups(
    metrics: list[dict[str, Any]],
    group_key: str = "axial_span_group",
) -> dict[str, float]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in metrics:
        label = item.get(group_key)
        if isinstance(label, str):
            grouped[label].append(item)
    summary: dict[str, float] = {}
    for label in ("small", "medium", "large"):
        items = grouped.get(label, [])
        if not items:
            continue
        group_summary = summarize_metric_dicts(items)
        summary[f"{label}/num_samples"] = float(len(items))
        for key, value in group_summary.items():
            summary[f"{label}/{key}"] = value
    return summary
