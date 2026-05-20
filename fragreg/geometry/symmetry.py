"""Symmetry-aware point and pose metrics."""

from __future__ import annotations

import math

import torch


def axis_to_vector(
    axis: str,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    axis = axis.lower()
    if axis == "x":
        values = (1.0, 0.0, 0.0)
    elif axis == "y":
        values = (0.0, 1.0, 0.0)
    elif axis == "z":
        values = (0.0, 0.0, 1.0)
    else:
        raise ValueError(f"Unsupported axis {axis!r}; expected 'x', 'y' or 'z'.")
    return torch.tensor(values, dtype=dtype, device=device)


def axis_to_index(axis: str) -> int:
    axis = axis.lower()
    if axis == "x":
        return 0
    if axis == "y":
        return 1
    if axis == "z":
        return 2
    raise ValueError(f"Unsupported axis {axis!r}; expected 'x', 'y' or 'z'.")


def points_to_profile(points: torch.Tensor, axis: str = "y", eps: float = 1e-12) -> torch.Tensor:
    """Convert 3D object-frame points to axisymmetric profile coordinates.

    The returned profile is ``[radial_distance_to_axis, axial_coordinate]``.
    """

    axis = axis.lower()
    if axis == "z":
        radial = torch.sqrt((points[..., 0] ** 2 + points[..., 1] ** 2).clamp_min(eps))
        axial = points[..., 2]
    elif axis == "y":
        radial = torch.sqrt((points[..., 0] ** 2 + points[..., 2] ** 2).clamp_min(eps))
        axial = points[..., 1]
    elif axis == "x":
        radial = torch.sqrt((points[..., 1] ** 2 + points[..., 2] ** 2).clamp_min(eps))
        axial = points[..., 0]
    else:
        raise ValueError(f"Unsupported axis {axis!r}; expected 'x', 'y' or 'z'.")
    return torch.stack([radial, axial], dim=-1)


def points_to_rz(points: torch.Tensor, axis: str = "z", eps: float = 1e-12) -> torch.Tensor:
    """Backward-compatible alias for profile coordinates."""

    return points_to_profile(points, axis=axis, eps=eps)


def _ensure_batched_matrix(matrix: torch.Tensor) -> tuple[torch.Tensor, bool]:
    if matrix.ndim == 2:
        return matrix.unsqueeze(0), True
    if matrix.ndim == 3:
        return matrix, False
    raise ValueError(f"Expected matrix shape [3, 3] or [B, 3, 3], got {tuple(matrix.shape)}")


def rotation_error_deg(R_pred: torch.Tensor, R_gt: torch.Tensor) -> torch.Tensor:
    R_pred, squeeze = _ensure_batched_matrix(R_pred)
    R_gt, _ = _ensure_batched_matrix(R_gt)
    relative = R_pred @ R_gt.transpose(-2, -1)
    trace = relative.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    cos_angle = ((trace - 1.0) * 0.5).clamp(-1.0, 1.0)
    angle = torch.rad2deg(torch.acos(cos_angle))
    return angle[0] if squeeze else angle


def translation_error_m(t_pred: torch.Tensor, t_gt: torch.Tensor) -> torch.Tensor:
    error = torch.linalg.norm(t_pred - t_gt, dim=-1)
    return error


def axis_error_deg(
    R_pred: torch.Tensor,
    R_gt: torch.Tensor,
    axis_O: str | torch.Tensor | list[float] | tuple[float, float, float] = (0.0, 0.0, 1.0),
) -> torch.Tensor:
    R_pred, squeeze = _ensure_batched_matrix(R_pred)
    R_gt, _ = _ensure_batched_matrix(R_gt)
    if isinstance(axis_O, str):
        axis = axis_to_vector(axis_O, dtype=R_pred.dtype, device=R_pred.device)
    else:
        axis = torch.as_tensor(axis_O, dtype=R_pred.dtype, device=R_pred.device)
    axis = axis / axis.norm().clamp_min(1e-12)
    pred_axis = R_pred @ axis
    gt_axis = R_gt @ axis
    pred_axis = pred_axis / pred_axis.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    gt_axis = gt_axis / gt_axis.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    cos_angle = (pred_axis * gt_axis).sum(dim=-1).abs().clamp(-1.0, 1.0)
    angle = torch.acos(cos_angle) * (180.0 / math.pi)
    return angle[0] if squeeze else angle
