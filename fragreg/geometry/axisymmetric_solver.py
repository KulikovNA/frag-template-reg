"""Axisymmetric pose solver for profile-coordinate registration."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from fragreg.geometry.symmetry import axis_to_index, axis_to_vector, points_to_profile


def normalize_quaternion(quaternion: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return quaternion / quaternion.norm(dim=-1, keepdim=True).clamp_min(eps)


def quaternion_to_matrix(quaternion: torch.Tensor) -> torch.Tensor:
    """Convert quaternions ``[w, x, y, z]`` to rotation matrices."""

    q = normalize_quaternion(quaternion)
    w, x, y, z = q.unbind(dim=-1)
    two = 2.0
    matrix = torch.stack(
        [
            1 - two * (y * y + z * z),
            two * (x * y - z * w),
            two * (x * z + y * w),
            two * (x * y + z * w),
            1 - two * (x * x + z * z),
            two * (y * z - x * w),
            two * (x * z - y * w),
            two * (y * z + x * w),
            1 - two * (x * x + y * y),
        ],
        dim=-1,
    )
    return matrix.reshape(q.shape[:-1] + (3, 3))


def _matrix_to_quaternion_single(matrix: torch.Tensor) -> torch.Tensor:
    """Small no-grad matrix-to-quaternion helper for initialization."""

    m = matrix.detach()
    trace = (m[0, 0] + m[1, 1] + m[2, 2]).item()
    if trace > 0.0:
        s = torch.sqrt((m[0, 0] + m[1, 1] + m[2, 2] + 1.0).clamp_min(1e-12)) * 2.0
        qw = 0.25 * s
        qx = (m[2, 1] - m[1, 2]) / s
        qy = (m[0, 2] - m[2, 0]) / s
        qz = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0].item() > m[1, 1].item() and m[0, 0].item() > m[2, 2].item():
        s = torch.sqrt((1.0 + m[0, 0] - m[1, 1] - m[2, 2]).clamp_min(1e-12)) * 2.0
        qw = (m[2, 1] - m[1, 2]) / s
        qx = 0.25 * s
        qy = (m[0, 1] + m[1, 0]) / s
        qz = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1].item() > m[2, 2].item():
        s = torch.sqrt((1.0 + m[1, 1] - m[0, 0] - m[2, 2]).clamp_min(1e-12)) * 2.0
        qw = (m[0, 2] - m[2, 0]) / s
        qx = (m[0, 1] + m[1, 0]) / s
        qy = 0.25 * s
        qz = (m[1, 2] + m[2, 1]) / s
    else:
        s = torch.sqrt((1.0 + m[2, 2] - m[0, 0] - m[1, 1]).clamp_min(1e-12)) * 2.0
        qw = (m[1, 0] - m[0, 1]) / s
        qx = (m[0, 2] + m[2, 0]) / s
        qy = (m[1, 2] + m[2, 1]) / s
        qz = 0.25 * s
    quat = torch.stack([qw, qx, qy, qz]).to(dtype=matrix.dtype, device=matrix.device)
    return normalize_quaternion(quat)


def _ensure_batched(points: torch.Tensor, channels: int) -> tuple[torch.Tensor, bool]:
    if points.ndim == 2:
        if points.shape[-1] != channels:
            raise ValueError(f"Expected last dim {channels}, got {points.shape[-1]}")
        return points.unsqueeze(0), True
    if points.ndim == 3:
        if points.shape[-1] != channels:
            raise ValueError(f"Expected last dim {channels}, got {points.shape[-1]}")
        return points, False
    raise ValueError(f"Expected shape [N, {channels}] or [B, N, {channels}], got {tuple(points.shape)}")


def _weighted_mean(values: torch.Tensor, weights: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return (values * weights).sum() / weights.sum().clamp_min(eps)


def _weighted_lstsq(A: torch.Tensor, y: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    sqrt_w = weights.clamp_min(0.0).sqrt().unsqueeze(-1)
    solution = torch.linalg.lstsq(A * sqrt_w, y.unsqueeze(-1) * sqrt_w).solution
    return solution.squeeze(-1)


def _make_rotation_from_axis(axis_C: torch.Tensor, axis: str) -> torch.Tensor:
    axis_idx = axis_to_index(axis)
    a = axis_C / axis_C.norm().clamp_min(1e-12)
    candidates = torch.eye(3, dtype=a.dtype, device=a.device)
    dots = (candidates @ a).abs()
    tmp = candidates[dots.argmin()]

    if axis_idx == 0:
        u = torch.linalg.cross(tmp, a)
        u = u / u.norm().clamp_min(1e-12)
        v = torch.linalg.cross(a, u)
        columns = [a, u, v]
    elif axis_idx == 1:
        u = torch.linalg.cross(tmp, a)
        u = u / u.norm().clamp_min(1e-12)
        v = torch.linalg.cross(u, a)
        columns = [u, a, v]
    else:
        u = torch.linalg.cross(tmp, a)
        u = u / u.norm().clamp_min(1e-12)
        v = torch.linalg.cross(a, u)
        columns = [u, v, a]

    R = torch.stack(columns, dim=1)
    if torch.det(R) < 0:
        flip_col = 1 if axis_idx == 0 else 0
        R[:, flip_col] = -R[:, flip_col]
    return R


def _auto_init_pose(
    points_C: torch.Tensor,
    profile_O: torch.Tensor,
    weights: torch.Tensor,
    axis: str,
    init_axis_C: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    axis_idx = axis_to_index(axis)
    axial = profile_O[:, 1]
    if init_axis_C is None:
        ones = torch.ones(points_C.shape[0], 1, dtype=points_C.dtype, device=points_C.device)
        A = torch.cat([points_C, -ones], dim=1)
        sol = _weighted_lstsq(A, axial, weights)
        axis_raw = sol[:3]
        axis_norm = axis_raw.norm()
        if axis_norm < 1e-6:
            axis_C = axis_to_vector(axis, dtype=points_C.dtype, device=points_C.device)
            b = torch.zeros((), dtype=points_C.dtype, device=points_C.device)
        else:
            axis_C = axis_raw / axis_norm
            b = sol[3] / axis_norm
    else:
        axis_raw = init_axis_C.to(device=points_C.device, dtype=points_C.dtype)
        axis_norm = axis_raw.norm()
        if axis_norm < 1e-6:
            axis_C = axis_to_vector(axis, dtype=points_C.dtype, device=points_C.device)
        else:
            axis_C = axis_raw / axis_norm
        b = _weighted_mean(points_C @ axis_C - axial, weights)

    R = _make_rotation_from_axis(axis_C, axis)
    non_axis_cols = [idx for idx in range(3) if idx != axis_idx]
    perp_basis = R[:, non_axis_cols]
    projected = points_C @ perp_basis
    radial = profile_O[:, 0].clamp_min(0.0)
    circle_A = torch.cat(
        [-2.0 * projected, torch.ones(projected.shape[0], 1, dtype=points_C.dtype, device=points_C.device)],
        dim=1,
    )
    circle_y = radial.square() - projected.square().sum(dim=1)
    circle_sol = _weighted_lstsq(circle_A, circle_y, weights)
    center_2d = circle_sol[:2]
    t = axis_C * b + perp_basis @ center_2d
    return R, t


def _pose_loss(
    points_C: torch.Tensor,
    profile_O: torch.Tensor,
    weights: torch.Tensor,
    quaternion: torch.Tensor,
    translation: torch.Tensor,
    axis: str,
    huber_beta: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    R = quaternion_to_matrix(quaternion)
    q_O = (points_C - translation.unsqueeze(0)) @ R
    profile_from_pose = points_to_profile(q_O, axis=axis)
    diff = profile_from_pose - profile_O
    raw = F.smooth_l1_loss(diff, torch.zeros_like(diff), beta=huber_beta, reduction="none")
    radial_loss = _weighted_mean(raw[:, 0], weights)
    axial_loss = _weighted_mean(raw[:, 1], weights)
    return radial_loss + axial_loss, diff, profile_from_pose


def _solve_single(
    points_C: torch.Tensor,
    profile_O: torch.Tensor,
    weights: torch.Tensor,
    axis: str,
    init: torch.Tensor | None,
    init_axis_C: torch.Tensor | None,
    num_iters: int,
    lr: float,
    huber_beta: float,
) -> tuple[torch.Tensor, dict[str, Any]]:
    valid = weights > 0
    if valid.sum() >= 4:
        points_C_fit = points_C[valid]
        profile_fit = profile_O[valid]
        weights_fit = weights[valid]
    else:
        points_C_fit = points_C
        profile_fit = profile_O
        weights_fit = torch.ones_like(weights)

    if init is not None:
        R_init = init[:3, :3].to(device=points_C.device, dtype=points_C.dtype)
        t_init = init[:3, 3].to(device=points_C.device, dtype=points_C.dtype)
        init_mode = "given"
    else:
        R_init, t_init = _auto_init_pose(points_C_fit, profile_fit, weights_fit, axis, init_axis_C=init_axis_C)
        init_mode = "axis" if init_axis_C is not None else "auto"

    with torch.no_grad():
        q_init = _matrix_to_quaternion_single(R_init)
    quaternion = torch.nn.Parameter(q_init.clone())
    translation = torch.nn.Parameter(t_init.clone())
    optimizer = torch.optim.Adam([quaternion, translation], lr=float(lr))

    final_loss = None
    with torch.enable_grad():
        for _ in range(int(num_iters)):
            optimizer.zero_grad(set_to_none=True)
            loss, _, _ = _pose_loss(
                points_C_fit,
                profile_fit,
                weights_fit,
                quaternion,
                translation,
                axis=axis,
                huber_beta=huber_beta,
            )
            loss.backward()
            optimizer.step()
            final_loss = loss.detach()

    with torch.no_grad():
        R = quaternion_to_matrix(quaternion.detach())
        t = translation.detach()
        loss, diff, _ = _pose_loss(
            points_C_fit,
            profile_fit,
            weights_fit,
            quaternion.detach(),
            t,
            axis=axis,
            huber_beta=huber_beta,
        )
        radial_rmse = torch.sqrt(_weighted_mean(diff[:, 0].square(), weights_fit))
        axial_rmse = torch.sqrt(_weighted_mean(diff[:, 1].square(), weights_fit))
        profile_rmse = torch.sqrt(_weighted_mean(diff.square().sum(dim=-1), weights_fit))
        transform = torch.eye(4, dtype=points_C.dtype, device=points_C.device)
        transform[:3, :3] = R
        transform[:3, 3] = t
        diagnostics = {
            "success": True,
            "final_loss": float((final_loss if final_loss is not None else loss).detach().cpu()),
            "radial_rmse": float(radial_rmse.detach().cpu()),
            "axial_rmse": float(axial_rmse.detach().cpu()),
            "profile_rmse": float(profile_rmse.detach().cpu()),
            "num_iters": int(num_iters),
            "init_mode": init_mode,
        }
    return transform, diagnostics


def estimate_axisymmetric_pose(
    points_C: torch.Tensor,
    profile_O: torch.Tensor,
    weights: torch.Tensor | None = None,
    axis: str = "y",
    init: torch.Tensor | None = None,
    init_axis_C: torch.Tensor | None = None,
    num_iters: int = 200,
    lr: float = 1e-2,
    huber_beta: float = 0.01,
    return_diagnostics: bool = True,
) -> tuple[torch.Tensor, dict[str, Any] | list[dict[str, Any]]]:
    """Estimate ``T_C_from_O`` from camera points and object profile coordinates."""

    points_C, squeeze = _ensure_batched(points_C, channels=3)
    profile_O, profile_squeeze = _ensure_batched(profile_O, channels=2)
    if squeeze != profile_squeeze or points_C.shape[:2] != profile_O.shape[:2]:
        raise ValueError(f"points_C and profile_O batch/point shapes differ: {points_C.shape}, {profile_O.shape}")

    batch_size, num_points, _ = points_C.shape
    if weights is None:
        weights = points_C.new_ones(batch_size, num_points)
    elif weights.ndim == 1:
        weights = weights.unsqueeze(0)
    weights = weights.to(device=points_C.device, dtype=points_C.dtype)
    if weights.shape != (batch_size, num_points):
        raise ValueError(f"weights must have shape [B, N], got {tuple(weights.shape)}")

    if init is not None:
        if init.ndim == 2:
            init = init.unsqueeze(0)
        init = init.to(device=points_C.device, dtype=points_C.dtype)
        if init.shape != (batch_size, 4, 4):
            raise ValueError(f"init must have shape [B, 4, 4], got {tuple(init.shape)}")
    if init_axis_C is not None:
        if init_axis_C.ndim == 1:
            init_axis_C = init_axis_C.unsqueeze(0)
        init_axis_C = init_axis_C.to(device=points_C.device, dtype=points_C.dtype)
        if init_axis_C.shape != (batch_size, 3):
            raise ValueError(f"init_axis_C must have shape [B, 3], got {tuple(init_axis_C.shape)}")

    transforms = []
    diagnostics = []
    for batch_idx in range(batch_size):
        T_i, diag_i = _solve_single(
            points_C=points_C[batch_idx],
            profile_O=profile_O[batch_idx],
            weights=weights[batch_idx],
            axis=axis,
            init=init[batch_idx] if init is not None else None,
            init_axis_C=init_axis_C[batch_idx] if init_axis_C is not None and init is None else None,
            num_iters=num_iters,
            lr=lr,
            huber_beta=huber_beta,
        )
        transforms.append(T_i)
        diagnostics.append(diag_i)

    stacked = torch.stack(transforms, dim=0)
    if squeeze:
        return stacked[0], diagnostics[0] if return_diagnostics else {}
    if not return_diagnostics:
        return stacked, {}
    summary: dict[str, Any] = {"per_sample": diagnostics}
    for key in ("final_loss", "radial_rmse", "axial_rmse", "profile_rmse"):
        summary[key] = float(torch.tensor([item[key] for item in diagnostics], dtype=torch.float64).mean().item())
    summary["success"] = all(item.get("success", False) for item in diagnostics)
    summary["num_iters"] = int(num_iters)
    summary["init_mode"] = diagnostics[0]["init_mode"] if diagnostics else "none"
    return stacked, summary
