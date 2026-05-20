"""Torch Kabsch rigid alignment."""

from __future__ import annotations

import torch


def _ensure_batched(points: torch.Tensor) -> tuple[torch.Tensor, bool]:
    if points.ndim == 2:
        return points.unsqueeze(0), True
    if points.ndim == 3:
        return points, False
    raise ValueError(f"Expected points with shape [N, 3] or [B, N, 3], got {tuple(points.shape)}")


def batch_kabsch(
    src: torch.Tensor,
    dst: torch.Tensor,
    weights: torch.Tensor | None = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Estimate T_dst_from_src for batched correspondences.

    The returned transform follows column-vector convention:
    ``dst_col = R @ src_col + t``. For row tensors use
    ``dst = src @ R.transpose(-1, -2) + t``.
    """

    src, squeeze_src = _ensure_batched(src)
    dst, squeeze_dst = _ensure_batched(dst)
    if squeeze_src != squeeze_dst and src.shape[0] != dst.shape[0]:
        raise ValueError("src and dst must either both be batched or broadcastable in batch.")
    if src.shape != dst.shape or src.shape[-1] != 3:
        raise ValueError(f"src and dst must both have shape [B, N, 3], got {src.shape} and {dst.shape}")

    batch_size, num_points, _ = src.shape
    if weights is None:
        weights = src.new_ones(batch_size, num_points)
    elif weights.ndim == 1:
        weights = weights.unsqueeze(0)
    weights = weights.to(dtype=src.dtype, device=src.device)
    if weights.shape != (batch_size, num_points):
        raise ValueError(f"weights must have shape [B, N], got {tuple(weights.shape)}")

    weight_sum = weights.sum(dim=1, keepdim=True).clamp_min(eps)
    normalized_weights = weights / weight_sum
    src_centroid = (src * normalized_weights.unsqueeze(-1)).sum(dim=1)
    dst_centroid = (dst * normalized_weights.unsqueeze(-1)).sum(dim=1)

    src_centered = src - src_centroid.unsqueeze(1)
    dst_centered = dst - dst_centroid.unsqueeze(1)
    covariance = src_centered.transpose(1, 2) @ (dst_centered * weights.unsqueeze(-1))

    u, _, vh = torch.linalg.svd(covariance, full_matrices=False)
    v = vh.transpose(-2, -1)
    det = torch.det(v @ u.transpose(-2, -1))
    diag = torch.ones(batch_size, 3, dtype=src.dtype, device=src.device)
    diag[:, -1] = torch.where(det < 0, -1.0, 1.0)
    correction = torch.diag_embed(diag)
    rotation = v @ correction @ u.transpose(-2, -1)
    src_centroid_transformed = torch.bmm(src_centroid.unsqueeze(1), rotation.transpose(1, 2)).squeeze(1)
    translation = dst_centroid - src_centroid_transformed

    transform = torch.eye(4, dtype=src.dtype, device=src.device).unsqueeze(0).repeat(batch_size, 1, 1)
    transform[:, :3, :3] = rotation
    transform[:, :3, 3] = translation
    return transform


def estimate_rigid_transform_kabsch(
    src: torch.Tensor,
    dst: torch.Tensor,
    weights: torch.Tensor | None = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Estimate a single or batched rigid transform with Kabsch."""

    was_unbatched = src.ndim == 2
    transform = batch_kabsch(src, dst, weights=weights, eps=eps)
    return transform[0] if was_unbatched else transform
