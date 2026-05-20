"""Point cloud sampling and normalization utilities."""

from __future__ import annotations

import numpy as np


def normalize_points(points: np.ndarray, eps: float = 1e-8) -> tuple[np.ndarray, np.ndarray, float]:
    """Center and scale points by their maximum radius."""

    centroid = points.mean(axis=0, keepdims=True).astype(np.float32)
    centered = points.astype(np.float32) - centroid
    scale = float(np.linalg.norm(centered, axis=1).max())
    scale = max(scale, eps)
    return centered / scale, centroid.reshape(3), scale


def sample_or_pad_indices(
    num_available: int,
    num_points: int,
    random_sample: bool = True,
    repeat_if_less: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Return indices into available points and a validity mask of length num_points."""

    if num_available <= 0:
        raise ValueError("num_available must be positive.")
    if num_points <= 0:
        raise ValueError("num_points must be positive.")

    if num_available >= num_points:
        if random_sample:
            indices = np.random.choice(num_available, size=num_points, replace=False)
        else:
            indices = np.linspace(0, num_available - 1, num_points, dtype=np.int64)
        valid = np.ones(num_points, dtype=bool)
        return indices.astype(np.int64), valid

    if repeat_if_less:
        if random_sample:
            extra = np.random.choice(num_available, size=num_points - num_available, replace=True)
            indices = np.concatenate([np.arange(num_available), extra])
            np.random.shuffle(indices)
        else:
            reps = int(np.ceil(num_points / num_available))
            indices = np.tile(np.arange(num_available), reps)[:num_points]
        valid = np.ones(num_points, dtype=bool)
        return indices.astype(np.int64), valid

    indices = np.zeros(num_points, dtype=np.int64)
    indices[:num_available] = np.arange(num_available, dtype=np.int64)
    valid = np.zeros(num_points, dtype=bool)
    valid[:num_available] = True
    return indices, valid

