"""Pure PyTorch EdgeConv blocks."""

from __future__ import annotations

import torch
from torch import nn


def knn_indices(x: torch.Tensor, k: int) -> torch.Tensor:
    """Compute k-nearest-neighbor indices for batched point/features [B, N, C]."""

    if x.ndim != 3:
        raise ValueError(f"x must have shape [B, N, C], got {tuple(x.shape)}")
    num_points = x.shape[1]
    if num_points == 0:
        raise ValueError("Cannot run kNN on an empty point set.")
    if num_points == 1:
        return torch.zeros(x.shape[0], 1, 1, dtype=torch.long, device=x.device)

    k_eff = min(int(k) + 1, num_points)
    distances = torch.cdist(x, x)
    idx = distances.topk(k=k_eff, dim=-1, largest=False).indices
    return idx[:, :, 1:]


def get_graph_feature(x: torch.Tensor, k: int) -> torch.Tensor:
    """Build EdgeConv features [x_i, x_j - x_i] as [B, 2C, N, K]."""

    batch_size, num_points, channels = x.shape
    idx = knn_indices(x, k)
    k_eff = idx.shape[-1]

    offsets = torch.arange(batch_size, device=x.device).view(batch_size, 1, 1) * num_points
    flat_idx = (idx + offsets).reshape(-1)
    flat_x = x.reshape(batch_size * num_points, channels)
    neighbors = flat_x[flat_idx].view(batch_size, num_points, k_eff, channels)
    center = x.unsqueeze(2).expand(-1, -1, k_eff, -1)
    features = torch.cat([center, neighbors - center], dim=-1)
    return features.permute(0, 3, 1, 2).contiguous()


class EdgeConv(nn.Module):
    """EdgeConv block with Conv2d + BatchNorm2d + ReLU and max neighbor pooling."""

    def __init__(self, in_channels: int, out_channels: int, k: int = 20) -> None:
        super().__init__()
        self.k = int(k)
        self.net = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = get_graph_feature(x, self.k)
        features = self.net(features)
        features = features.max(dim=-1).values
        return features.transpose(1, 2).contiguous()

