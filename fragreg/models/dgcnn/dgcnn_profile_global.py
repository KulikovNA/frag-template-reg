"""DGCNN profile model with global fragment context."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from fragreg.models.dgcnn.edgeconv import EdgeConv
from fragreg.registry import MODELS


@MODELS.register_module()
class DGCNNProfileGlobal(nn.Module):
    """Predict profile coordinates from local EdgeConv and global shell context."""

    def __init__(
        self,
        in_channels: int = 3,
        k: int = 20,
        emb_dims: int = 512,
        global_dims: int = 512,
        output_confidence: bool = True,
        output_axis: bool = True,
        output_axial_stats: bool = False,
        use_mean_pool: bool = True,
        use_max_pool: bool = True,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.k = int(k)
        self.emb_dims = int(emb_dims)
        self.global_dims = int(global_dims)
        self.output_confidence = bool(output_confidence)
        self.output_axis = bool(output_axis)
        self.output_axial_stats = bool(output_axial_stats)
        self.use_mean_pool = bool(use_mean_pool)
        self.use_max_pool = bool(use_max_pool)
        if not self.use_mean_pool and not self.use_max_pool:
            raise ValueError("At least one of use_mean_pool/use_max_pool must be enabled.")

        self.edge1 = EdgeConv(self.in_channels, 64, k=self.k)
        self.edge2 = EdgeConv(64, 64, k=self.k)
        self.edge3 = EdgeConv(64, 128, k=self.k)

        local_channels = 64 + 64 + 128
        pool_channels = local_channels * int(self.use_max_pool) + local_channels * int(self.use_mean_pool)
        self.global_mlp = nn.Sequential(
            nn.Linear(pool_channels, self.global_dims),
            nn.LayerNorm(self.global_dims),
            nn.ReLU(inplace=True),
            nn.Linear(self.global_dims, self.global_dims),
            nn.LayerNorm(self.global_dims),
            nn.ReLU(inplace=True),
        )
        self.point_fuse = nn.Sequential(
            nn.Conv1d(local_channels + self.global_dims, self.emb_dims, kernel_size=1, bias=False),
            nn.BatchNorm1d(self.emb_dims),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.emb_dims, self.emb_dims, kernel_size=1, bias=False),
            nn.BatchNorm1d(self.emb_dims),
            nn.ReLU(inplace=True),
        )
        self.profile_head = nn.Sequential(
            nn.Conv1d(self.emb_dims, 128, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 2, kernel_size=1),
        )
        if self.output_confidence:
            self.confidence_head = nn.Sequential(
                nn.Conv1d(self.emb_dims, 128, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv1d(128, 1, kernel_size=1),
            )
        else:
            self.confidence_head = None
        if self.output_axis:
            self.axis_head = nn.Sequential(
                nn.Linear(self.global_dims, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, 3),
            )
        else:
            self.axis_head = None
        if self.output_axial_stats:
            self.axial_stats_head = nn.Sequential(
                nn.Linear(self.global_dims, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, 3),
            )
        else:
            self.axial_stats_head = None

    def _global_pool(self, point_feat: torch.Tensor) -> torch.Tensor:
        pooled = []
        if self.use_max_pool:
            pooled.append(point_feat.max(dim=1).values)
        if self.use_mean_pool:
            pooled.append(point_feat.mean(dim=1))
        return torch.cat(pooled, dim=-1)

    def forward(self, points: torch.Tensor | dict[str, torch.Tensor]) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        if isinstance(points, dict):
            points = points["points_C"]
        if points.ndim != 3:
            raise ValueError(f"points must have shape [B, N, C], got {tuple(points.shape)}")

        x1 = self.edge1(points)
        x2 = self.edge2(x1)
        x3 = self.edge3(x2)
        point_feat = torch.cat([x1, x2, x3], dim=-1)

        global_feat = self.global_mlp(self._global_pool(point_feat))
        global_tiled = global_feat.unsqueeze(1).expand(-1, point_feat.shape[1], -1)
        fused = torch.cat([point_feat, global_tiled], dim=-1).transpose(1, 2).contiguous()
        fused = self.point_fuse(fused)

        pred_profile_O = self.profile_head(fused).transpose(1, 2).contiguous()
        outputs: dict[str, torch.Tensor | dict[str, torch.Tensor]] = {"pred_profile_O": pred_profile_O}
        if self.confidence_head is not None:
            outputs["confidence_logits"] = self.confidence_head(fused).transpose(1, 2).contiguous()
        else:
            outputs["confidence_logits"] = pred_profile_O.new_zeros(*pred_profile_O.shape[:2], 1)
        if self.axis_head is not None:
            outputs["pred_axis_C"] = F.normalize(self.axis_head(global_feat), dim=-1)
        if self.axial_stats_head is not None:
            raw_stats = self.axial_stats_head(global_feat)
            outputs["pred_axial_stats"] = {
                "mean": raw_stats[:, 0],
                "std": F.softplus(raw_stats[:, 1]),
                "span": F.softplus(raw_stats[:, 2]),
            }
        return outputs
