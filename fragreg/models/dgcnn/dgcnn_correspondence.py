"""DGCNN correspondence model for canonical-coordinate prediction."""

from __future__ import annotations

import torch
from torch import nn

from fragreg.models.dgcnn.edgeconv import EdgeConv
from fragreg.registry import MODELS


@MODELS.register_module()
class DGCNNCorrespondence(nn.Module):
    """Predict per-point coordinates in the original object's canonical frame."""

    def __init__(
        self,
        in_channels: int = 3,
        k: int = 20,
        emb_dims: int = 256,
        output_confidence: bool = True,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.k = int(k)
        self.output_confidence = bool(output_confidence)

        self.edge1 = EdgeConv(self.in_channels, 64, k=self.k)
        self.edge2 = EdgeConv(64, 64, k=self.k)
        self.edge3 = EdgeConv(64, 128, k=self.k)

        local_channels = 64 + 64 + 128
        self.point_mlp = nn.Sequential(
            nn.Conv1d(local_channels, emb_dims, kernel_size=1, bias=False),
            nn.BatchNorm1d(emb_dims),
            nn.ReLU(inplace=True),
            nn.Conv1d(emb_dims, emb_dims, kernel_size=1, bias=False),
            nn.BatchNorm1d(emb_dims),
            nn.ReLU(inplace=True),
        )
        self.coord_head = nn.Sequential(
            nn.Conv1d(emb_dims, 128, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 3, kernel_size=1),
        )
        if self.output_confidence:
            self.confidence_head = nn.Sequential(
                nn.Conv1d(emb_dims, 128, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv1d(128, 1, kernel_size=1),
            )
        else:
            self.confidence_head = None

    def forward(self, points: torch.Tensor | dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if isinstance(points, dict):
            points = points["points_C"]
        if points.ndim != 3:
            raise ValueError(f"points must have shape [B, N, C], got {tuple(points.shape)}")

        x1 = self.edge1(points)
        x2 = self.edge2(x1)
        x3 = self.edge3(x2)
        features = torch.cat([x1, x2, x3], dim=-1).transpose(1, 2).contiguous()
        features = self.point_mlp(features)

        pred_points_O = self.coord_head(features).transpose(1, 2).contiguous()
        outputs = {"pred_points_O": pred_points_O}
        if self.confidence_head is not None:
            outputs["confidence_logits"] = self.confidence_head(features).transpose(1, 2).contiguous()
        else:
            outputs["confidence_logits"] = pred_points_O.new_zeros(*pred_points_O.shape[:2], 1)
        return outputs

