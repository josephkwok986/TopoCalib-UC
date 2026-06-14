"""Plain PyTorch implementation of the SSRL downstream segmentation head.

The CVPR 2023 SSRL for CAD paper describes the face segmentation head as a
2-layer Residual MR-GCN over pre-computed face embeddings, followed by a
two-hidden-layer fully connected classifier. The original public code delegates
the residual max-relative graph convolution to AutoMate; this module keeps the
same algorithmic shape without depending on AutoMate, PyG, or torch_scatter.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, layer_sizes: list[int], *, dropout: float = 0.0, last_linear: bool = True):
        super().__init__()
        layers: list[nn.Module] = []
        for idx, (in_dim, out_dim) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            layers.append(nn.Linear(in_dim, out_dim))
            is_last = idx == len(layer_sizes) - 2
            if is_last and last_linear:
                continue
            layers.append(nn.LeakyReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResidualMRConv(nn.Module):
    """Residual max-relative graph convolution used by the SSRL downstream head."""

    def __init__(self, width: int, *, dropout: float = 0.0):
        super().__init__()
        self.update = MLP([2 * width, width], dropout=dropout, last_linear=False)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        if edge_index.numel() == 0:
            max_relative = torch.zeros_like(x)
        else:
            if edge_index.ndim != 2 or edge_index.shape[1] != 2:
                raise ValueError(f"edge_index must be [num_edges, 2], got {tuple(edge_index.shape)}")
            src = edge_index[:, 0].long()
            dst = edge_index[:, 1].long()
            diffs = x.index_select(0, dst) - x.index_select(0, src)
            max_relative = _scatter_max(diffs, dst, dim_size=x.shape[0])
        return x + self.update(torch.cat([x, max_relative], dim=1))


class SSRLMRGCN(nn.Module):
    """2-layer residual MR-GCN + two-hidden-layer MLP face classifier."""

    def __init__(
        self,
        embedding_dim: int,
        num_classes: int,
        *,
        hidden_dim: int = 64,
        mp_layers: int = 2,
        mlp_hidden_dim: int | None = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        if mp_layers <= 0:
            raise ValueError("mp_layers must be positive.")
        mlp_hidden = int(mlp_hidden_dim or hidden_dim)
        self.input_proj = nn.Linear(embedding_dim, hidden_dim)
        self.message_passing = nn.ModuleList([ResidualMRConv(hidden_dim, dropout=dropout) for _ in range(mp_layers)])
        self.classifier = MLP(
            [hidden_dim, mlp_hidden, mlp_hidden, num_classes],
            dropout=dropout,
            last_linear=True,
        )

    def forward(self, z: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = F.leaky_relu(self.input_proj(z))
        for layer in self.message_passing:
            x = layer(x, edge_index)
        return self.classifier(x)


def _scatter_max(values: torch.Tensor, index: torch.Tensor, *, dim_size: int) -> torch.Tensor:
    output = values.new_full((dim_size, values.shape[1]), -torch.inf)
    if hasattr(output, "scatter_reduce_"):
        expanded_index = index[:, None].expand_as(values)
        output.scatter_reduce_(0, expanded_index, values, reduce="amax", include_self=True)
    else:
        for row, dst in zip(values, index.tolist()):
            output[dst] = torch.maximum(output[dst], row)
    return torch.where(torch.isfinite(output), output, torch.zeros_like(output))
