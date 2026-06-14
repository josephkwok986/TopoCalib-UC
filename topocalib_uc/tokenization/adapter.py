"""Lightweight surface-type token adapter."""

from __future__ import annotations

import torch
from torch import nn


class SurfaceTokenAdapter(nn.Module):
    def __init__(self, embedding_dim: int, hidden_dim: int, num_surface_types: int = 5):
        super().__init__()
        self.proj = nn.Linear(embedding_dim, hidden_dim, bias=False)
        self.surface_embedding = nn.Embedding(num_surface_types, hidden_dim)
        self.surface_gate_logit = nn.Parameter(torch.zeros(()))

    def forward(self, z: torch.Tensor, surface_type: torch.Tensor) -> torch.Tensor:
        max_id = self.surface_embedding.num_embeddings - 1
        surface_type = surface_type.long().clamp(min=0, max=max_id)
        gate = torch.sigmoid(self.surface_gate_logit)
        return self.proj(z) + gate * self.surface_embedding(surface_type)
