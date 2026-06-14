"""Prototype construction and matching."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class ClassIndex:
    classes: list[int]

    @property
    def num_classes(self) -> int:
        return len(self.classes)

    def encode(self, labels: torch.Tensor) -> torch.Tensor:
        mapping = {label: idx for idx, label in enumerate(self.classes)}
        return torch.as_tensor([mapping[int(label)] for label in labels.tolist()], dtype=torch.long)


def build_prototypes(tokens: torch.Tensor, encoded_labels: torch.Tensor, num_classes: int) -> tuple[torch.Tensor, torch.Tensor]:
    hidden_dim = tokens.shape[1]
    sums = tokens.new_zeros(num_classes, hidden_dim)
    counts = tokens.new_zeros(num_classes)
    sums.index_add_(0, encoded_labels, tokens)
    counts.index_add_(0, encoded_labels, torch.ones_like(encoded_labels, dtype=tokens.dtype))
    if torch.any(counts <= 0):
        missing = torch.nonzero(counts <= 0, as_tuple=False).flatten().tolist()
        raise ValueError(f"Cannot build prototypes for missing encoded classes: {missing}")
    return sums / counts[:, None], counts


def prototype_logits(tokens: torch.Tensor, prototypes: torch.Tensor, tau: float) -> torch.Tensor:
    x = F.normalize(tokens, p=2, dim=1)
    p = F.normalize(prototypes, p=2, dim=1)
    return x @ p.T / tau


def leave_one_out_logits(
    tokens: torch.Tensor,
    encoded_labels: torch.Tensor,
    *,
    num_classes: int,
    tau: float,
) -> torch.Tensor:
    sums = tokens.new_zeros(num_classes, tokens.shape[1])
    counts = tokens.new_zeros(num_classes)
    sums.index_add_(0, encoded_labels, tokens)
    counts.index_add_(0, encoded_labels, torch.ones_like(encoded_labels, dtype=tokens.dtype))
    prototypes = sums / counts.clamp_min(1.0)[:, None]
    logits = prototype_logits(tokens, prototypes, tau)

    own_counts = counts.index_select(0, encoded_labels)
    replace = own_counts > 1
    if torch.any(replace):
        row_idx = torch.nonzero(replace, as_tuple=False).flatten()
        cls_idx = encoded_labels.index_select(0, row_idx)
        loo_proto = (sums.index_select(0, cls_idx) - tokens.index_select(0, row_idx)) / (own_counts.index_select(0, row_idx) - 1)[:, None]
        x = F.normalize(tokens.index_select(0, row_idx), p=2, dim=1)
        p = F.normalize(loo_proto, p=2, dim=1)
        logits = logits.clone()
        logits[row_idx, cls_idx] = (x * p).sum(dim=1) / tau
    return logits
