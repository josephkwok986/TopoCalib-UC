"""Epoch-level prototypes from the complete labeled support set."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from topocalib_uc.prototypes.matching import prototype_logits
from topocalib_uc.prototypes.strategies import (
    PrototypeStrategyConfig, build_prototypes_with_strategy, build_single_class_prototype,
)


@dataclass
class PrototypeBank:
    tokens: torch.Tensor
    labels: torch.Tensor
    prototypes: torch.Tensor
    counts: torch.Tensor
    info: dict[str, Any]
    config: PrototypeStrategyConfig

    @classmethod
    def build(cls, tokens, labels, num_classes, config):
        tokens = tokens.detach()
        labels = labels.detach()
        prototypes, counts, info = build_prototypes_with_strategy(tokens, labels, num_classes, config)
        return cls(tokens, labels, prototypes.detach(), counts.detach(), info, config)

    def training_logits(self, query_tokens, support_indices, *, tau):
        """Match live queries against frozen support; exclude the cached query face."""
        indices = support_indices.to(device=self.tokens.device, dtype=torch.long)
        if indices.shape != (len(query_tokens),):
            raise ValueError('One support index is required per query face')
        own_labels = self.labels[indices]
        logits = prototype_logits(query_tokens, self.prototypes, tau)
        rows = torch.where(self.counts[own_labels] > 1)[0]
        if not rows.numel():
            return logits
        classes = own_labels[rows]
        if self.config.method == 'mean':
            counts = self.counts[classes, None]
            excluded = (self.prototypes[classes] * counts - self.tokens[indices[rows]]) / (counts - 1)
        else:
            excluded = []
            for row in rows.tolist():
                support_row = indices[row]
                keep = (self.labels == own_labels[row]) & (torch.arange(len(self.labels), device=self.labels.device) != support_row)
                prototype, _ = build_single_class_prototype(self.tokens[keep], self.config)
                excluded.append(prototype)
            excluded = torch.stack(excluded)
        similarities = (F.normalize(query_tokens[rows], dim=1) * F.normalize(excluded, dim=1)).sum(1) / tau
        logits = logits.clone()
        logits[rows, classes] = similarities
        return logits
