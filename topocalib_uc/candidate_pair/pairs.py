"""Candidate-pair selection for training and inference."""

from __future__ import annotations

import torch


def training_candidate_pairs(probs: torch.Tensor, encoded_y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (a, b, margin) where a is GT and b is hardest negative."""

    if probs.ndim != 2:
        raise ValueError("probs must have shape [num_faces, num_classes].")
    masked = probs.clone()
    masked[torch.arange(probs.shape[0]), encoded_y] = -1.0
    b = masked.argmax(dim=1)
    a = encoded_y.long()
    margin = probs[torch.arange(probs.shape[0]), a] - probs[torch.arange(probs.shape[0]), b]
    return a, b, margin


def inference_candidate_pairs(probs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (a, b, margin) where a/b are top-1/top-2 classes."""

    if probs.ndim != 2:
        raise ValueError("probs must have shape [num_faces, num_classes].")
    if probs.shape[1] < 2:
        raise ValueError("candidate-pair calibration requires at least two classes.")
    top = torch.topk(probs, k=2, dim=1)
    a = top.indices[:, 0]
    b = top.indices[:, 1]
    margin = top.values[:, 0] - top.values[:, 1]
    return a, b, margin
