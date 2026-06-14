"""Face-level segmentation metrics."""

from __future__ import annotations

import torch


def accuracy(pred: torch.Tensor, target: torch.Tensor) -> float:
    if target.numel() == 0:
        return 0.0
    return float((pred == target).float().mean().item())


def mean_iou(pred: torch.Tensor, target: torch.Tensor, classes: list[int]) -> float:
    ious: list[float] = []
    for cls in classes:
        pred_mask = pred == cls
        target_mask = target == cls
        union = torch.logical_or(pred_mask, target_mask).sum().item()
        if union == 0:
            continue
        inter = torch.logical_and(pred_mask, target_mask).sum().item()
        ious.append(float(inter / union))
    return float(sum(ious) / len(ious)) if ious else 0.0

