"""Face-level segmentation metrics."""

from __future__ import annotations

from typing import TypedDict

import torch


class PerClassIoU(TypedDict):
    class_id: int
    intersection: int
    union: int
    iou: float | None


def accuracy(pred: torch.Tensor, target: torch.Tensor) -> float:
    if target.numel() == 0:
        return 0.0
    return float((pred == target).float().mean().item())


def mean_iou(pred: torch.Tensor, target: torch.Tensor, classes: list[int]) -> float:
    ious = [row["iou"] for row in per_class_iou(pred, target, classes) if row["iou"] is not None]
    return float(sum(ious) / len(ious)) if ious else 0.0


def per_class_iou(
    pred: torch.Tensor,
    target: torch.Tensor,
    classes: list[int],
) -> list[PerClassIoU]:
    """Return traceable intersection, union, and IoU for every class."""

    if pred.shape != target.shape:
        raise ValueError(f"pred and target must have the same shape, got {tuple(pred.shape)} and {tuple(target.shape)}")
    if pred.ndim != 1:
        raise ValueError("pred and target must be one-dimensional")
    if len(classes) != len(set(classes)):
        raise ValueError("classes contains duplicates")
    rows: list[PerClassIoU] = []
    for cls in classes:
        pred_mask = pred == cls
        target_mask = target == cls
        union = int(torch.logical_or(pred_mask, target_mask).sum().item())
        intersection = int(torch.logical_and(pred_mask, target_mask).sum().item())
        rows.append(
            {
                "class_id": int(cls),
                "intersection": intersection,
                "union": union,
                "iou": float(intersection / union) if union else None,
            }
        )
    return rows


def grouped_mean_iou(
    class_rows: list[PerClassIoU],
    groups: dict[str, list[int]],
) -> dict[str, float]:
    """Compute unweighted group means from saved per-class IoUs."""

    by_class = {int(row["class_id"]): row["iou"] for row in class_rows}
    assigned = [class_id for class_ids in groups.values() for class_id in class_ids]
    if len(assigned) != len(set(assigned)):
        raise ValueError("class groups overlap")
    if set(assigned) != set(by_class):
        raise ValueError("class groups must partition all per-class IoU rows")
    output: dict[str, float] = {}
    for name, class_ids in groups.items():
        values = [by_class[class_id] for class_id in class_ids]
        if any(value is None for value in values):
            missing = [class_id for class_id, value in zip(class_ids, values) if value is None]
            raise ValueError(f"group {name!r} has undefined IoU for classes {missing}")
        numeric_values = [float(value) for value in values if value is not None]
        output[name] = float(sum(numeric_values) / len(numeric_values)) if numeric_values else 0.0
    return output
