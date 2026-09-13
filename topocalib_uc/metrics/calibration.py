"""Face-level confidence calibration metrics."""

from __future__ import annotations

from typing import Any, Sequence

import torch
import torch.nn.functional as F


def calibration_metrics_from_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    classes: Sequence[int] | None = None,
    num_bins: int = 15,
) -> dict[str, Any]:
    """Compute top-label ECE and numerically stable NLL from logits."""

    _validate_matrix(logits, targets, num_bins=num_bins)
    if targets.numel() == 0:
        return _empty_metrics(num_bins)
    encoded_targets = _encode_targets(targets, classes, logits.shape[1], logits.device)
    log_probs = F.log_softmax(logits, dim=1)
    probs = log_probs.exp()
    nll = -log_probs[torch.arange(targets.numel(), device=logits.device), encoded_targets].mean()
    result = _ece_from_valid_probabilities(probs, encoded_targets, num_bins=num_bins)
    result["nll"] = float(nll.item())
    return result


def calibration_metrics_from_probabilities(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    *,
    classes: Sequence[int] | None = None,
    num_bins: int = 15,
) -> dict[str, Any]:
    """Compute top-label ECE and NLL from normalized probabilities."""

    _validate_matrix(probabilities, targets, num_bins=num_bins)
    if targets.numel() == 0:
        return _empty_metrics(num_bins)
    if not torch.isfinite(probabilities).all() or torch.any(probabilities < 0):
        raise ValueError("probabilities must be finite and non-negative")
    row_sums = probabilities.sum(dim=1)
    if not torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-6, rtol=1e-6):
        raise ValueError("each probability row must sum to one")
    encoded_targets = _encode_targets(targets, classes, probabilities.shape[1], probabilities.device)
    tiny = torch.finfo(probabilities.dtype).tiny
    gt = probabilities[torch.arange(targets.numel(), device=probabilities.device), encoded_targets]
    nll = -torch.log(gt.clamp_min(tiny)).mean()
    result = _ece_from_valid_probabilities(probabilities, encoded_targets, num_bins=num_bins)
    result["nll"] = float(nll.item())
    return result


def _ece_from_valid_probabilities(
    probabilities: torch.Tensor,
    encoded_targets: torch.Tensor,
    *,
    num_bins: int,
) -> dict[str, Any]:
    confidence, prediction = probabilities.max(dim=1)
    correct = prediction.eq(encoded_targets)
    bin_index = torch.floor(confidence * num_bins).long().clamp(min=0, max=num_bins - 1)
    ece = probabilities.new_zeros(())
    bins: list[dict[str, float]] = []
    total = int(confidence.numel())
    for index in range(num_bins):
        selected = bin_index == index
        count = int(selected.sum().item())
        lower = index / num_bins
        upper = (index + 1) / num_bins
        if count:
            bin_accuracy = correct[selected].to(probabilities.dtype).mean()
            bin_confidence = confidence[selected].mean()
            ece = ece + (count / total) * torch.abs(bin_accuracy - bin_confidence)
            accuracy_value = float(bin_accuracy.item())
            confidence_value = float(bin_confidence.item())
        else:
            accuracy_value = 0.0
            confidence_value = 0.0
        bins.append(
            {
                "index": float(index),
                "lower": float(lower),
                "upper": float(upper),
                "upper_inclusive": float(index == num_bins - 1),
                "count": float(count),
                "accuracy": accuracy_value,
                "mean_confidence": confidence_value,
            }
        )
    value = float(ece.item())
    return {
        "num_faces": float(total),
        "num_bins": float(num_bins),
        "ece": value,
        "ece_percent": 100.0 * value,
        "bins": bins,
    }


def _validate_matrix(values: torch.Tensor, targets: torch.Tensor, *, num_bins: int) -> None:
    if values.ndim != 2:
        raise ValueError(f"scores must have shape [num_faces, num_classes], got {tuple(values.shape)}")
    if targets.ndim != 1 or targets.shape[0] != values.shape[0]:
        raise ValueError("targets must contain one label per score row")
    if values.shape[1] <= 0:
        raise ValueError("scores must contain at least one class")
    if num_bins <= 0:
        raise ValueError("num_bins must be positive")


def _encode_targets(
    targets: torch.Tensor,
    classes: Sequence[int] | None,
    num_classes: int,
    device: torch.device,
) -> torch.Tensor:
    raw_targets = [int(item) for item in targets.detach().cpu().tolist()]
    if classes is None:
        encoded = raw_targets
    else:
        class_list = [int(item) for item in classes]
        if len(class_list) != num_classes or len(set(class_list)) != len(class_list):
            raise ValueError("classes must be unique and match the score column count")
        mapping = {label: index for index, label in enumerate(class_list)}
        unknown = sorted(set(raw_targets) - set(mapping))
        if unknown:
            raise ValueError(f"targets contain labels absent from classes: {unknown}")
        encoded = [mapping[label] for label in raw_targets]
    result = torch.as_tensor(encoded, dtype=torch.long, device=device)
    if result.numel() and (int(result.min()) < 0 or int(result.max()) >= num_classes):
        raise ValueError("encoded targets are outside the score column range")
    return result


def _empty_metrics(num_bins: int) -> dict[str, Any]:
    return {
        "num_faces": 0.0,
        "num_bins": float(num_bins),
        "ece": 0.0,
        "ece_percent": 0.0,
        "nll": 0.0,
        "bins": [],
    }
