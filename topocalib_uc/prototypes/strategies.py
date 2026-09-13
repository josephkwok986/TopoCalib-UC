"""Mean, medoid, and sDBSCAN-filtered single-prototype strategies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from topocalib_uc.prototypes.matching import build_prototypes, leave_one_out_logits, prototype_logits
from topocalib_uc.prototypes.sdbscan_backend import SDBSCANConfig, sdbscan_labels


@dataclass(frozen=True)
class PrototypeStrategyConfig:
    method: str = "mean"
    medoid_chunk_size: int = 1024
    sdbscan: SDBSCANConfig | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "medoid_chunk_size": self.medoid_chunk_size,
            "sdbscan": self.sdbscan.as_dict() if self.sdbscan is not None else None,
            "leave_one_out": "exact exclusion before true-class prototype construction",
        }


def build_prototypes_with_strategy(
    tokens: torch.Tensor,
    encoded_labels: torch.Tensor,
    num_classes: int,
    config: PrototypeStrategyConfig,
    *,
    sdbscan_backend_module: Any | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Build one prototype per encoded class and return traceable class statistics."""

    _validate_inputs(tokens, encoded_labels, num_classes, config)
    if config.method == "mean":
        prototypes, counts = build_prototypes(tokens, encoded_labels, num_classes)
        class_stats = [
            {
                "encoded_class": class_id,
                "support_count": int(counts[class_id].item()),
                "retained_count": int(counts[class_id].item()),
                "noise_count": 0,
                "fallback_to_mean": False,
            }
            for class_id in range(num_classes)
        ]
    else:
        prototype_rows = []
        count_rows = []
        class_stats = []
        for class_id in range(num_classes):
            indices = torch.nonzero(encoded_labels == class_id, as_tuple=False).flatten()
            if indices.numel() == 0:
                raise ValueError(f"Cannot build prototype for missing encoded class {class_id}")
            class_tokens = tokens.index_select(0, indices)
            prototype, stats = build_single_class_prototype(
                class_tokens,
                config,
                sdbscan_backend_module=sdbscan_backend_module,
            )
            stats["encoded_class"] = class_id
            prototype_rows.append(prototype)
            count_rows.append(float(indices.numel()))
            class_stats.append(stats)
        prototypes = torch.stack(prototype_rows)
        counts = tokens.new_tensor(count_rows)
    info = {
        "strategy": config.as_dict(),
        "class_stats": class_stats,
        "fallback_count": sum(bool(row["fallback_to_mean"]) for row in class_stats),
        "total_retained": sum(int(row["retained_count"]) for row in class_stats),
        "total_noise": sum(int(row["noise_count"]) for row in class_stats),
    }
    return prototypes, counts, info


def build_single_class_prototype(
    class_tokens: torch.Tensor,
    config: PrototypeStrategyConfig,
    *,
    sdbscan_backend_module: Any | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if class_tokens.ndim != 2 or class_tokens.shape[0] == 0:
        raise ValueError("class_tokens must be non-empty with shape [num_supports, hidden_dim]")
    support_count = int(class_tokens.shape[0])
    if config.method == "mean":
        prototype = class_tokens.mean(dim=0)
        retained = support_count
        noise = 0
        fallback = False
        effective = None
    elif config.method == "medoid":
        prototype = medoid_prototype(class_tokens, chunk_size=config.medoid_chunk_size)
        retained = support_count
        noise = 0
        fallback = False
        effective = None
    elif config.method == "sdbscan":
        if config.sdbscan is None:
            raise ValueError("sDBSCAN strategy requires explicit sDBSCAN configuration")
        labels, effective = sdbscan_labels(
            class_tokens,
            config.sdbscan,
            backend_module=sdbscan_backend_module,
        )
        retained_mask = torch.as_tensor(labels != -1, dtype=torch.bool, device=class_tokens.device)
        retained = int(retained_mask.sum().item())
        noise = support_count - retained
        fallback = retained == 0
        prototype = class_tokens.mean(dim=0) if fallback else class_tokens[retained_mask].mean(dim=0)
    else:
        raise ValueError(f"unsupported prototype method {config.method!r}")
    return prototype, {
        "support_count": support_count,
        "retained_count": retained,
        "noise_count": noise,
        "fallback_to_mean": fallback,
        "effective_sdbscan": effective,
    }


def medoid_prototype(class_tokens: torch.Tensor, *, chunk_size: int = 1024) -> torch.Tensor:
    """Select the first minimum-distance support using chunked cosine sums."""

    if class_tokens.ndim != 2 or class_tokens.shape[0] == 0:
        raise ValueError("class_tokens must be non-empty with shape [num_supports, hidden_dim]")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if class_tokens.shape[0] == 1:
        return class_tokens[0]
    normalized = F.normalize(class_tokens, p=2, dim=1)
    distance_sums = class_tokens.new_empty(class_tokens.shape[0])
    for start in range(0, class_tokens.shape[0], chunk_size):
        stop = min(start + chunk_size, class_tokens.shape[0])
        distance_sums[start:stop] = (1.0 - normalized[start:stop] @ normalized.T).sum(dim=1)
    index = int(torch.argmin(distance_sums).item())
    return class_tokens[index]


def leave_one_out_logits_with_strategy(
    tokens: torch.Tensor,
    encoded_labels: torch.Tensor,
    *,
    num_classes: int,
    tau: float,
    config: PrototypeStrategyConfig,
    sdbscan_backend_module: Any | None = None,
) -> torch.Tensor:
    """Use exact true-class leave-one-out construction for every strategy."""

    if config.method == "mean":
        return leave_one_out_logits(tokens, encoded_labels, num_classes=num_classes, tau=tau)
    prototypes, _, _ = build_prototypes_with_strategy(
        tokens,
        encoded_labels,
        num_classes,
        config,
        sdbscan_backend_module=sdbscan_backend_module,
    )
    logits = prototype_logits(tokens, prototypes, tau)
    for row in range(tokens.shape[0]):
        class_id = int(encoded_labels[row].item())
        same_class = torch.nonzero(encoded_labels == class_id, as_tuple=False).flatten()
        if same_class.numel() <= 1:
            continue
        retained = same_class[same_class != row]
        loo_prototype, _ = build_single_class_prototype(
            tokens.index_select(0, retained),
            config,
            sdbscan_backend_module=sdbscan_backend_module,
        )
        similarity = (
            F.normalize(tokens[row : row + 1], p=2, dim=1)
            * F.normalize(loo_prototype[None, :], p=2, dim=1)
        ).sum()
        logits[row, class_id] = similarity / tau
    return logits


def _validate_inputs(
    tokens: torch.Tensor,
    encoded_labels: torch.Tensor,
    num_classes: int,
    config: PrototypeStrategyConfig,
) -> None:
    if tokens.ndim != 2 or encoded_labels.ndim != 1 or tokens.shape[0] != encoded_labels.shape[0]:
        raise ValueError("tokens and encoded_labels must have shapes [N,D] and [N]")
    if num_classes <= 0:
        raise ValueError("num_classes must be positive")
    if config.method not in {"mean", "medoid", "sdbscan"}:
        raise ValueError(f"unsupported prototype method {config.method!r}")
    if config.medoid_chunk_size <= 0:
        raise ValueError("medoid_chunk_size must be positive")
