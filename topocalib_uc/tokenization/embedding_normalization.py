"""Frozen embedding normalization used before the token adapter."""

from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.nn.functional as F


DEFAULT_EPS = 1e-6


def fit_embedding_normalizer(
    z_by_part: dict[str, torch.Tensor],
    train_part_ids: list[str],
    *,
    eps: float = DEFAULT_EPS,
) -> dict[str, Any]:
    chunks = []
    missing = []
    for part_id in train_part_ids:
        z = z_by_part.get(part_id)
        if z is None:
            missing.append(part_id)
        elif z.numel():
            chunks.append(z.float())
    if missing:
        raise KeyError(f"Missing frozen embeddings for train parts: {missing[:5]}")
    if not chunks:
        raise ValueError("Cannot fit embedding normalizer from an empty training split.")
    train_z = torch.cat(chunks, dim=0)
    return {
        "type": "train_split_zscore_l2",
        "mean": train_z.mean(dim=0).detach().cpu(),
        "std": train_z.std(dim=0, unbiased=False).detach().cpu(),
        "eps": float(eps),
        "num_train_faces": int(train_z.shape[0]),
    }


def apply_embedding_normalizer(
    z_by_part: dict[str, torch.Tensor],
    normalizer: Mapping[str, Any] | None,
) -> dict[str, torch.Tensor]:
    if not normalizer:
        return {part_id: z.float() for part_id, z in z_by_part.items()}
    mean = torch.as_tensor(normalizer["mean"], dtype=torch.float32)
    std = torch.as_tensor(normalizer["std"], dtype=torch.float32)
    eps = float(normalizer.get("eps", DEFAULT_EPS))
    out = {}
    for part_id, z in z_by_part.items():
        x = z.float()
        mean_i = mean.to(device=x.device)
        std_i = std.to(device=x.device)
        out[part_id] = F.normalize((x - mean_i) / (std_i + eps), p=2, dim=1, eps=eps)
    return out


def normalizer_json_info(normalizer: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "type": str(normalizer.get("type", "train_split_zscore_l2")),
        "eps": float(normalizer.get("eps", DEFAULT_EPS)),
        "num_train_faces": int(normalizer.get("num_train_faces", 0)),
    }
