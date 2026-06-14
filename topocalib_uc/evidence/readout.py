"""Read-only evidence for TopoCalib-UC candidate-pair calibration."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class FaceBatchMeta:
    part_index: torch.Tensor
    edges: torch.Tensor
    edge_type: torch.Tensor


def local_candidate_pair_evidence(
    probs: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    meta: FaceBatchMeta,
    *,
    num_relation_types: int = 4,
) -> torch.Tensor:
    """Relation-stratified 1-hop support difference for each candidate pair."""

    num_faces = probs.shape[0]
    if meta.edges.numel() == 0:
        return probs.new_zeros(num_faces)
    sums = probs.new_zeros(num_faces, num_relation_types)
    counts = probs.new_zeros(num_faces, num_relation_types)
    edges = meta.edges.long().reshape(-1, 2)
    edge_type = meta.edge_type.long().clamp(min=0, max=num_relation_types - 1)
    src = edges[:, 0]
    dst = edges[:, 1]
    valid = (src != dst) & (src >= 0) & (dst >= 0) & (src < num_faces) & (dst < num_faces)
    if torch.any(valid):
        src = src[valid]
        dst = dst[valid]
        rel = edge_type[valid]
        flat_sums = sums.reshape(-1)
        flat_counts = counts.reshape(-1)
        src_rel = src * num_relation_types + rel
        dst_rel = dst * num_relation_types + rel
        src_delta = probs[dst, a[src]] - probs[dst, b[src]]
        dst_delta = probs[src, a[dst]] - probs[src, b[dst]]
        flat_sums.index_add_(0, src_rel, src_delta)
        flat_sums.index_add_(0, dst_rel, dst_delta)
        ones = torch.ones_like(src_delta)
        flat_counts.index_add_(0, src_rel, ones)
        flat_counts.index_add_(0, dst_rel, ones)
    nonempty = counts > 0
    per_relation = torch.where(nonempty, sums / counts.clamp_min(1.0), torch.zeros_like(sums))
    denom = nonempty.sum(dim=1).clamp_min(1).to(probs.dtype)
    return per_relation.sum(dim=1) / denom


def part_level_candidate_pair_prior(
    probs: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    meta: FaceBatchMeta,
    *,
    top_k: int,
) -> torch.Tensor:
    """Same-part Top-K prototype evidence difference for each candidate pair."""

    if top_k <= 0:
        raise ValueError("top_k must be positive.")
    out = probs.new_zeros(probs.shape[0])
    part_index = meta.part_index.long()
    for part_id in torch.unique(part_index, sorted=True).tolist():
        face_idx = torch.nonzero(part_index == int(part_id), as_tuple=False).flatten()
        m = int(face_idx.numel())
        if m <= 1:
            continue
        k = min(top_k, m - 1)
        part_probs = probs.index_select(0, face_idx)
        candidates = part_probs.unsqueeze(0).expand(m, m, part_probs.shape[1]).clone()
        eye = torch.eye(m, dtype=torch.bool, device=probs.device)
        candidates[eye] = -torch.inf
        topk_mean = torch.topk(candidates, k=k, dim=1).values.mean(dim=1)
        out[face_idx] = topk_mean[torch.arange(m, device=probs.device), a.index_select(0, face_idx)] - topk_mean[
            torch.arange(m, device=probs.device), b.index_select(0, face_idx)
        ]
    return out
