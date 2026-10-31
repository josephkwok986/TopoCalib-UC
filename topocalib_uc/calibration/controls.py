"""Structural controls and training-frequency adjustment (Appendix B)."""
from __future__ import annotations

import torch
from topocalib_uc.calibration.margin import ambiguity_gate, apply_margin_calibration
from topocalib_uc.evidence.readout import FaceBatchMeta


def structural_evidence(
    probs: torch.Tensor, meta: FaceBatchMeta, *, lambda_part: float = 0.5, top_k: int = 5,
) -> torch.Tensor:
    """Full-class evidence with detached, relation-balanced neighbor support."""
    probs = probs.detach()
    if top_k <= 0:
        raise ValueError('top_k must be positive')
    n, c = probs.shape
    sums = probs.new_zeros(n * 4, c)
    counts = probs.new_zeros(n * 4)
    edges = meta.edges.long().reshape(-1, 2)
    rel = meta.edge_type.long().clamp(0, 3)
    if edges.numel():
        src, dst = edges.T
        valid = (src != dst) & (src >= 0) & (dst >= 0) & (src < n) & (dst < n)
        src, dst, rel = src[valid], dst[valid], rel[valid]
        for a, b in ((src, dst), (dst, src)):
            idx = a * 4 + rel
            sums.index_add_(0, idx, probs[b])
            counts.index_add_(0, idx, torch.ones_like(idx, dtype=probs.dtype))
    counts = counts.reshape(n, 4)
    local = (sums.reshape(n, 4, c) / counts.clamp_min(1).unsqueeze(-1)).sum(1)
    local /= (counts > 0).sum(1).clamp_min(1).unsqueeze(-1)
    part = torch.zeros_like(probs)
    for pid in torch.unique(meta.part_index):
        idx = torch.where(meta.part_index == pid)[0]
        m = len(idx)
        if m <= 1:
            continue
        # Per-query exclusion preserves the original same-part evidence rule.
        for face in idx:
            others = idx[idx != face]
            part[face] = probs[others].topk(min(top_k, m - 1), dim=0).values.mean(0)
    return local + float(lambda_part) * part


def apply_control(logits, probs, a, b, margin, meta, *, mode='pairwise',
                  smoothing_alpha=0.5, lambda_all=1.0, **kwargs):
    if mode == 'pairwise':
        return apply_margin_calibration(logits, probs, a, b, margin, meta, **kwargs)
    if mode not in {'smoothing', 'all_class'}:
        raise ValueError(f'Unknown calibration mode: {mode}')
    evidence = structural_evidence(probs, meta, lambda_part=kwargs['lambda_part'], top_k=kwargs['top_k'])
    if mode == 'smoothing':
        if not 0 <= smoothing_alpha <= 1:
            raise ValueError('smoothing_alpha must be in [0, 1]')
        total = evidence.sum(1, keepdim=True)
        base = torch.softmax(logits, dim=1)
        normalized = evidence / total.clamp_min(torch.finfo(logits.dtype).tiny)
        mixed = torch.where(total > 0, (1 - smoothing_alpha) * base + smoothing_alpha * normalized, base)
        return mixed.clamp_min(torch.finfo(logits.dtype).tiny).log()
    gate = ambiguity_gate(margin, kwargs['beta'])
    return logits + 2 * gate[:, None] * lambda_all * (evidence - evidence.mean(1, keepdim=True))


def frequency_adjustment(logits: torch.Tensor, counts, eta: float) -> torch.Tensor:
    counts = torch.as_tensor(counts, dtype=logits.dtype, device=logits.device)
    if counts.shape != (logits.shape[1],) or not torch.isfinite(counts).all() or (counts <= 0).any():
        raise ValueError('Frequency adjustment requires positive labeled-training counts for every class')
    if eta < 0:
        raise ValueError('eta must be nonnegative')
    return logits - eta * (counts / counts.sum()).log()
