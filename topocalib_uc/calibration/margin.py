"""Unified candidate-pair margin calibration."""

from __future__ import annotations

import torch

from topocalib_uc.evidence.readout import FaceBatchMeta, local_candidate_pair_evidence, part_level_candidate_pair_prior


def ambiguity_gate(margin: torch.Tensor, beta: float) -> torch.Tensor:
    if beta <= 0:
        raise ValueError("beta must be positive.")
    return torch.clamp((beta - margin) / beta, min=0.0, max=1.0)


def apply_margin_calibration(
    logits: torch.Tensor,
    probs_for_evidence: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    margin: torch.Tensor,
    meta: FaceBatchMeta,
    *,
    use_local_evidence: bool,
    use_ambiguity_gate: bool,
    use_part_prior: bool,
    beta: float,
    lambda_cal: float,
    lambda_part: float,
    top_k: int,
) -> torch.Tensor:
    """Apply the TopoCalib-UC logit correction to a single face batch."""

    if not (use_local_evidence or use_part_prior):
        return logits
    evidence = logits.new_zeros(logits.shape[0])
    read_probs = probs_for_evidence.detach()
    if use_local_evidence:
        evidence = evidence + local_candidate_pair_evidence(read_probs, a, b, meta)
    if use_part_prior:
        evidence = evidence + lambda_part * part_level_candidate_pair_prior(read_probs, a, b, meta, top_k=top_k)
    gate = ambiguity_gate(margin, beta) if use_ambiguity_gate else torch.ones_like(margin)
    delta = gate * float(lambda_cal) * evidence
    calibrated = logits.clone()
    rows = torch.arange(logits.shape[0], device=logits.device)
    calibrated[rows, a] = calibrated[rows, a] + delta
    calibrated[rows, b] = calibrated[rows, b] - delta
    return calibrated
