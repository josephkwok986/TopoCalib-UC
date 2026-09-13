"""Unified candidate-pair margin calibration."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from topocalib_uc.evidence.readout import FaceBatchMeta, local_candidate_pair_evidence, part_level_candidate_pair_prior


@dataclass(frozen=True)
class MarginCalibrationDiagnostics:
    """Per-face terms used by candidate-pair margin calibration."""

    local_evidence: torch.Tensor
    part_prior: torch.Tensor
    combined_evidence: torch.Tensor
    gate: torch.Tensor
    delta: torch.Tensor


def ambiguity_gate(margin: torch.Tensor, beta: float) -> torch.Tensor:
    if beta <= 0:
        raise ValueError("beta must be positive.")
    return torch.clamp((beta - margin) / beta, min=0.0, max=1.0)


def compute_margin_calibration(
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
) -> MarginCalibrationDiagnostics:
    """Compute calibration terms without modifying logits."""

    read_probs = probs_for_evidence.detach()
    zeros = read_probs.new_zeros(read_probs.shape[0])
    local = (
        local_candidate_pair_evidence(read_probs, a, b, meta)
        if use_local_evidence
        else zeros
    )
    part = (
        part_level_candidate_pair_prior(read_probs, a, b, meta, top_k=top_k)
        if use_part_prior
        else zeros
    )
    combined = local + float(lambda_part) * part
    gate = ambiguity_gate(margin, beta) if use_ambiguity_gate else torch.ones_like(margin)
    delta = gate * float(lambda_cal) * combined
    return MarginCalibrationDiagnostics(
        local_evidence=local,
        part_prior=part,
        combined_evidence=combined,
        gate=gate,
        delta=delta,
    )


def apply_pairwise_logit_update(
    logits: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    delta: torch.Tensor,
) -> torch.Tensor:
    """Apply a precomputed signed update to the active class pair."""

    if logits.ndim != 2:
        raise ValueError(f"logits must have shape [num_faces, num_classes], got {tuple(logits.shape)}")
    if any(item.shape != (logits.shape[0],) for item in (a, b, delta)):
        raise ValueError("a, b, and delta must contain one value per logit row")
    calibrated = logits.clone()
    rows = torch.arange(logits.shape[0], device=logits.device)
    calibrated[rows, a] = calibrated[rows, a] + delta
    calibrated[rows, b] = calibrated[rows, b] - delta
    return calibrated


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
    diagnostics = compute_margin_calibration(
        probs_for_evidence,
        a,
        b,
        margin,
        meta,
        use_local_evidence=use_local_evidence,
        use_ambiguity_gate=use_ambiguity_gate,
        use_part_prior=use_part_prior,
        beta=beta,
        lambda_cal=lambda_cal,
        lambda_part=lambda_part,
        top_k=top_k,
    )
    return apply_pairwise_logit_update(logits, a, b, diagnostics.delta)
