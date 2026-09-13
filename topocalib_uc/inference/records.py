"""Stable per-face prediction records for post-hoc analyses."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F

from topocalib_uc.calibration.margin import MarginCalibrationDiagnostics
from topocalib_uc.prototypes.matching import ClassIndex
from topocalib_uc.train.partgraph_dataset import PartRecord


RECORD_SCHEMA_VERSION = 1


def build_face_prediction_records(
    record: PartRecord,
    class_index: ClassIndex,
    base_logits: torch.Tensor,
    final_logits: torch.Tensor,
    candidate_a: torch.Tensor,
    candidate_b: torch.Tensor,
    probability_margin_before: torch.Tensor,
    diagnostics: MarginCalibrationDiagnostics | None,
    *,
    method: str,
    variant: str,
    budget: str,
    seed: int,
) -> list[dict[str, Any]]:
    """Build JSON-compatible rows keyed by ``(dataset, part_id, face_id)``."""

    num_faces = record.num_faces
    tensors = [base_logits, final_logits]
    if any(item.ndim != 2 or item.shape[0] != num_faces for item in tensors):
        raise ValueError("base_logits and final_logits must contain one row per face")
    vectors = [candidate_a, candidate_b, probability_margin_before]
    if any(item.shape != (num_faces,) for item in vectors):
        raise ValueError("candidate pair and margin tensors must contain one value per face")

    base_probs = F.softmax(base_logits, dim=1)
    final_probs = F.softmax(final_logits, dim=1)
    pred_before = base_logits.argmax(dim=1)
    pred_after = final_logits.argmax(dim=1)
    rows_index = torch.arange(num_faces, device=base_logits.device)
    pair_margin_before = base_logits[rows_index, candidate_a] - base_logits[rows_index, candidate_b]
    pair_margin_after = final_logits[rows_index, candidate_a] - final_logits[rows_index, candidate_b]
    probability_margin_after = final_probs[rows_index, candidate_a] - final_probs[rows_index, candidate_b]
    neighbors, relations = _face_neighborhoods(record)

    if diagnostics is None:
        zeros = probability_margin_before.new_zeros(num_faces)
        local_evidence = part_prior = gate = delta = zeros
    else:
        local_evidence = diagnostics.local_evidence
        part_prior = diagnostics.part_prior
        gate = diagnostics.gate
        delta = diagnostics.delta

    output: list[dict[str, Any]] = []
    for face_id in range(num_faces):
        a_encoded = int(candidate_a[face_id].item())
        b_encoded = int(candidate_b[face_id].item())
        before_encoded = int(pred_before[face_id].item())
        after_encoded = int(pred_after[face_id].item())
        output.append(
            {
                "schema_version": RECORD_SCHEMA_VERSION,
                "dataset": record.dataset,
                "method": method,
                "variant": variant,
                "budget": str(budget),
                "seed": int(seed),
                "part_id": record.part_id,
                "face_id": face_id,
                "face_key": [record.dataset, record.part_id, face_id],
                "classes": list(class_index.classes),
                "target": int(record.y[face_id].item()),
                "pred_before": int(class_index.classes[before_encoded]),
                "pred_after": int(class_index.classes[after_encoded]),
                "candidate_a": int(class_index.classes[a_encoded]),
                "candidate_b": int(class_index.classes[b_encoded]),
                "base_logits": _float_row(base_logits[face_id]),
                "base_probabilities": _float_row(base_probs[face_id]),
                "calibrated_logits": _float_row(final_logits[face_id]),
                "calibrated_probabilities": _float_row(final_probs[face_id]),
                "probability_margin_before": float(probability_margin_before[face_id].item()),
                "probability_margin_after": float(probability_margin_after[face_id].item()),
                "margin_before": float(probability_margin_before[face_id].item()),
                "margin_after": float(probability_margin_after[face_id].item()),
                "pair_logit_margin_before": float(pair_margin_before[face_id].item()),
                "pair_logit_margin_after": float(pair_margin_after[face_id].item()),
                "local_evidence": float(local_evidence[face_id].item()),
                "part_prior": float(part_prior[face_id].item()),
                "gate": float(gate[face_id].item()),
                "calibration_delta": float(delta[face_id].item()),
                "neighbor_face_ids": neighbors[face_id],
                "num_neighbors": len(neighbors[face_id]),
                "relation_types": relations[face_id],
            }
        )
    return output


def write_face_prediction_records(path: str | Path, records: Iterable[dict[str, Any]]) -> int:
    """Write records as JSONL and reject duplicate stable face keys."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    seen: set[tuple[str, str, int]] = set()
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in records:
            key = (str(row["dataset"]), str(row["part_id"]), int(row["face_id"]))
            if key in seen:
                raise ValueError(f"Duplicate face key while writing records: {key}")
            seen.add(key)
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def _face_neighborhoods(record: PartRecord) -> tuple[list[list[int]], list[list[int]]]:
    neighbors: list[set[int]] = [set() for _ in range(record.num_faces)]
    relations: list[set[int]] = [set() for _ in range(record.num_faces)]
    for edge, edge_type in zip(record.edges.tolist(), record.edge_type.tolist()):
        src, dst = int(edge[0]), int(edge[1])
        if src == dst:
            continue
        neighbors[src].add(dst)
        neighbors[dst].add(src)
        relations[src].add(int(edge_type))
        relations[dst].add(int(edge_type))
    return (
        [sorted(items) for items in neighbors],
        [sorted(items) for items in relations],
    )


def _float_row(value: torch.Tensor) -> list[float]:
    return [float(item) for item in value.detach().cpu().tolist()]
