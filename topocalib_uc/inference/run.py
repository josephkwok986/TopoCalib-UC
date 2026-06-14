#!/usr/bin/env python3
"""Run downstream TopoCalib-UC/Frozen inference from a saved checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from frozen_representation.cache import FrozenEmbeddingCache
from topocalib_uc.calibration.margin import apply_margin_calibration
from topocalib_uc.candidate_pair.pairs import inference_candidate_pairs
from topocalib_uc.evidence.readout import FaceBatchMeta
from topocalib_uc.metrics.segmentation import accuracy, mean_iou
from topocalib_uc.prototypes.matching import ClassIndex, prototype_logits
from topocalib_uc.tokenization.adapter import SurfaceTokenAdapter
from topocalib_uc.tokenization.embedding_normalization import apply_embedding_normalizer
from topocalib_uc.train.partgraph_dataset import PartGraphCache, PartRecord
from topocalib_uc.train.run_frozen_linear import FrozenLinearModel
from topocalib_uc.train.variants import VariantConfig


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--ssrl-cache-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--part-id", action="append", default=None, help="Optional part_id to infer. Can be repeated.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--save-logits", action="store_true")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    result = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(_json_safe(result), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(_json_safe({"output": args.output, "metrics": result.get("metrics", {})}), ensure_ascii=False, sort_keys=True))
    return 0


@torch.no_grad()
def run(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    method = str(checkpoint.get("method", ""))
    cache = PartGraphCache(args.cache_dir)
    class_index = ClassIndex([int(x) for x in checkpoint["classes"]])
    records = _select_records(cache, args.part_id)
    z_by_part = _load_embeddings(cache, args.ssrl_cache_dir, checkpoint)
    if method == "frozen_linear":
        rows = _infer_linear(checkpoint, records, z_by_part, class_index, device, save_logits=args.save_logits)
    elif method in {"frozen_proto", "topocalib_uc"}:
        rows = _infer_proto(checkpoint, records, z_by_part, class_index, device, save_logits=args.save_logits)
    else:
        raise ValueError(f"Unsupported checkpoint method {method!r}")
    metrics = _overall_metrics(rows, class_index.classes)
    return {
        "method": method,
        "checkpoint": str(args.checkpoint),
        "cache_dir": str(args.cache_dir),
        "ssrl_cache_dir": str(args.ssrl_cache_dir) if args.ssrl_cache_dir is not None else None,
        "classes": class_index.classes,
        "num_parts": len(rows),
        "metrics": metrics,
        "parts": rows,
    }


def _select_records(cache: PartGraphCache, part_ids: list[str] | None) -> list[PartRecord]:
    if not part_ids:
        return cache.records
    return [cache.by_part_id(part_id) for part_id in part_ids]


def _load_embeddings(
    cache: PartGraphCache,
    ssrl_cache_dir: Path | None,
    checkpoint: dict[str, Any],
) -> dict[str, torch.Tensor]:
    if ssrl_cache_dir is None:
        configured = checkpoint.get("config", {}).get("ssrl_cache_dir")
        ssrl_cache_dir = Path(configured) if configured else None
    if ssrl_cache_dir is None:
        z_by_part = {record.part_id: record.face_features_raw.float() for record in cache.records}
    else:
        embedding_cache = FrozenEmbeddingCache(ssrl_cache_dir)
        z_by_part = embedding_cache.transform_cache(cache)
    expected_dim = int(checkpoint["embedding_dim"])
    dims = {int(z.shape[1]) for z in z_by_part.values()}
    if dims != {expected_dim}:
        raise ValueError(f"Checkpoint expects embedding_dim={expected_dim}, got cache dims={sorted(dims)}")
    return apply_embedding_normalizer(z_by_part, checkpoint.get("embedding_normalization"))


def _infer_linear(
    checkpoint: dict[str, Any],
    records: list[PartRecord],
    z_by_part: dict[str, torch.Tensor],
    class_index: ClassIndex,
    device: torch.device,
    *,
    save_logits: bool,
) -> list[dict[str, Any]]:
    model = FrozenLinearModel(
        int(checkpoint["embedding_dim"]),
        int(checkpoint["hidden_dim"]),
        int(checkpoint["num_surface_types"]),
        class_index.num_classes,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    rows = []
    for record in records:
        z = z_by_part[record.part_id].to(device)
        surface_type = record.surface_type.to(device)
        logits = model(z, surface_type)
        rows.append(_part_result(record, logits, class_index, save_logits=save_logits))
    return rows


def _infer_proto(
    checkpoint: dict[str, Any],
    records: list[PartRecord],
    z_by_part: dict[str, torch.Tensor],
    class_index: ClassIndex,
    device: torch.device,
    *,
    save_logits: bool,
) -> list[dict[str, Any]]:
    model = SurfaceTokenAdapter(
        int(checkpoint["embedding_dim"]),
        int(checkpoint["hidden_dim"]),
        int(checkpoint["num_surface_types"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    prototypes = checkpoint["prototypes"].to(device)
    tau = float(checkpoint["tau"])
    variant = _variant_from_checkpoint(checkpoint)
    calibration = checkpoint.get("calibration", {})
    rows = []
    for record in records:
        z = z_by_part[record.part_id].to(device)
        surface_type = record.surface_type.to(device)
        tokens = model(z, surface_type)
        logits = prototype_logits(tokens, prototypes, tau)
        if variant.use_inference_calibration:
            probs = F.softmax(logits, dim=1)
            pair_a, pair_b, margin = inference_candidate_pairs(probs)
            logits = apply_margin_calibration(
                logits,
                probs,
                pair_a,
                pair_b,
                margin,
                _meta_for_record(record, device),
                use_local_evidence=variant.use_local_evidence,
                use_ambiguity_gate=variant.use_ambiguity_gate,
                use_part_prior=variant.use_part_prior,
                beta=float(calibration.get("beta", 0.10)),
                lambda_cal=float(calibration.get("lambda_cal", 1.0)),
                lambda_part=float(calibration.get("lambda_part", 0.5)),
                top_k=int(calibration.get("part_top_k", 5)),
            )
        rows.append(_part_result(record, logits, class_index, save_logits=save_logits))
    return rows


def _variant_from_checkpoint(checkpoint: dict[str, Any]) -> VariantConfig:
    raw = checkpoint.get("variant", {})
    return VariantConfig(
        name=str(raw.get("name", "B0")),
        use_train_calibration=bool(raw.get("use_train_calibration", False)),
        use_inference_calibration=bool(raw.get("use_inference_calibration", False)),
        use_local_evidence=bool(raw.get("use_local_evidence", False)),
        use_ambiguity_gate=bool(raw.get("use_ambiguity_gate", False)),
        use_part_prior=bool(raw.get("use_part_prior", False)),
    )


def _meta_for_record(record: PartRecord, device: torch.device) -> FaceBatchMeta:
    return FaceBatchMeta(
        part_index=torch.zeros(record.num_faces, dtype=torch.long, device=device),
        edges=record.edges.to(device),
        edge_type=record.edge_type.to(device),
    )


def _part_result(record: PartRecord, logits: torch.Tensor, class_index: ClassIndex, *, save_logits: bool) -> dict[str, Any]:
    pred_encoded = logits.argmax(dim=1).detach().cpu()
    pred = torch.as_tensor([class_index.classes[int(idx)] for idx in pred_encoded.tolist()], dtype=torch.long)
    target = record.y.detach().cpu().long()
    result: dict[str, Any] = {
        "part_id": record.part_id,
        "num_faces": record.num_faces,
        "pred": [int(x) for x in pred.tolist()],
        "target": [int(x) for x in target.tolist()],
        "metrics": {
            "accuracy": accuracy(pred, target),
            "miou": mean_iou(pred, target, class_index.classes),
            "num_faces": float(target.numel()),
        },
    }
    if save_logits:
        result["logits"] = logits.detach().cpu().tolist()
    return result


def _overall_metrics(rows: list[dict[str, Any]], classes: list[int]) -> dict[str, float]:
    if not rows:
        return {"accuracy": 0.0, "miou": 0.0, "num_faces": 0.0}
    pred = torch.as_tensor([label for row in rows for label in row["pred"]], dtype=torch.long)
    target = torch.as_tensor([label for row in rows for label in row["target"]], dtype=torch.long)
    return {
        "accuracy": accuracy(pred, target),
        "miou": mean_iou(pred, target, classes),
        "num_faces": float(target.numel()),
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    return value


if __name__ == "__main__":
    raise SystemExit(main())
