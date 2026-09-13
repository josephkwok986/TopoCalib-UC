#!/usr/bin/env python3
"""Build fixed workloads and benchmark downstream cached-feature inference."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import random
import time
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F

from baselines.ssrl_mrgcn.model import SSRLMRGCN
from frozen_representation.cache import FrozenEmbeddingCache
from topocalib_uc.calibration.margin import apply_margin_calibration
from topocalib_uc.candidate_pair.pairs import inference_candidate_pairs
from topocalib_uc.evidence.readout import FaceBatchMeta
from topocalib_uc.prototypes.matching import prototype_logits
from topocalib_uc.tokenization.adapter import SurfaceTokenAdapter
from topocalib_uc.tokenization.embedding_normalization import apply_embedding_normalizer
from topocalib_uc.train.partgraph_dataset import PartGraphCache, PartRecord
from topocalib_uc.train.splitting import load_split_manifest
from topocalib_uc.train.variants import VariantConfig


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build-workload", help="Build a fixed part workload manifest.")
    build.add_argument("--cache-dir", type=Path, required=True)
    build.add_argument("--split-manifest", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--name", required=True)
    build.add_argument("--num-parts", type=int, required=True)
    build.add_argument("--min-faces", type=int, required=True)
    build.add_argument("--max-faces", type=int, required=True)
    build.add_argument("--seed", type=int, default=0)
    build.add_argument("--allow-repeats", action="store_true")

    run = subparsers.add_parser("run", help="Benchmark one or more downstream checkpoints.")
    run.add_argument("--cache-dir", type=Path, required=True)
    run.add_argument("--ssrl-cache-dir", type=Path, required=True)
    run.add_argument("--workload-manifest", type=Path, required=True)
    run.add_argument("--split-manifest", type=Path, required=True)
    run.add_argument(
        "--checkpoint",
        action="append",
        required=True,
        metavar="NAME=PATH",
        help="Method display name and checkpoint path; can be repeated.",
    )
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--device", default="cuda")
    run.add_argument("--batch-size", type=int, default=16)
    run.add_argument("--warmup-repetitions", type=int, default=1)
    run.add_argument("--repetitions", type=int, default=5)
    run.add_argument("--std-ddof", type=int, choices=[0, 1], default=1)
    run.add_argument(
        "--cpu-smoke",
        action="store_true",
        help="Allow CPU timing for implementation checks; CUDA memory fields remain unavailable.",
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    if args.command == "build-workload":
        return build_workload_command(args)
    if args.command == "run":
        return benchmark_command(args)
    raise AssertionError(args.command)


def build_workload_command(args: argparse.Namespace) -> int:
    cache = PartGraphCache(args.cache_dir)
    split = load_split_manifest(args.split_manifest, cache)
    manifest = build_workload_manifest(
        cache,
        split.test,
        name=args.name,
        num_parts=args.num_parts,
        min_faces=args.min_faces,
        max_faces=args.max_faces,
        seed=args.seed,
        allow_repeats=args.allow_repeats,
    )
    manifest["split_manifest"] = str(args.split_manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"parts": len(manifest["part_ids"]), "output": str(args.output)}, sort_keys=True))
    return 0


def benchmark_command(args: argparse.Namespace) -> int:
    if args.batch_size <= 0 or args.repetitions <= 0 or args.warmup_repetitions < 0:
        raise ValueError("batch size and repetitions must be positive; warm-up repetitions must be non-negative")
    device = torch.device(args.device)
    if device.type != "cuda" and not args.cpu_smoke:
        raise ValueError("CUDA is required for publication-protocol timing and memory; use --cpu-smoke only for checks")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    cache = PartGraphCache(args.cache_dir)
    split = load_split_manifest(args.split_manifest, cache)
    manifest = load_workload_manifest(args.workload_manifest, cache, allowed_part_ids=set(split.test))
    embedding_cache = FrozenEmbeddingCache(args.ssrl_cache_dir)
    raw_embeddings = embedding_cache.transform_cache(cache)
    records = [cache.by_part_id(part_id) for part_id in manifest["part_ids"]]
    checkpoints = parse_checkpoint_arguments(args.checkpoint)
    results = []
    for display_name, checkpoint_path in checkpoints:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        runner, parameter_count = build_runner(checkpoint, records, raw_embeddings, device, args.batch_size)
        result = measure_runner(
            runner,
            num_parts=len(records),
            device=device,
            warmup_repetitions=args.warmup_repetitions,
            repetitions=args.repetitions,
            std_ddof=args.std_ddof,
        )
        result.update(
            {
                "name": display_name,
                "method": str(checkpoint.get("method", "")),
                "variant": str(checkpoint.get("variant", {}).get("name", "")),
                "checkpoint": str(checkpoint_path),
                "checkpoint_sha256": file_sha256(checkpoint_path),
                "trainable_downstream_parameters": parameter_count,
            }
        )
        results.append(result)
    output = {
        "schema_version": 1,
        "publication_protocol": (
            device.type == "cuda"
            and args.batch_size == 16
            and args.repetitions == 5
            and args.warmup_repetitions > 0
            and len(records) == 1000
        ),
        "protocol": {
            "input": "device-resident cached face embeddings",
            "timed_region": "downstream forward through argmax",
            "excluded": [
                "frozen encoder",
                "B-Rep parsing",
                "data loading",
                "host-to-device transfer",
                "metric computation",
                "serialization",
                "prototype construction",
            ],
            "batch_size_parts": args.batch_size,
            "num_batches_per_workload": math.ceil(len(records) / args.batch_size),
            "warmup_repetitions": args.warmup_repetitions,
            "repetitions": args.repetitions,
            "std_ddof": args.std_ddof,
        },
        "workload_manifest": str(args.workload_manifest),
        "split_manifest": str(args.split_manifest),
        "workload": manifest,
        "environment": environment_info(device),
        "methods": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"methods": len(results), "parts": len(records), "output": str(args.output)}, sort_keys=True))
    return 0


def build_workload_manifest(
    cache: PartGraphCache,
    split_part_ids: list[str],
    *,
    name: str,
    num_parts: int,
    min_faces: int,
    max_faces: int,
    seed: int,
    allow_repeats: bool,
) -> dict[str, Any]:
    if num_parts <= 0:
        raise ValueError("num_parts must be positive")
    if min_faces <= 0 or max_faces < min_faces:
        raise ValueError("face range must satisfy 0 < min_faces <= max_faces")
    if len(split_part_ids) != len(set(split_part_ids)):
        raise ValueError("split_part_ids contains duplicates")
    candidates = [
        part_id
        for part_id in split_part_ids
        if min_faces <= cache.by_part_id(part_id).num_faces <= max_faces
    ]
    if not candidates:
        raise ValueError(f"no test parts fall in face range [{min_faces}, {max_faces}]")
    rng = random.Random(seed)
    candidates.sort()
    if num_parts <= len(candidates):
        selected = rng.sample(candidates, num_parts)
    elif allow_repeats:
        selected = candidates[:]
        rng.shuffle(selected)
        selected.extend(rng.choice(candidates) for _ in range(num_parts - len(selected)))
    else:
        raise ValueError(
            f"requested {num_parts} parts but only {len(candidates)} are available; "
            "pass --allow-repeats only if repeated sampling is the intended protocol"
        )
    face_counts = [cache.by_part_id(part_id).num_faces for part_id in selected]
    datasets = {cache.by_part_id(part_id).dataset for part_id in selected}
    if len(datasets) != 1:
        raise ValueError(f"workload must contain one dataset, got {sorted(datasets)}")
    return {
        "schema_version": 1,
        "kind": "inference_workload",
        "dataset": datasets.pop(),
        "split": "test",
        "name": name,
        "sampling_seed": int(seed),
        "requested_num_parts": num_parts,
        "candidate_parts": len(candidates),
        "allow_repeats": bool(allow_repeats),
        "has_repeats": len(selected) != len(set(selected)),
        "requested_face_range": [min_faces, max_faces],
        "actual_face_range": [min(face_counts), max(face_counts)],
        "mean_faces_per_part": sum(face_counts) / len(face_counts),
        "total_faces": sum(face_counts),
        "part_ids": selected,
    }


def load_workload_manifest(
    path: Path,
    cache: PartGraphCache,
    *,
    allowed_part_ids: set[str] | None = None,
) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("schema_version") != 1 or raw.get("kind") != "inference_workload":
        raise ValueError(f"{path}: unsupported workload manifest")
    part_ids = [str(value) for value in raw.get("part_ids", [])]
    if not part_ids:
        raise ValueError(f"{path}: workload is empty")
    if int(raw.get("requested_num_parts", -1)) != len(part_ids):
        raise ValueError(f"{path}: requested_num_parts metadata is inconsistent")
    unknown = set(part_ids) - set(cache.part_ids)
    if unknown:
        raise ValueError(f"{path}: unknown part IDs {sorted(unknown)[:10]}")
    outside_split = set(part_ids) - allowed_part_ids if allowed_part_ids is not None else set()
    if outside_split:
        raise ValueError(f"{path}: part IDs outside the fixed test split: {sorted(outside_split)[:10]}")
    has_repeats = len(part_ids) != len(set(part_ids))
    if has_repeats != bool(raw.get("has_repeats")):
        raise ValueError(f"{path}: has_repeats metadata is inconsistent")
    if has_repeats and not bool(raw.get("allow_repeats")):
        raise ValueError(f"{path}: repeated part IDs are not allowed")
    requested_range = [int(value) for value in raw["requested_face_range"]]
    face_counts = [cache.by_part_id(part_id).num_faces for part_id in part_ids]
    if any(not requested_range[0] <= count <= requested_range[1] for count in face_counts):
        raise ValueError(f"{path}: workload includes parts outside the declared face range")
    datasets = {cache.by_part_id(part_id).dataset for part_id in part_ids}
    if datasets != {str(raw.get("dataset", ""))}:
        raise ValueError(f"{path}: dataset metadata does not match cached parts")
    normalized = dict(raw)
    normalized["part_ids"] = part_ids
    normalized["actual_face_range"] = [min(face_counts), max(face_counts)]
    normalized["mean_faces_per_part"] = sum(face_counts) / len(face_counts)
    normalized["total_faces"] = sum(face_counts)
    return normalized


def parse_checkpoint_arguments(values: list[str]) -> list[tuple[str, Path]]:
    output = []
    seen = set()
    for value in values:
        if "=" not in value:
            raise ValueError(f"checkpoint must use NAME=PATH syntax, got {value!r}")
        name, raw_path = value.split("=", 1)
        if not name or not raw_path:
            raise ValueError(f"checkpoint must use non-empty NAME=PATH syntax, got {value!r}")
        if name in seen:
            raise ValueError(f"duplicate checkpoint name {name!r}")
        seen.add(name)
        path = Path(raw_path)
        if not path.is_file():
            raise FileNotFoundError(path)
        output.append((name, path))
    return output


def build_runner(
    checkpoint: dict[str, Any],
    records: list[PartRecord],
    raw_embeddings: dict[str, torch.Tensor],
    device: torch.device,
    batch_size: int,
) -> tuple[Callable[[], None], int]:
    method = str(checkpoint.get("method", ""))
    if method in {"frozen_proto", "topocalib_uc"}:
        return build_proto_runner(checkpoint, records, raw_embeddings, device, batch_size)
    if method == "ssrl_mrgcn":
        return build_mrgcn_runner(checkpoint, records, raw_embeddings, device, batch_size)
    raise ValueError(f"unsupported benchmark checkpoint method {method!r}")


def build_proto_runner(
    checkpoint: dict[str, Any],
    records: list[PartRecord],
    raw_embeddings: dict[str, torch.Tensor],
    device: torch.device,
    batch_size: int,
) -> tuple[Callable[[], None], int]:
    classes = [int(value) for value in checkpoint["classes"]]
    _validate_checkpoint_records(classes, int(checkpoint["embedding_dim"]), records, raw_embeddings)
    normalized = apply_embedding_normalizer(raw_embeddings, checkpoint.get("embedding_normalization"))
    workload_ids = {record.part_id for record in records}
    z_device = {part_id: normalized[part_id].to(device) for part_id in workload_ids}
    model = SurfaceTokenAdapter(
        int(checkpoint["embedding_dim"]),
        int(checkpoint["hidden_dim"]),
        int(checkpoint["num_surface_types"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    prototypes = checkpoint["prototypes"].to(device)
    variant = variant_from_checkpoint(checkpoint)
    calibration = checkpoint.get("calibration", {})
    batches = [
        collate_proto(records[start : start + batch_size], z_device, device)
        for start in range(0, len(records), batch_size)
    ]

    @torch.no_grad()
    def runner() -> None:
        for batch in batches:
            tokens = model(batch["z"], batch["surface_type"])
            logits = prototype_logits(tokens, prototypes, float(checkpoint["tau"]))
            if variant.use_inference_calibration:
                probs = F.softmax(logits, dim=1)
                pair_a, pair_b, margin = inference_candidate_pairs(probs)
                logits = apply_margin_calibration(
                    logits,
                    probs,
                    pair_a,
                    pair_b,
                    margin,
                    batch["meta"],
                    use_local_evidence=variant.use_local_evidence,
                    use_ambiguity_gate=variant.use_ambiguity_gate,
                    use_part_prior=variant.use_part_prior,
                    beta=float(calibration.get("beta", 0.10)),
                    lambda_cal=float(calibration.get("lambda_cal", 1.0)),
                    lambda_part=float(calibration.get("lambda_part", 0.5)),
                    top_k=int(calibration.get("part_top_k", 5)),
                )
            _ = logits.argmax(dim=1)

    return runner, trainable_parameter_count(model)


def build_mrgcn_runner(
    checkpoint: dict[str, Any],
    records: list[PartRecord],
    raw_embeddings: dict[str, torch.Tensor],
    device: torch.device,
    batch_size: int,
) -> tuple[Callable[[], None], int]:
    classes = [int(value) for value in checkpoint["classes"]]
    embedding_dim = int(checkpoint["embedding_dim"])
    _validate_checkpoint_records(classes, embedding_dim, records, raw_embeddings)
    config = checkpoint.get("config", {})
    model = SSRLMRGCN(
        embedding_dim,
        len(classes),
        hidden_dim=int(config.get("hidden_dim", 64)),
        mp_layers=int(config.get("mp_layers", 2)),
        mlp_hidden_dim=int(config.get("mlp_hidden_dim", 64)),
        dropout=float(config.get("dropout", 0.0)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    undirected = str(config.get("undirected", True)).lower() == "true"
    workload_ids = {record.part_id for record in records}
    z_device = {part_id: raw_embeddings[part_id].to(device) for part_id in workload_ids}
    batches = [
        collate_mrgcn(records[start : start + batch_size], z_device, device, undirected)
        for start in range(0, len(records), batch_size)
    ]

    @torch.no_grad()
    def runner() -> None:
        for batch in batches:
            logits = model(batch["z"], batch["edges"])
            _ = logits.argmax(dim=1)

    return runner, trainable_parameter_count(model)


def collate_proto(
    records: list[PartRecord],
    z_by_part: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, Any]:
    edge_chunks = []
    edge_type_chunks = []
    part_index = []
    offset = 0
    for part_index_value, record in enumerate(records):
        part_index.append(torch.full((record.num_faces,), part_index_value, dtype=torch.long, device=device))
        if record.edges.numel():
            edge_chunks.append(record.edges.to(device) + offset)
            edge_type_chunks.append(record.edge_type.to(device))
        offset += record.num_faces
    edges = torch.cat(edge_chunks) if edge_chunks else torch.zeros((0, 2), dtype=torch.long, device=device)
    edge_type = torch.cat(edge_type_chunks) if edge_type_chunks else torch.zeros((0,), dtype=torch.long, device=device)
    return {
        "z": torch.cat([z_by_part[record.part_id] for record in records]),
        "surface_type": torch.cat([record.surface_type.to(device) for record in records]),
        "meta": FaceBatchMeta(part_index=torch.cat(part_index), edges=edges, edge_type=edge_type),
    }


def collate_mrgcn(
    records: list[PartRecord],
    z_by_part: dict[str, torch.Tensor],
    device: torch.device,
    undirected: bool,
) -> dict[str, torch.Tensor]:
    edge_chunks = []
    offset = 0
    for record in records:
        edges = record.edges.to(device)
        if undirected and edges.numel():
            edges = torch.cat([edges, edges[:, [1, 0]]])
        if edges.numel():
            edge_chunks.append(edges + offset)
        offset += record.num_faces
    return {
        "z": torch.cat([z_by_part[record.part_id] for record in records]),
        "edges": torch.cat(edge_chunks) if edge_chunks else torch.zeros((0, 2), dtype=torch.long, device=device),
    }


def measure_runner(
    runner: Callable[[], None],
    *,
    num_parts: int,
    device: torch.device,
    warmup_repetitions: int,
    repetitions: int,
    std_ddof: int,
) -> dict[str, Any]:
    synchronize(device)
    for _ in range(warmup_repetitions):
        runner()
        synchronize(device)
    times_ms = []
    baseline_memory = None
    peak_memory = None
    incremental_memory = None
    for _ in range(repetitions):
        synchronize(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            baseline = int(torch.cuda.memory_allocated(device))
        else:
            baseline = 0
        start = time.perf_counter()
        runner()
        synchronize(device)
        elapsed_ms = 1000.0 * (time.perf_counter() - start)
        times_ms.append(elapsed_ms)
        if device.type == "cuda":
            peak = int(torch.cuda.max_memory_allocated(device))
            increment = max(peak - baseline, 0)
            baseline_memory = baseline if baseline_memory is None else min(baseline_memory, baseline)
            peak_memory = peak if peak_memory is None else max(peak_memory, peak)
            incremental_memory = increment if incremental_memory is None else max(incremental_memory, increment)
    per_part = [value / num_parts for value in times_ms]
    return {
        "num_parts": num_parts,
        "workload_time_ms_values": times_ms,
        "workload_time_ms_mean": sum(times_ms) / len(times_ms),
        "workload_time_ms_std": standard_deviation(times_ms, ddof=std_ddof),
        "amortized_ms_per_part_values": per_part,
        "amortized_ms_per_part_mean": sum(per_part) / len(per_part),
        "amortized_ms_per_part_std": standard_deviation(per_part, ddof=std_ddof),
        "memory_baseline_allocated_bytes": baseline_memory,
        "peak_allocated_bytes": peak_memory,
        "incremental_peak_allocated_bytes": incremental_memory,
        "std_ddof": std_ddof,
    }


def variant_from_checkpoint(checkpoint: dict[str, Any]) -> VariantConfig:
    raw = checkpoint.get("variant", {})
    return VariantConfig(
        name=str(raw.get("name", "B0")),
        use_train_calibration=bool(raw.get("use_train_calibration", False)),
        use_inference_calibration=bool(raw.get("use_inference_calibration", False)),
        use_local_evidence=bool(raw.get("use_local_evidence", False)),
        use_ambiguity_gate=bool(raw.get("use_ambiguity_gate", False)),
        use_part_prior=bool(raw.get("use_part_prior", False)),
    )


def trainable_parameter_count(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def environment_info(device: torch.device) -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
    }


def standard_deviation(values: list[float], *, ddof: int) -> float:
    if len(values) <= ddof:
        return 0.0
    average = sum(values) / len(values)
    return math.sqrt(sum((value - average) ** 2 for value in values) / (len(values) - ddof))


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_checkpoint_records(
    classes: list[int],
    embedding_dim: int,
    records: list[PartRecord],
    embeddings: dict[str, torch.Tensor],
) -> None:
    workload_classes = {int(value) for record in records for value in record.y.tolist()}
    unknown = workload_classes - set(classes)
    if unknown:
        raise ValueError(f"workload labels are absent from checkpoint classes: {sorted(unknown)}")
    dimensions = {int(embeddings[record.part_id].shape[1]) for record in records}
    if dimensions != {embedding_dim}:
        raise ValueError(f"checkpoint embedding_dim={embedding_dim}, workload dimensions={sorted(dimensions)}")


if __name__ == "__main__":
    raise SystemExit(main())
