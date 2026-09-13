#!/usr/bin/env python3
"""Run Frozen + Token Adapter + Prototype Matching."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, TypedDict

import torch
import torch.nn.functional as F

from frozen_representation.cache import FrozenEmbeddingCache
from topocalib_uc.calibration.margin import apply_margin_calibration
from topocalib_uc.candidate_pair.pairs import inference_candidate_pairs, training_candidate_pairs
from topocalib_uc.evidence.readout import FaceBatchMeta
from topocalib_uc.metrics.segmentation import accuracy, mean_iou
from topocalib_uc.prototypes.matching import ClassIndex, prototype_logits
from topocalib_uc.prototypes.sdbscan_backend import (
    SDBSCANConfig,
    SDBSCAN_DEFAULT_MIN_PTS,
    SDBSCAN_DEFAULT_N_PROJ,
    SDBSCAN_DEFAULT_N_THREADS,
    SDBSCAN_DEFAULT_TOP_K,
    SDBSCAN_DEFAULT_TOP_M,
)
from topocalib_uc.prototypes.strategies import (
    PrototypeStrategyConfig,
    build_prototypes_with_strategy,
    leave_one_out_logits_with_strategy,
)
from topocalib_uc.tokenization.adapter import SurfaceTokenAdapter
from topocalib_uc.tokenization.embedding_normalization import (
    apply_embedding_normalizer,
    fit_embedding_normalizer,
    normalizer_json_info,
)
from topocalib_uc.train.partgraph_dataset import PartGraphCache
from topocalib_uc.train.splitting import (
    class_aware_subset,
    dataset_split,
    load_labeled_parts_manifest,
    load_split_manifest,
    random_part_split,
    sample_labeled_parts,
    write_labeled_parts_manifest,
    write_split_manifest,
)


class FaceCollection(TypedDict):
    z: torch.Tensor
    surface_type: torch.Tensor
    y: torch.Tensor
    encoded_y: torch.Tensor
    meta: FaceBatchMeta


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a frozen prototype baseline on PartGraph cache.")
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split-source", choices=["random_filtered", "dataset"], default="random_filtered")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split-seed", type=int, default=None, help="Split seed; defaults to --seed.")
    parser.add_argument("--split-manifest", type=Path, default=None, help="Load a fixed part split JSON manifest.")
    parser.add_argument("--write-split-manifest", type=Path, default=None, help="Write the resolved split manifest.")
    parser.add_argument("--labeled-parts-manifest", type=Path, default=None, help="Load a fixed labeled-parts JSON manifest.")
    parser.add_argument(
        "--write-labeled-parts-manifest",
        type=Path,
        default=None,
        help="Write the resolved labeled-part selection.",
    )
    parser.add_argument("--min-parts-per-class", type=int, default=0)
    parser.add_argument("--labeled-part-budget", required=True)
    parser.add_argument("--ssrl-cache-dir", type=Path, default=None)
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--tau", type=float, default=0.07)
    parser.add_argument("--num-surface-types", type=int, default=5)
    parser.add_argument("--variant", choices=["B0", "B1", "B2", "B3", "B4", "B5"], default="B0")
    parser.add_argument("--beta", type=float, default=0.10)
    parser.add_argument("--part-top-k", type=int, default=5)
    parser.add_argument("--lambda-cal", type=float, default=1.0)
    parser.add_argument("--lambda-part", type=float, default=0.5)
    parser.add_argument("--checkpoint-output", type=Path, default=None)
    parser.add_argument("--prototype-method", choices=["mean", "medoid", "sdbscan"], default="mean")
    parser.add_argument("--medoid-chunk-size", type=int, default=1024)
    parser.add_argument("--sdbscan-eps", type=float, default=None, help="Dataset-selected cosine radius.")
    parser.add_argument("--sdbscan-min-pts", type=int, default=SDBSCAN_DEFAULT_MIN_PTS)
    parser.add_argument("--sdbscan-n-proj", type=int, default=SDBSCAN_DEFAULT_N_PROJ)
    parser.add_argument("--sdbscan-top-k", type=int, default=SDBSCAN_DEFAULT_TOP_K)
    parser.add_argument("--sdbscan-top-m", type=int, default=SDBSCAN_DEFAULT_TOP_M)
    parser.add_argument("--sdbscan-n-threads", type=int, default=SDBSCAN_DEFAULT_N_THREADS)
    parser.add_argument(
        "--sdbscan-random-seed",
        type=int,
        default=None,
        help="Defaults to the run --seed.",
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    result = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(_json_safe(result), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(_json_safe(result), ensure_ascii=False, sort_keys=True))
    return 0


def run(args: argparse.Namespace) -> dict[str, Any]:
    _ensure_defaults(args)
    _log(f"start prototype variant={args.variant} cache={args.cache_dir} embeddings={args.ssrl_cache_dir} output={args.output}")
    torch.manual_seed(args.seed)
    cache = PartGraphCache(args.cache_dir)
    _log(f"loaded PartGraph parts={len(cache.records)} classes={cache.classes}")
    class_index = ClassIndex(cache.classes)
    variant = _variant_from_args(args)
    prototype_config = _prototype_config_from_args(args)

    selected_part_ids: list[str] | None = None
    if args.split_manifest is not None:
        if args.min_parts_per_class > 0:
            raise ValueError("--min-parts-per-class cannot be combined with --split-manifest")
        split = load_split_manifest(args.split_manifest, cache)
    else:
        if args.min_parts_per_class > 0:
            selected_part_ids = class_aware_subset(
                cache,
                min_parts_per_class=args.min_parts_per_class,
                seed=args.split_seed,
            )
        if args.split_source == "dataset":
            split = dataset_split(cache)
            if selected_part_ids is not None:
                selected = set(selected_part_ids)
                split = type(split)(
                    train=sorted(set(split.train) & selected),
                    val=sorted(set(split.val) & selected),
                    test=sorted(set(split.test) & selected),
                )
        else:
            split = random_part_split(
                cache,
                part_ids=selected_part_ids,
                train_ratio=args.train_ratio,
                val_ratio=args.val_ratio,
                test_ratio=args.test_ratio,
                seed=args.split_seed,
                require_train_all_classes=True,
            )
    if args.write_split_manifest is not None:
        write_split_manifest(args.write_split_manifest, cache, split)

    budget = _resolve_budget(args.labeled_part_budget, split.train)
    if args.labeled_parts_manifest is not None:
        labeled_parts = load_labeled_parts_manifest(
            args.labeled_parts_manifest,
            cache,
            split,
            expected_budget=budget,
        )
    else:
        labeled_parts = sample_labeled_parts(
            cache,
            train_part_ids=split.train,
            budget=budget,
            seed=args.seed,
            required_classes=cache.classes,
        )
    if args.write_labeled_parts_manifest is not None:
        write_labeled_parts_manifest(args.write_labeled_parts_manifest, cache, split, labeled_parts)
    _log(f"split train={len(split.train)} val={len(split.val)} test={len(split.test)} labeled={len(labeled_parts)}")

    z_by_part, embedding_dim, embedding_normalizer = _build_frozen_embeddings(cache, split.train, args)
    _log(
        f"loaded SSRL embeddings parts={len(z_by_part)} dim={embedding_dim} "
        f"normalization={embedding_normalizer['type']}"
    )
    model = SurfaceTokenAdapter(embedding_dim, args.hidden_dim, args.num_surface_types)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    labeled_batch = _collect_faces(cache, z_by_part, labeled_parts, class_index)
    best_state: dict[str, torch.Tensor] | None = None
    best_monitor = -1.0
    best_val_miou = -1.0
    best_epoch = -1
    epochs_without_improvement = 0
    stopped_early = False
    history: list[dict[str, float]] = []
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        tokens = model(labeled_batch["z"], labeled_batch["surface_type"])
        logits = leave_one_out_logits_with_strategy(
            tokens,
            labeled_batch["encoded_y"],
            num_classes=class_index.num_classes,
            tau=args.tau,
            config=prototype_config,
        )
        if variant.use_train_calibration:
            probs = F.softmax(logits, dim=1)
            pair_a, pair_b, margin = training_candidate_pairs(probs, labeled_batch["encoded_y"])
            logits = apply_margin_calibration(
                logits,
                probs,
                pair_a,
                pair_b,
                margin,
                labeled_batch["meta"],
                use_local_evidence=variant.use_local_evidence,
                use_ambiguity_gate=variant.use_ambiguity_gate,
                use_part_prior=variant.use_part_prior,
                beta=args.beta,
                lambda_cal=args.lambda_cal,
                lambda_part=args.lambda_part,
                top_k=args.part_top_k,
            )
        loss = F.cross_entropy(logits, labeled_batch["encoded_y"])
        loss.backward()
        optimizer.step()

        train_eval = _evaluate(cache, z_by_part, model, labeled_parts, labeled_parts, class_index, args.tau, args, variant, prototype_config)
        val_eval = _evaluate(cache, z_by_part, model, labeled_parts, split.val, class_index, args.tau, args, variant, prototype_config)
        history.append({"epoch": float(epoch), "loss": float(loss.item()), "val_miou": val_eval["miou"]})
        _log(f"epoch={epoch} loss={loss.item():.6f} train_miou={train_eval['miou']:.6f} val_miou={val_eval['miou']:.6f}")
        has_val = bool(split.val)
        monitor = val_eval["miou"] if has_val else train_eval["miou"]
        if monitor >= best_monitor:
            best_epoch = epoch
            best_monitor = monitor
            best_val_miou = val_eval["miou"]
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if has_val and args.patience > 0 and epochs_without_improvement >= args.patience:
            stopped_early = True
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    checkpoint_path, prototype_info = _write_checkpoint(
        args,
        cache,
        z_by_part,
        model,
        labeled_parts,
        class_index,
        embedding_dim,
        variant,
        embedding_normalizer,
        prototype_config,
    )
    metrics = {
        "train_labeled": _evaluate(cache, z_by_part, model, labeled_parts, labeled_parts, class_index, args.tau, args, variant, prototype_config),
        "val": _evaluate(cache, z_by_part, model, labeled_parts, split.val, class_index, args.tau, args, variant, prototype_config),
        "test": _evaluate(cache, z_by_part, model, labeled_parts, split.test, class_index, args.tau, args, variant, prototype_config),
    }
    _log(f"done test_miou={metrics['test']['miou']:.6f} test_accuracy={metrics['test']['accuracy']:.6f}")
    return {
        "method": "frozen_proto" if variant.name == "B0" else "topocalib_uc",
        "cache_dir": str(args.cache_dir),
        "classes": cache.classes,
        "variant": variant.as_dict(),
        "prototype_construction": prototype_info,
        "config": vars(args),
        "frozen_embedding": {
            "source": "ssrl_cache" if args.ssrl_cache_dir is not None else "partgraph_face_features",
            "embedding_dim": embedding_dim,
            "ssrl_cache_dir": str(args.ssrl_cache_dir) if args.ssrl_cache_dir is not None else None,
            "normalization": normalizer_json_info(embedding_normalizer),
        },
        "split": split.as_dict(),
        "selected_part_ids": selected_part_ids,
        "labeled_parts": labeled_parts,
        "checkpoint": str(checkpoint_path) if checkpoint_path is not None else None,
        "metrics": metrics,
        "history_tail": history[-10:],
        "training": {
            "epochs_ran": len(history),
            "best_epoch": best_epoch,
            "best_monitor_miou": best_monitor,
            "best_val_miou": best_val_miou,
            "stopped_early": stopped_early,
            "patience": args.patience,
        },
    }


def _collect_faces(
    cache: PartGraphCache,
    z_by_part: dict[str, torch.Tensor],
    part_ids: list[str],
    class_index: ClassIndex,
) -> FaceCollection:
    records = [cache.by_part_id(part_id) for part_id in part_ids]
    part_index_chunks: list[torch.Tensor] = []
    edge_chunks: list[torch.Tensor] = []
    edge_type_chunks: list[torch.Tensor] = []
    offset = 0
    for part_idx, record in enumerate(records):
        part_index_chunks.append(torch.full((record.num_faces,), part_idx, dtype=torch.long))
        if record.edges.numel():
            edge_chunks.append(record.edges + offset)
            edge_type_chunks.append(record.edge_type)
        offset += record.num_faces
    z = torch.cat([z_by_part[record.part_id] for record in records], dim=0)
    surface_type = torch.cat([record.surface_type for record in records], dim=0)
    y = torch.cat([record.y for record in records], dim=0)
    edges = torch.cat(edge_chunks, dim=0) if edge_chunks else torch.zeros((0, 2), dtype=torch.long)
    edge_type = torch.cat(edge_type_chunks, dim=0) if edge_type_chunks else torch.zeros((0,), dtype=torch.long)
    meta = FaceBatchMeta(part_index=torch.cat(part_index_chunks, dim=0), edges=edges, edge_type=edge_type)
    return {"z": z, "surface_type": surface_type, "y": y, "encoded_y": class_index.encode(y), "meta": meta}


@torch.no_grad()
def _evaluate(
    cache: PartGraphCache,
    z_by_part: dict[str, torch.Tensor],
    model: SurfaceTokenAdapter,
    support_part_ids: list[str],
    eval_part_ids: list[str],
    class_index: ClassIndex,
    tau: float,
    args: argparse.Namespace,
    variant,
    prototype_config: PrototypeStrategyConfig,
) -> dict[str, float]:
    if not eval_part_ids:
        return {"accuracy": 0.0, "miou": 0.0, "num_faces": 0.0}
    model.eval()
    support = _collect_faces(cache, z_by_part, support_part_ids, class_index)
    support_tokens = model(support["z"], support["surface_type"])
    prototypes, _, _ = build_prototypes_with_strategy(
        support_tokens,
        support["encoded_y"],
        class_index.num_classes,
        prototype_config,
    )
    eval_batch = _collect_faces(cache, z_by_part, eval_part_ids, class_index)
    eval_tokens = model(eval_batch["z"], eval_batch["surface_type"])
    logits = prototype_logits(eval_tokens, prototypes, tau)
    if variant.use_inference_calibration:
        probs = F.softmax(logits, dim=1)
        pair_a, pair_b, margin = inference_candidate_pairs(probs)
        logits = apply_margin_calibration(
            logits,
            probs,
            pair_a,
            pair_b,
            margin,
            eval_batch["meta"],
            use_local_evidence=variant.use_local_evidence,
            use_ambiguity_gate=variant.use_ambiguity_gate,
            use_part_prior=variant.use_part_prior,
            beta=args.beta,
            lambda_cal=args.lambda_cal,
            lambda_part=args.lambda_part,
            top_k=args.part_top_k,
        )
    pred_encoded = logits.argmax(dim=1)
    pred = torch.as_tensor([class_index.classes[int(idx)] for idx in pred_encoded.tolist()], dtype=torch.long)
    return {
        "accuracy": accuracy(pred, eval_batch["y"]),
        "miou": mean_iou(pred, eval_batch["y"], class_index.classes),
        "num_faces": float(eval_batch["y"].numel()),
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


def _build_frozen_embeddings(
    cache: PartGraphCache,
    train_part_ids: list[str],
    args: argparse.Namespace,
) -> tuple[dict[str, torch.Tensor], int, dict[str, Any]]:
    if getattr(args, "ssrl_cache_dir", None) is None:
        z_by_part = {record.part_id: record.face_features_raw.float() for record in cache.records}
    else:
        embedding_cache = FrozenEmbeddingCache(args.ssrl_cache_dir)
        z_by_part = embedding_cache.transform_cache(cache)
    embedding_dim = _single_embedding_dim(z_by_part)
    normalizer = fit_embedding_normalizer(z_by_part, train_part_ids)
    return apply_embedding_normalizer(z_by_part, normalizer), embedding_dim, normalizer


@torch.no_grad()
def _write_checkpoint(
    args: argparse.Namespace,
    cache: PartGraphCache,
    z_by_part: dict[str, torch.Tensor],
    model: SurfaceTokenAdapter,
    labeled_parts: list[str],
    class_index: ClassIndex,
    embedding_dim: int,
    variant,
    embedding_normalizer: dict[str, Any],
    prototype_config: PrototypeStrategyConfig,
) -> tuple[Path | None, dict[str, Any]]:
    output = getattr(args, "checkpoint_output", None)
    if output is None and getattr(args, "output", None) is not None:
        output = Path(args.output).with_suffix(".pt")
    model.eval()
    support = _collect_faces(cache, z_by_part, labeled_parts, class_index)
    support_tokens = model(support["z"], support["surface_type"])
    prototypes, prototype_counts, prototype_info = build_prototypes_with_strategy(
        support_tokens,
        support["encoded_y"],
        class_index.num_classes,
        prototype_config,
    )
    if output is None:
        return None, prototype_info
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "checkpoint_type": "topocalib_uc_downstream",
            "method": "frozen_proto" if variant.name == "B0" else "topocalib_uc",
            "model_state_dict": model.state_dict(),
            "prototypes": prototypes.detach().cpu(),
            "prototype_counts": prototype_counts.detach().cpu(),
            "prototype_construction": prototype_info,
            "classes": list(class_index.classes),
            "embedding_dim": int(embedding_dim),
            "embedding_normalization": embedding_normalizer,
            "hidden_dim": int(args.hidden_dim),
            "num_surface_types": int(args.num_surface_types),
            "tau": float(args.tau),
            "variant": variant.as_dict(),
            "calibration": {
                "beta": float(args.beta),
                "part_top_k": int(args.part_top_k),
                "lambda_cal": float(args.lambda_cal),
                "lambda_part": float(args.lambda_part),
            },
            "labeled_parts": list(labeled_parts),
            "config": _json_safe(vars(args)),
        },
        output,
    )
    return output, prototype_info


def _single_embedding_dim(z_by_part: dict[str, torch.Tensor]) -> int:
    dims = {int(z.shape[1]) for z in z_by_part.values()}
    if len(dims) != 1:
        raise ValueError(f"All frozen embeddings must have the same embedding dimension, got {sorted(dims)}")
    return dims.pop()


def _variant_from_args(args: argparse.Namespace):
    from topocalib_uc.train.variants import variant_config
    return variant_config(args.variant)


def _ensure_defaults(args: argparse.Namespace) -> None:
    for name, default in [
        ("split_source", "random_filtered"),
        ("train_ratio", 0.7),
        ("val_ratio", 0.15),
        ("test_ratio", 0.15),
        ("seed", 0),
        ("split_seed", None),
        ("split_manifest", None),
        ("write_split_manifest", None),
        ("labeled_parts_manifest", None),
        ("write_labeled_parts_manifest", None),
        ("min_parts_per_class", 0),
        ("ssrl_cache_dir", None),
        ("embedding_dim", 256),
        ("hidden_dim", 128),
        ("epochs", 200),
        ("patience", 20),
        ("lr", 1e-3),
        ("weight_decay", 1e-4),
        ("tau", 0.07),
        ("num_surface_types", 5),
        ("variant", "B0"),
        ("beta", 0.10),
        ("part_top_k", 5),
        ("lambda_cal", 1.0),
        ("lambda_part", 0.5),
        ("checkpoint_output", None),
        ("prototype_method", "mean"),
        ("medoid_chunk_size", 1024),
        ("sdbscan_eps", None),
        ("sdbscan_min_pts", SDBSCAN_DEFAULT_MIN_PTS),
        ("sdbscan_n_proj", SDBSCAN_DEFAULT_N_PROJ),
        ("sdbscan_top_k", SDBSCAN_DEFAULT_TOP_K),
        ("sdbscan_top_m", SDBSCAN_DEFAULT_TOP_M),
        ("sdbscan_n_threads", SDBSCAN_DEFAULT_N_THREADS),
        ("sdbscan_random_seed", None),
    ]:
        if not hasattr(args, name):
            setattr(args, name, default)
    if args.split_seed is None:
        args.split_seed = args.seed
    if args.sdbscan_random_seed is None:
        args.sdbscan_random_seed = args.seed


def _prototype_config_from_args(args: argparse.Namespace) -> PrototypeStrategyConfig:
    if args.prototype_method != "sdbscan":
        return PrototypeStrategyConfig(
            method=args.prototype_method,
            medoid_chunk_size=args.medoid_chunk_size,
        )
    names = (
        "sdbscan_eps",
        "sdbscan_min_pts",
        "sdbscan_n_proj",
        "sdbscan_top_k",
        "sdbscan_top_m",
        "sdbscan_n_threads",
        "sdbscan_random_seed",
    )
    missing = [f"--{name.replace('_', '-')}" for name in names if getattr(args, name) is None]
    if missing:
        raise ValueError(f"sDBSCAN prototype construction requires explicit parameters: {', '.join(missing)}")
    return PrototypeStrategyConfig(
        method="sdbscan",
        medoid_chunk_size=args.medoid_chunk_size,
        sdbscan=SDBSCANConfig(
            eps=float(args.sdbscan_eps),
            min_pts=int(args.sdbscan_min_pts),
            n_proj=int(args.sdbscan_n_proj),
            top_k=int(args.sdbscan_top_k),
            top_m=int(args.sdbscan_top_m),
            n_threads=int(args.sdbscan_n_threads),
            random_seed=int(args.sdbscan_random_seed),
        ),
    )


def _resolve_budget(value: int | str, train_part_ids: list[str]) -> int:
    if isinstance(value, str) and value.lower() == "all":
        return len(train_part_ids)
    return int(value)


def _log(message: str) -> None:
    print(f"[run_frozen_proto] {message}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
