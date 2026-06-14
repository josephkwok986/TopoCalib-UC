#!/usr/bin/env python3
"""Run SSRL-style frozen encoder + residual MR-GCN segmentation baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
from typing import Any

import torch
import torch.nn.functional as F

from baselines.ssrl_mrgcn.model import SSRLMRGCN
from frozen_representation.cache import FrozenEmbeddingCache
from topocalib_uc.metrics.segmentation import accuracy, mean_iou
from topocalib_uc.prototypes.matching import ClassIndex
from topocalib_uc.train.partgraph_dataset import PartGraphCache, PartRecord
from topocalib_uc.train.splitting import (
    class_aware_subset,
    dataset_split,
    random_part_split,
    sample_labeled_parts,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train SSRL frozen embeddings with a residual MR-GCN head.")
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--ssrl-cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split-source", choices=["random_filtered", "dataset"], required=True)
    parser.add_argument("--train-ratio", type=float, required=True)
    parser.add_argument("--val-ratio", type=float, required=True)
    parser.add_argument("--test-ratio", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--min-parts-per-class", type=int, required=True)
    parser.add_argument("--labeled-part-budget", required=True)
    parser.add_argument("--hidden-dim", type=int, required=True)
    parser.add_argument("--mp-layers", type=int, required=True)
    parser.add_argument("--mlp-hidden-dim", type=int, required=True)
    parser.add_argument("--dropout", type=float, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--patience", type=int, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--weight-decay", type=float, required=True)
    parser.add_argument("--batch-part-count", type=int, required=True)
    parser.add_argument("--undirected", choices=["true", "false"], required=True)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    result = run(args)
    args.output.write_text(json.dumps(_json_safe(result), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(_json_safe(result), ensure_ascii=False, sort_keys=True))
    return 0


def run(args: argparse.Namespace) -> dict[str, Any]:
    args.undirected = _parse_bool(args.undirected)
    _log(f"start cache={args.cache_dir} embeddings={args.ssrl_cache_dir} output={args.output}")
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    cache = PartGraphCache(args.cache_dir)
    _log(f"loaded PartGraph parts={len(cache.records)} classes={cache.classes}")
    class_index = ClassIndex(cache.classes)
    split, selected_part_ids = _make_split(cache, args)
    labeled_budget = _resolve_budget(args.labeled_part_budget, split.train)
    labeled_parts = sample_labeled_parts(
        cache,
        train_part_ids=split.train,
        budget=labeled_budget,
        seed=args.seed,
        required_classes=cache.classes,
    )
    _log(f"split train={len(split.train)} val={len(split.val)} test={len(split.test)} labeled={len(labeled_parts)}")

    embedding_cache = FrozenEmbeddingCache(args.ssrl_cache_dir)
    z_by_part = embedding_cache.transform_cache(cache)
    embedding_dim = _single_embedding_dim(z_by_part)
    _log(f"loaded SSRL embeddings parts={len(z_by_part)} dim={embedding_dim}")
    model = SSRLMRGCN(
        embedding_dim,
        class_index.num_classes,
        hidden_dim=args.hidden_dim,
        mp_layers=args.mp_layers,
        mlp_hidden_dim=args.mlp_hidden_dim,
        dropout=args.dropout,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_state: dict[str, torch.Tensor] | None = None
    best_monitor = -1.0
    best_epoch = -1
    epochs_without_improvement = 0
    stopped_early = False
    history: list[dict[str, float]] = []

    for epoch in range(args.epochs):
        train_loss = _train_epoch(cache, z_by_part, model, optimizer, labeled_parts, class_index, args, epoch)
        train_eval = _evaluate(cache, z_by_part, model, labeled_parts, class_index, args)
        val_eval = _evaluate(cache, z_by_part, model, split.val, class_index, args)
        monitor = val_eval["miou"] if split.val else train_eval["miou"]
        history.append(
            {
                "epoch": float(epoch),
                "loss": float(train_loss),
                "train_labeled_miou": train_eval["miou"],
                "val_miou": val_eval["miou"],
            }
        )
        _log(
            f"epoch={epoch} loss={train_loss:.6f} "
            f"train_miou={train_eval['miou']:.6f} val_miou={val_eval['miou']:.6f}"
        )
        if monitor >= best_monitor:
            best_monitor = monitor
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if split.val and args.patience > 0 and epochs_without_improvement >= args.patience:
            stopped_early = True
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    checkpoint_path = _write_checkpoint(args, model, embedding_dim, cache.classes)
    val_metrics = _evaluate(cache, z_by_part, model, split.val, class_index, args)
    test_metrics = _evaluate(cache, z_by_part, model, split.test, class_index, args)
    _log(f"done checkpoint={checkpoint_path} test_miou={test_metrics['miou']:.6f} test_accuracy={test_metrics['accuracy']:.6f}")
    return {
        "method": "ssrl_mrgcn",
        "dataset": _dataset_name(cache),
        "cache": {
            "partgraph_cache_dir": str(args.cache_dir),
            "ssrl_cache_dir": str(args.ssrl_cache_dir),
        },
        "budget": args.labeled_part_budget,
        "seed": args.seed,
        "classes": cache.classes,
        "config": vars(args),
        "val": val_metrics,
        "test": test_metrics,
        "metrics": {
            "train_labeled": _evaluate(cache, z_by_part, model, labeled_parts, class_index, args),
            "val": val_metrics,
            "test": test_metrics,
        },
        "source": {
            "algorithm": "CVPR2023 SSRL downstream segmentation: precomputed face embeddings, face-face adjacency, 2-layer residual MR-GCN, two-hidden-layer MLP classifier.",
            "partgraph_cache_dir": str(args.cache_dir),
            "ssrl_embedding_cache_dir": str(args.ssrl_cache_dir),
            "split_source": args.split_source,
            "selected_part_ids": selected_part_ids,
            "split": split.as_dict(),
            "labeled_parts": labeled_parts,
            "embedding_encoder": sorted({record.encoder for record in embedding_cache.records}),
            "embedding_checkpoint": sorted({record.checkpoint for record in embedding_cache.records if record.checkpoint is not None}),
        },
        "checkpoint": str(checkpoint_path) if checkpoint_path is not None else None,
        "raw_result": {
            "history_tail": history[-10:],
            "training": {
                "epochs_ran": len(history),
                "best_epoch": best_epoch,
                "best_monitor_miou": best_monitor,
                "stopped_early": stopped_early,
                "patience": args.patience,
            },
            "embedding_dim": embedding_dim,
            "num_train_labeled_parts": len(labeled_parts),
            "num_model_parameters": sum(parameter.numel() for parameter in model.parameters()),
        },
    }


def _train_epoch(
    cache: PartGraphCache,
    z_by_part: dict[str, torch.Tensor],
    model: SSRLMRGCN,
    optimizer: torch.optim.Optimizer,
    labeled_parts: list[str],
    class_index: ClassIndex,
    args: argparse.Namespace,
    epoch: int,
) -> float:
    model.train()
    part_ids = labeled_parts[:]
    random.Random(args.seed + epoch).shuffle(part_ids)
    losses: list[float] = []
    for batch_ids in _part_batches(part_ids, args.batch_part_count):
        batch = _collate_parts(cache, z_by_part, batch_ids, class_index, undirected=args.undirected)
        optimizer.zero_grad()
        logits = model(batch["z"], batch["edges"])
        loss = F.cross_entropy(logits, batch["encoded_y"])
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))
    return float(sum(losses) / len(losses)) if losses else 0.0


@torch.no_grad()
def _evaluate(
    cache: PartGraphCache,
    z_by_part: dict[str, torch.Tensor],
    model: SSRLMRGCN,
    part_ids: list[str],
    class_index: ClassIndex,
    args: argparse.Namespace,
) -> dict[str, float]:
    if not part_ids:
        return {"accuracy": 0.0, "miou": 0.0, "num_faces": 0.0}
    model.eval()
    pred_chunks: list[torch.Tensor] = []
    target_chunks: list[torch.Tensor] = []
    for batch_ids in _part_batches(part_ids, args.batch_part_count):
        batch = _collate_parts(cache, z_by_part, batch_ids, class_index, undirected=args.undirected)
        logits = model(batch["z"], batch["edges"])
        pred_encoded = logits.argmax(dim=1)
        pred = torch.as_tensor([class_index.classes[int(idx)] for idx in pred_encoded.tolist()], dtype=torch.long)
        pred_chunks.append(pred)
        target_chunks.append(batch["y"])
    pred_all = torch.cat(pred_chunks, dim=0)
    target_all = torch.cat(target_chunks, dim=0)
    return {
        "accuracy": accuracy(pred_all, target_all),
        "miou": mean_iou(pred_all, target_all, class_index.classes),
        "num_faces": float(target_all.numel()),
    }


def _collate_parts(
    cache: PartGraphCache,
    z_by_part: dict[str, torch.Tensor],
    part_ids: list[str],
    class_index: ClassIndex,
    *,
    undirected: bool,
) -> dict[str, torch.Tensor]:
    records = [cache.by_part_id(part_id) for part_id in part_ids]
    edge_chunks: list[torch.Tensor] = []
    offset = 0
    for record in records:
        edges = _part_edges(record, undirected=undirected)
        if edges.numel():
            edge_chunks.append(edges + offset)
        offset += record.num_faces
    z = torch.cat([z_by_part[record.part_id] for record in records], dim=0)
    y = torch.cat([record.y for record in records], dim=0)
    edges = torch.cat(edge_chunks, dim=0) if edge_chunks else torch.zeros((0, 2), dtype=torch.long)
    return {"z": z, "y": y, "encoded_y": class_index.encode(y), "edges": edges}


def _part_edges(record: PartRecord, *, undirected: bool) -> torch.Tensor:
    edges = record.edges.long().reshape(-1, 2)
    if not undirected or edges.numel() == 0:
        return edges
    return torch.cat([edges, edges[:, [1, 0]]], dim=0)


def _make_split(cache: PartGraphCache, args: argparse.Namespace):
    selected_part_ids: list[str] | None = None
    if args.min_parts_per_class > 0:
        selected_part_ids = class_aware_subset(cache, min_parts_per_class=args.min_parts_per_class, seed=args.seed)
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
            seed=args.seed,
            require_train_all_classes=True,
        )
    return split, selected_part_ids


def _write_checkpoint(
    args: argparse.Namespace,
    model: SSRLMRGCN,
    embedding_dim: int,
    classes: list[int],
) -> Path | None:
    output = getattr(args, "output", None)
    if output is None:
        return None
    path = Path(output).with_suffix(".pt")
    torch.save(
        {
            "method": "ssrl_mrgcn",
            "model_state_dict": model.state_dict(),
            "embedding_dim": embedding_dim,
            "classes": classes,
            "config": vars(args),
        },
        path,
    )
    return path


def _single_embedding_dim(z_by_part: dict[str, torch.Tensor]) -> int:
    dims = {int(z.shape[1]) for z in z_by_part.values()}
    if len(dims) != 1:
        raise ValueError(f"All SSRL cached embeddings must have the same embedding dimension, got {sorted(dims)}")
    return dims.pop()


def _part_batches(part_ids: list[str], batch_part_count: int):
    if batch_part_count <= 0:
        raise ValueError("batch_part_count must be positive.")
    for start in range(0, len(part_ids), batch_part_count):
        yield part_ids[start : start + batch_part_count]


def _resolve_budget(value: int | str, train_part_ids: list[str]) -> int:
    if isinstance(value, str) and value.lower() == "all":
        return len(train_part_ids)
    return int(value)


def _parse_bool(value: bool | str) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() == "true"


def _dataset_name(cache: PartGraphCache) -> str:
    names = sorted({record.dataset for record in cache.records if record.dataset})
    if len(names) == 1:
        return names[0]
    return ",".join(names) if names else ""


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


def _log(message: str) -> None:
    print(f"[ssrl_mrgcn] {message}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
