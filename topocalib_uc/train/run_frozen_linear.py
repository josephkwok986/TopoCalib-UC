#!/usr/bin/env python3
"""Run Frozen + Token Adapter + Linear Head."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from frozen_representation.cache import FrozenEmbeddingCache
from topocalib_uc.metrics.segmentation import accuracy, mean_iou
from topocalib_uc.prototypes.matching import ClassIndex
from topocalib_uc.tokenization.adapter import SurfaceTokenAdapter
from topocalib_uc.tokenization.embedding_normalization import (
    apply_embedding_normalizer,
    fit_embedding_normalizer,
    normalizer_json_info,
)
from topocalib_uc.train.partgraph_dataset import PartGraphCache
from topocalib_uc.train.splitting import class_aware_subset, dataset_split, random_part_split, sample_labeled_parts


class FrozenLinearModel(nn.Module):
    def __init__(self, embedding_dim: int, hidden_dim: int, num_surface_types: int, num_classes: int):
        super().__init__()
        self.adapter = SurfaceTokenAdapter(embedding_dim, hidden_dim, num_surface_types)
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, z: torch.Tensor, surface_type: torch.Tensor) -> torch.Tensor:
        tokens = self.adapter(z, surface_type)
        return self.head(tokens)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a frozen linear baseline on PartGraph cache.")
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split-source", choices=["random_filtered", "dataset"], default="random_filtered")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-parts-per-class", type=int, default=0)
    parser.add_argument("--labeled-part-budget", required=True)
    parser.add_argument("--ssrl-cache-dir", type=Path, default=None)
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-surface-types", type=int, default=5)
    parser.add_argument("--checkpoint-output", type=Path, default=None)
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
    _log(f"start frozen_linear cache={args.cache_dir} embeddings={args.ssrl_cache_dir} output={args.output}")
    torch.manual_seed(args.seed)
    cache = PartGraphCache(args.cache_dir)
    _log(f"loaded PartGraph parts={len(cache.records)} classes={cache.classes}")
    class_index = ClassIndex(cache.classes)
    split, selected_part_ids = _make_split(cache, args)
    labeled_parts = sample_labeled_parts(
        cache,
        train_part_ids=split.train,
        budget=_resolve_budget(args.labeled_part_budget, split.train),
        seed=args.seed,
        required_classes=cache.classes,
    )
    _log(f"split train={len(split.train)} val={len(split.val)} test={len(split.test)} labeled={len(labeled_parts)}")

    z_by_part, embedding_dim, embedding_normalizer = _build_frozen_embeddings(cache, split.train, args)
    _log(
        f"loaded SSRL embeddings parts={len(z_by_part)} dim={embedding_dim} "
        f"normalization={embedding_normalizer['type']}"
    )
    model = FrozenLinearModel(embedding_dim, args.hidden_dim, args.num_surface_types, class_index.num_classes)
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
        logits = model(labeled_batch["z"], labeled_batch["surface_type"])
        loss = F.cross_entropy(logits, labeled_batch["encoded_y"])
        loss.backward()
        optimizer.step()

        train_eval = _evaluate(cache, z_by_part, model, labeled_parts, class_index)
        val_eval = _evaluate(cache, z_by_part, model, split.val, class_index)
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

    checkpoint_path = _write_checkpoint(args, model, embedding_dim, class_index.classes, embedding_normalizer)
    metrics = {
        "train_labeled": _evaluate(cache, z_by_part, model, labeled_parts, class_index),
        "val": _evaluate(cache, z_by_part, model, split.val, class_index),
        "test": _evaluate(cache, z_by_part, model, split.test, class_index),
    }
    _log(f"done test_miou={metrics['test']['miou']:.6f} test_accuracy={metrics['test']['accuracy']:.6f}")
    return {
        "method": "frozen_linear",
        "cache_dir": str(args.cache_dir),
        "classes": cache.classes,
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


def _collect_faces(
    cache: PartGraphCache,
    z_by_part: dict[str, torch.Tensor],
    part_ids: list[str],
    class_index: ClassIndex,
) -> dict[str, torch.Tensor]:
    records = [cache.by_part_id(part_id) for part_id in part_ids]
    z = torch.cat([z_by_part[record.part_id] for record in records], dim=0)
    surface_type = torch.cat([record.surface_type for record in records], dim=0)
    y = torch.cat([record.y for record in records], dim=0)
    return {"z": z, "surface_type": surface_type, "y": y, "encoded_y": class_index.encode(y)}


@torch.no_grad()
def _evaluate(
    cache: PartGraphCache,
    z_by_part: dict[str, torch.Tensor],
    model: FrozenLinearModel,
    eval_part_ids: list[str],
    class_index: ClassIndex,
) -> dict[str, float]:
    if not eval_part_ids:
        return {"accuracy": 0.0, "miou": 0.0, "num_faces": 0.0}
    model.eval()
    batch = _collect_faces(cache, z_by_part, eval_part_ids, class_index)
    logits = model(batch["z"], batch["surface_type"])
    pred_encoded = logits.argmax(dim=1)
    pred = torch.as_tensor([class_index.classes[int(idx)] for idx in pred_encoded.tolist()], dtype=torch.long)
    return {
        "accuracy": accuracy(pred, batch["y"]),
        "miou": mean_iou(pred, batch["y"], class_index.classes),
        "num_faces": float(batch["y"].numel()),
    }


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


def _write_checkpoint(
    args: argparse.Namespace,
    model: FrozenLinearModel,
    embedding_dim: int,
    classes: list[int],
    embedding_normalizer: dict[str, Any],
) -> Path | None:
    output = getattr(args, "checkpoint_output", None)
    if output is None and getattr(args, "output", None) is not None:
        output = Path(args.output).with_suffix(".pt")
    if output is None:
        return None
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "checkpoint_type": "topocalib_uc_downstream",
            "method": "frozen_linear",
            "model_state_dict": model.state_dict(),
            "classes": list(classes),
            "embedding_dim": int(embedding_dim),
            "embedding_normalization": embedding_normalizer,
            "hidden_dim": int(args.hidden_dim),
            "num_surface_types": int(args.num_surface_types),
            "config": _json_safe(vars(args)),
        },
        output,
    )
    return output


def _single_embedding_dim(z_by_part: dict[str, torch.Tensor]) -> int:
    dims = {int(z.shape[1]) for z in z_by_part.values()}
    if len(dims) != 1:
        raise ValueError(f"All frozen embeddings must have the same embedding dimension, got {sorted(dims)}")
    return dims.pop()


def _ensure_defaults(args: argparse.Namespace) -> None:
    for name, default in [
        ("split_source", "random_filtered"),
        ("train_ratio", 0.7),
        ("val_ratio", 0.15),
        ("test_ratio", 0.15),
        ("seed", 0),
        ("min_parts_per_class", 0),
        ("ssrl_cache_dir", None),
        ("embedding_dim", 256),
        ("hidden_dim", 128),
        ("epochs", 200),
        ("patience", 20),
        ("lr", 1e-3),
        ("weight_decay", 1e-4),
        ("num_surface_types", 5),
        ("checkpoint_output", None),
    ]:
        if not hasattr(args, name):
            setattr(args, name, default)


def _log(message: str) -> None:
    print(f"[run_frozen_linear] {message}", flush=True)


def _resolve_budget(value: int | str, train_part_ids: list[str]) -> int:
    if isinstance(value, str) and value.lower() == "all":
        return len(train_part_ids)
    return int(value)


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
