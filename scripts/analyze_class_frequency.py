#!/usr/bin/env python3
"""Build fixed class-frequency groups and compute group-wise test mIoU."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
from typing import Any, Iterable

import torch

from topocalib_uc.metrics.segmentation import grouped_mean_iou, mean_iou, per_class_iou
from topocalib_uc.train.partgraph_dataset import PartGraphCache
from topocalib_uc.train.splitting import PartSplit, load_split_manifest


GROUP_NAMES = ("Head", "Body", "Tail")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--input-jsonl", type=Path, action="append", required=True)
    parser.add_argument("--group-sizes", type=int, nargs=3, required=True, metavar=("HEAD", "BODY", "TAIL"))
    parser.add_argument("--class-names-json", type=Path, default=None)
    parser.add_argument("--group-manifest-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--std-ddof", type=int, choices=[0, 1], default=1)
    parser.add_argument("--expected-seeds", type=int, nargs="+", default=None)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    cache = PartGraphCache(args.cache_dir)
    split = load_split_manifest(args.split_manifest, cache)
    class_names = load_class_names(args.class_names_json)
    manifest = build_frequency_manifest(cache, split, args.group_sizes, class_names)
    args.group_manifest_output.parent.mkdir(parents=True, exist_ok=True)
    args.group_manifest_output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    expected_faces = expected_test_faces(cache, split)
    runs = [analyze_run(path, manifest, expected_faces) for path in args.input_jsonl]
    validate_runs(runs)
    aggregates = aggregate_runs(runs, std_ddof=args.std_ddof, expected_seeds=args.expected_seeds)
    result = {
        "schema_version": 1,
        "split_manifest": str(args.split_manifest),
        "group_manifest": str(args.group_manifest_output),
        "std_ddof": args.std_ddof,
        "runs": [{key: value for key, value in run.items() if not key.startswith("_")} for run in runs],
        "aggregates": aggregates,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"runs": len(runs), "groups": len(aggregates), "output": str(args.output)}, sort_keys=True))
    return 0


def build_frequency_manifest(
    cache: PartGraphCache,
    split: PartSplit,
    group_sizes: list[int] | tuple[int, int, int],
    class_names: dict[int, str] | None = None,
) -> dict[str, Any]:
    """Rank classes using the full fixed training split before label sampling."""

    sizes = [int(value) for value in group_sizes]
    if any(value <= 0 for value in sizes) or len(sizes) != 3:
        raise ValueError("group_sizes must contain three positive integers")
    if sum(sizes) != len(cache.classes):
        raise ValueError(f"group sizes sum to {sum(sizes)}, but cache has {len(cache.classes)} classes")
    counts: Counter[int] = Counter()
    for part_id in split.train:
        counts.update(int(value) for value in cache.by_part_id(part_id).y.tolist())
    if set(counts) != set(cache.classes):
        raise ValueError(f"training split does not cover classes {sorted(set(cache.classes) - set(counts))}")
    total = sum(counts.values())
    ranked = sorted(cache.classes, key=lambda class_id: (-counts[class_id], class_id))
    names = class_names or {}
    class_rows: list[dict[str, Any]] = []
    groups: dict[str, list[int]] = {}
    offset = 0
    for group_name, size in zip(GROUP_NAMES, sizes):
        group_classes = ranked[offset : offset + size]
        groups[group_name] = group_classes
        for index, class_id in enumerate(group_classes, start=offset + 1):
            class_rows.append(
                {
                    "class_id": class_id,
                    "class_name": names.get(class_id, f"class_{class_id}"),
                    "rank": index,
                    "training_face_count": counts[class_id],
                    "training_face_share": counts[class_id] / total,
                    "group": group_name,
                }
            )
        offset += size
    if len({row["class_id"] for row in class_rows}) != len(cache.classes):
        raise AssertionError("frequency groups do not form a class partition")
    group_rows = []
    by_class = {int(row["class_id"]): row for row in class_rows}
    start_rank = 1
    for group_name, size in zip(GROUP_NAMES, sizes):
        class_ids = groups[group_name]
        group_rows.append(
            {
                "group": group_name,
                "rank_start": start_rank,
                "rank_end": start_rank + size - 1,
                "training_face_count": sum(counts[class_id] for class_id in class_ids),
                "training_face_share": sum(float(by_class[class_id]["training_face_share"]) for class_id in class_ids),
                "class_ids": class_ids,
                "class_names": [str(by_class[class_id]["class_name"]) for class_id in class_ids],
            }
        )
        start_rank += size
    datasets = {record.dataset for record in cache.records}
    if len(datasets) != 1:
        raise ValueError(f"cache must contain one dataset, got {sorted(datasets)}")
    return {
        "schema_version": 1,
        "dataset": datasets.pop(),
        "frequency_source": "fixed training split before labeled-part sampling",
        "total_training_faces": total,
        "group_sizes": dict(zip(GROUP_NAMES, sizes)),
        "groups": groups,
        "classes": sorted(class_rows, key=lambda row: int(row["rank"])),
        "group_summary": group_rows,
    }


def expected_test_faces(cache: PartGraphCache, split: PartSplit) -> dict[tuple[str, str, int], int]:
    output: dict[tuple[str, str, int], int] = {}
    for part_id in split.test:
        record = cache.by_part_id(part_id)
        for face_id, target in enumerate(record.y.tolist()):
            output[(record.dataset, part_id, face_id)] = int(target)
    return output


def analyze_run(
    path: Path,
    manifest: dict[str, Any],
    expected_faces: dict[tuple[str, str, int], int],
) -> dict[str, Any]:
    rows = list(read_jsonl(path))
    if not rows:
        raise ValueError(f"{path}: no face records found")
    run_keys = {
        (str(row["dataset"]), str(row["method"]), str(row["variant"]), str(row["budget"]), int(row["seed"]))
        for row in rows
    }
    if len(run_keys) != 1:
        raise ValueError(f"{path}: expected one run, found {sorted(run_keys)}")
    run_key = next(iter(run_keys))
    classes = [int(row["class_id"]) for row in manifest["classes"]]
    if run_key[0] != manifest["dataset"]:
        raise ValueError(f"{path}: dataset does not match frequency manifest")
    by_face: dict[tuple[str, str, int], dict[str, Any]] = {}
    for row in rows:
        if [int(value) for value in row["classes"]] != sorted(classes):
            raise ValueError(f"{path}: class mapping does not match frequency manifest")
        key = (str(row["dataset"]), str(row["part_id"]), int(row["face_id"]))
        if key in by_face:
            raise ValueError(f"{path}: duplicate face key {key}")
        by_face[key] = row
    observed = {key: int(row["target"]) for key, row in by_face.items()}
    if observed != expected_faces:
        raise ValueError(f"{path}: face keys or targets do not match the fixed test split")
    ordered = [by_face[key] for key in sorted(by_face)]
    pred = torch.as_tensor([int(row["pred_after"]) for row in ordered], dtype=torch.long)
    target = torch.as_tensor([int(row["target"]) for row in ordered], dtype=torch.long)
    base_class_rows = per_class_iou(pred, target, sorted(classes))
    names = {int(row["class_id"]): str(row["class_name"]) for row in manifest["classes"]}
    group_by_class = {int(row["class_id"]): str(row["group"]) for row in manifest["classes"]}
    group_miou = grouped_mean_iou(base_class_rows, {name: manifest["groups"][name] for name in GROUP_NAMES})
    class_rows = [
        {
            **row,
            "class_name": names[row["class_id"]],
            "group": group_by_class[row["class_id"]],
        }
        for row in base_class_rows
    ]
    overall = mean_iou(pred, target, sorted(classes))
    return {
        "dataset": run_key[0],
        "method": run_key[1],
        "variant": run_key[2],
        "budget": run_key[3],
        "seed": run_key[4],
        "num_faces": len(rows),
        "per_class": class_rows,
        "group_miou": group_miou,
        "overall_miou": overall,
        "_face_targets": observed,
    }


def validate_runs(runs: list[dict[str, Any]]) -> None:
    seen: set[tuple[str, str, str, str, int]] = set()
    references: dict[tuple[str, str], dict[tuple[str, str, int], int]] = {}
    for run in runs:
        key = (run["dataset"], run["method"], run["variant"], run["budget"], run["seed"])
        if key in seen:
            raise ValueError(f"duplicate run {key}")
        seen.add(key)
        comparison = (str(run["dataset"]), str(run["budget"]))
        if comparison not in references:
            references[comparison] = run["_face_targets"]
        elif references[comparison] != run["_face_targets"]:
            raise ValueError(f"fixed test split mismatch for {comparison}")


def aggregate_runs(
    runs: list[dict[str, Any]],
    *,
    std_ddof: int,
    expected_seeds: list[int] | None,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for run in runs:
        key = (run["dataset"], run["method"], run["variant"], run["budget"])
        grouped.setdefault(key, []).append(run)
    output = []
    for key, items in sorted(grouped.items()):
        items.sort(key=lambda row: int(row["seed"]))
        seeds = [int(row["seed"]) for row in items]
        if len(seeds) != len(set(seeds)):
            raise ValueError(f"duplicate seeds for group {key}")
        if expected_seeds is not None and seeds != sorted(expected_seeds):
            raise ValueError(f"group {key} has seeds={seeds}, expected={sorted(expected_seeds)}")
        row: dict[str, Any] = {
            "dataset": key[0],
            "method": key[1],
            "variant": key[2],
            "budget": key[3],
            "seeds": seeds,
            "num_runs": len(items),
            "std_ddof": std_ddof,
        }
        metric_sources = {name: [float(item["group_miou"][name]) for item in items] for name in GROUP_NAMES}
        metric_sources["Overall"] = [float(item["overall_miou"]) for item in items]
        for name, values in metric_sources.items():
            prefix = name.lower()
            row[f"{prefix}_values"] = values
            row[f"{prefix}_mean"] = sum(values) / len(values)
            row[f"{prefix}_std"] = standard_deviation(values, ddof=std_ddof)
        output.append(row)
    return output


def load_class_names(path: Path | None) -> dict[int, str]:
    if path is None:
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, list):
        return {index: str(value) for index, value in enumerate(raw)}
    if isinstance(raw, dict):
        return {int(key): str(value) for key, value in raw.items()}
    raise ValueError("class names JSON must be a list or object")


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: record must be a JSON object")
            yield row


def standard_deviation(values: list[float], *, ddof: int) -> float:
    if len(values) <= ddof:
        return 0.0
    average = sum(values) / len(values)
    return math.sqrt(sum((value - average) ** 2 for value in values) / (len(values) - ddof))


if __name__ == "__main__":
    raise SystemExit(main())
