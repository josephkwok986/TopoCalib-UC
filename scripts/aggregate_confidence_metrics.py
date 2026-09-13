#!/usr/bin/env python3
"""Aggregate face-level mIoU, ECE, and NLL from inference JSONL records."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable

import torch

from topocalib_uc.metrics.calibration import calibration_metrics_from_logits
from topocalib_uc.metrics.segmentation import mean_iou


RunKey = tuple[str, str, str, str, int]
FaceKey = tuple[str, str, int]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--csv-output", type=Path, default=None)
    parser.add_argument("--num-bins", type=int, default=15)
    parser.add_argument("--std-ddof", type=int, choices=[0, 1], default=1)
    parser.add_argument("--expected-seeds", type=int, nargs="+", default=None)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    runs = [read_run(path, num_bins=args.num_bins) for path in args.input_jsonl]
    validate_fixed_faces(runs)
    aggregated = aggregate_runs(runs, std_ddof=args.std_ddof, expected_seeds=args.expected_seeds)
    result = {
        "schema_version": 1,
        "metric_definition": {
            "ece_bins": args.num_bins,
            "ece_range": "[0,1] equal-width; internal boundaries enter the upper bin",
            "ece_report_unit": "percent",
            "nll_source": "final logits via log_softmax",
            "std_ddof": args.std_ddof,
        },
        "inputs": [str(path) for path in args.input_jsonl],
        "runs": [{key: value for key, value in run.items() if not key.startswith("_")} for run in runs],
        "aggregates": aggregated,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.csv_output is not None:
        write_csv(args.csv_output, aggregated)
    print(json.dumps({"runs": len(runs), "groups": len(aggregated), "output": str(args.output)}, sort_keys=True))
    return 0


def read_run(path: Path, *, num_bins: int = 15) -> dict[str, Any]:
    rows = list(_read_jsonl(path))
    if not rows:
        raise ValueError(f"{path}: no face records found")
    run_keys = {_run_key(row) for row in rows}
    if len(run_keys) != 1:
        raise ValueError(f"{path}: expected exactly one run, found {sorted(run_keys)}")
    run_key = next(iter(run_keys))
    classes_set = {tuple(int(item) for item in row["classes"]) for row in rows}
    if len(classes_set) != 1:
        raise ValueError(f"{path}: class mapping changes within one run")
    classes = list(next(iter(classes_set)))
    face_rows: dict[FaceKey, dict[str, Any]] = {}
    for row in rows:
        key = _face_key(row)
        if key in face_rows:
            raise ValueError(f"{path}: duplicate face key {key}")
        face_rows[key] = row
    logits = torch.as_tensor([row["calibrated_logits"] for row in rows], dtype=torch.float64)
    targets = torch.as_tensor([int(row["target"]) for row in rows], dtype=torch.long)
    calibration = calibration_metrics_from_logits(logits, targets, classes=classes, num_bins=num_bins)
    prediction = torch.as_tensor([int(row["pred_after"]) for row in rows], dtype=torch.long)
    logits_prediction = torch.as_tensor(
        [classes[int(index)] for index in logits.argmax(dim=1).tolist()],
        dtype=torch.long,
    )
    if not torch.equal(prediction, logits_prediction):
        raise ValueError(f"{path}: pred_after is inconsistent with calibrated_logits")
    return {
        "dataset": run_key[0],
        "method": run_key[1],
        "variant": run_key[2],
        "budget": run_key[3],
        "seed": run_key[4],
        "classes": classes,
        "num_faces": len(rows),
        "miou": mean_iou(prediction, targets, classes),
        "ece": calibration["ece"],
        "ece_percent": calibration["ece_percent"],
        "nll": calibration["nll"],
        "_face_keys": [list(key) for key in sorted(face_rows)],
        "_targets_by_face": {json.dumps(key): int(face_rows[key]["target"]) for key in sorted(face_rows)},
    }


def validate_fixed_faces(runs: list[dict[str, Any]]) -> None:
    """Require exact face IDs, targets, and class mappings within each dataset/budget."""

    reference: dict[tuple[str, str], dict[str, Any]] = {}
    seen_runs: set[RunKey] = set()
    for run in runs:
        run_key = (
            str(run["dataset"]),
            str(run["method"]),
            str(run["variant"]),
            str(run["budget"]),
            int(run["seed"]),
        )
        if run_key in seen_runs:
            raise ValueError(f"duplicate run {run_key}")
        seen_runs.add(run_key)
        comparison_key = (str(run["dataset"]), str(run["budget"]))
        current = {
            "classes": run["classes"],
            "face_keys": run["_face_keys"],
            "targets_by_face": run["_targets_by_face"],
        }
        if comparison_key not in reference:
            reference[comparison_key] = current
        elif current != reference[comparison_key]:
            raise ValueError(
                f"fixed test split validation failed for dataset={comparison_key[0]!r}, budget={comparison_key[1]!r}"
            )


def aggregate_runs(
    runs: list[dict[str, Any]],
    *,
    std_ddof: int,
    expected_seeds: list[int] | None = None,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for run in runs:
        key = (str(run["dataset"]), str(run["method"]), str(run["variant"]), str(run["budget"]))
        grouped.setdefault(key, []).append(run)
    output: list[dict[str, Any]] = []
    for key, items in sorted(grouped.items()):
        items.sort(key=lambda row: int(row["seed"]))
        seeds = [int(row["seed"]) for row in items]
        if len(seeds) != len(set(seeds)):
            raise ValueError(f"duplicate seeds for aggregate group {key}")
        if expected_seeds is not None and seeds != sorted(expected_seeds):
            raise ValueError(f"aggregate group {key} has seeds={seeds}, expected={sorted(expected_seeds)}")
        result: dict[str, Any] = {
            "dataset": key[0],
            "method": key[1],
            "variant": key[2],
            "budget": key[3],
            "num_runs": len(items),
            "seeds": seeds,
            "num_faces": int(items[0]["num_faces"]),
            "std_ddof": std_ddof,
        }
        for metric in ("miou", "ece", "ece_percent", "nll"):
            values = [float(row[metric]) for row in items]
            result[f"{metric}_values"] = values
            result[f"{metric}_mean"] = sum(values) / len(values)
            result[f"{metric}_std"] = _standard_deviation(values, ddof=std_ddof)
        output.append(result)
    return output


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {key: json.dumps(value) if isinstance(value, list) else value for key, value in row.items()}
            )


def _standard_deviation(values: list[float], *, ddof: int) -> float:
    if len(values) <= ddof:
        return 0.0
    average = sum(values) / len(values)
    return math.sqrt(sum((value - average) ** 2 for value in values) / (len(values) - ddof))


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
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


def _run_key(row: dict[str, Any]) -> RunKey:
    return (
        str(row["dataset"]),
        str(row["method"]),
        str(row["variant"]),
        str(row["budget"]),
        int(row["seed"]),
    )


def _face_key(row: dict[str, Any]) -> FaceKey:
    return str(row["dataset"]), str(row["part_id"]), int(row["face_id"])


if __name__ == "__main__":
    raise SystemExit(main())
