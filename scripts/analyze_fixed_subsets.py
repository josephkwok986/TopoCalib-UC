#!/usr/bin/env python3
"""Analyze fixed hard/ambiguity subsets and B0-referenced margin buckets."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable


FaceKey = tuple[str, str, int]
RunKey = tuple[str, str, str, str, int]
BUCKETS = (
    ("[0.00,0.02)", 0.00, 0.02, False),
    ("[0.02,0.05)", 0.02, 0.05, False),
    ("[0.05,0.10)", 0.05, 0.10, False),
    ("[0.10,0.20)", 0.10, 0.20, False),
    ("[0.20,1.00]", 0.20, 1.00, True),
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", type=Path, action="append", required=True)
    parser.add_argument("--membership-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--beta", type=float, default=0.10)
    parser.add_argument("--reference-seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--std-ddof", type=int, choices=[0, 1], default=1)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    runs = [load_run(path) for path in args.input_jsonl]
    validate_aligned_runs(runs)
    b0_runs = select_b0_reference_runs(runs, args.reference_seeds)
    membership = build_membership(b0_runs, beta=args.beta)
    args.membership_output.parent.mkdir(parents=True, exist_ok=True)
    args.membership_output.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "dataset": b0_runs[0]["dataset"],
                "budget": b0_runs[0]["budget"],
                "reference_variant": "B0",
                "reference_seeds": sorted(args.reference_seeds),
                "beta": args.beta,
                "buckets": [name for name, _, _, _ in BUCKETS],
                "faces": [membership[key] for key in sorted(membership)],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    subset_runs, subset_aggregates = subset_statistics(runs, membership, std_ddof=args.std_ddof)
    bucket_runs, bucket_aggregates = bucket_gain_statistics(runs, membership, std_ddof=args.std_ddof)
    result = {
        "schema_version": 1,
        "membership_manifest": str(args.membership_output),
        "std_ddof": args.std_ddof,
        "subset_run_values": subset_runs,
        "subset_aggregates": subset_aggregates,
        "bucket_gain_run_values": bucket_runs,
        "bucket_gain_aggregates": bucket_aggregates,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "runs": len(runs),
                "faces": len(membership),
                "hard_faces": sum(bool(row["hard"]) for row in membership.values()),
                "ambiguity_faces": sum(bool(row["ambiguity"]) for row in membership.values()),
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )
    return 0


def load_run(path: Path) -> dict[str, Any]:
    rows = list(read_jsonl(path))
    if not rows:
        raise ValueError(f"{path}: no face records found")
    keys = {_run_key(row) for row in rows}
    if len(keys) != 1:
        raise ValueError(f"{path}: expected one run, found {sorted(keys)}")
    run_key = next(iter(keys))
    classes = {tuple(int(value) for value in row["classes"]) for row in rows}
    if len(classes) != 1:
        raise ValueError(f"{path}: class mapping changes within one run")
    faces: dict[FaceKey, dict[str, Any]] = {}
    for row in rows:
        key = _face_key(row)
        if key in faces:
            raise ValueError(f"{path}: duplicate face key {key}")
        neighbors = [int(value) for value in row.get("neighbor_face_ids", [])]
        if len(neighbors) != len(set(neighbors)):
            raise ValueError(f"{path}: duplicate neighbor IDs at face {key}")
        normalized = dict(row)
        normalized["neighbor_face_ids"] = sorted(neighbors)
        faces[key] = normalized
    return {
        "path": str(path),
        "dataset": run_key[0],
        "method": run_key[1],
        "variant": run_key[2],
        "budget": run_key[3],
        "seed": run_key[4],
        "classes": list(next(iter(classes))),
        "faces": faces,
    }


def validate_aligned_runs(runs: list[dict[str, Any]]) -> None:
    if not runs:
        raise ValueError("at least one run is required")
    seen: set[RunKey] = set()
    reference = runs[0]
    reference_faces = {
        key: (int(row["target"]), tuple(row["neighbor_face_ids"]))
        for key, row in reference["faces"].items()
    }
    for run in runs:
        key = (run["dataset"], run["method"], run["variant"], run["budget"], run["seed"])
        if key in seen:
            raise ValueError(f"duplicate run {key}")
        seen.add(key)
        if run["dataset"] != reference["dataset"] or run["budget"] != reference["budget"]:
            raise ValueError("all inputs must have the same dataset and budget")
        if run["classes"] != reference["classes"]:
            raise ValueError(f"{run['path']}: class mapping differs from the other runs")
        current_faces = {
            face_key: (int(row["target"]), tuple(row["neighbor_face_ids"]))
            for face_key, row in run["faces"].items()
        }
        if current_faces != reference_faces:
            raise ValueError(f"{run['path']}: face keys, targets, or topology differ from the fixed test records")
    by_part: dict[tuple[str, str], set[int]] = {}
    for dataset, part_id, face_id in reference["faces"]:
        by_part.setdefault((dataset, part_id), set()).add(face_id)
    for key, row in reference["faces"].items():
        known = by_part[(key[0], key[1])]
        unknown = set(row["neighbor_face_ids"]) - known
        if unknown:
            raise ValueError(f"face {key} refers to missing neighbor face IDs {sorted(unknown)}")


def select_b0_reference_runs(runs: list[dict[str, Any]], seeds: list[int]) -> list[dict[str, Any]]:
    requested = sorted(set(int(seed) for seed in seeds))
    selected = [run for run in runs if run["variant"] == "B0" and int(run["seed"]) in requested]
    by_seed: dict[int, list[dict[str, Any]]] = {}
    for run in selected:
        by_seed.setdefault(int(run["seed"]), []).append(run)
    missing = [seed for seed in requested if seed not in by_seed]
    duplicated = [seed for seed, items in by_seed.items() if len(items) != 1]
    if missing or duplicated:
        raise ValueError(f"B0 reference runs must contain exactly one run per seed; missing={missing}, duplicated={duplicated}")
    return [by_seed[seed][0] for seed in requested]


def build_membership(b0_runs: list[dict[str, Any]], *, beta: float) -> dict[FaceKey, dict[str, Any]]:
    if not 0 < beta <= 1:
        raise ValueError("beta must be in (0, 1]")
    reference_faces = b0_runs[0]["faces"]
    membership: dict[FaceKey, dict[str, Any]] = {}
    for key in sorted(reference_faces):
        margins = [
            float(
                run["faces"][key]["probability_margin_before"]
                if "probability_margin_before" in run["faces"][key]
                else run["faces"][key]["margin_before"]
            )
            for run in b0_runs
        ]
        if any(not math.isfinite(value) or value < -1e-7 or value > 1.0 + 1e-7 for value in margins):
            raise ValueError(f"face {key} has invalid B0 probability margins {margins}")
        mean_margin = sum(margins) / len(margins)
        bucket = margin_bucket(mean_margin)
        row = reference_faces[key]
        target = int(row["target"])
        neighbor_targets = {
            int(reference_faces[(key[0], key[1], neighbor)]["target"])
            for neighbor in row["neighbor_face_ids"]
            if int(reference_faces[(key[0], key[1], neighbor)]["target"]) != target
        }
        membership[key] = {
            "dataset": key[0],
            "part_id": key[1],
            "face_id": key[2],
            "face_key": list(key),
            "target": target,
            "hard": len(neighbor_targets) >= 2,
            "distinct_different_neighbor_classes": sorted(neighbor_targets),
            "b0_margin_values": margins,
            "mean_b0_probability_margin": mean_margin,
            "ambiguity": mean_margin < beta,
            "bucket": bucket,
        }
    first_three = {name for name, _, upper, _ in BUCKETS if upper <= beta + 1e-12}
    if beta == 0.10:
        for row in membership.values():
            if bool(row["ambiguity"]) != (row["bucket"] in first_three):
                raise AssertionError("ambiguity membership must equal the first three fixed buckets")
    return membership


def margin_bucket(value: float) -> str:
    if not math.isfinite(value) or value < -1e-7 or value > 1.0 + 1e-7:
        raise ValueError(f"margin must be in [0,1], got {value}")
    value = min(max(value, 0.0), 1.0)
    for name, lower, upper, upper_inclusive in BUCKETS:
        if value >= lower and (value <= upper if upper_inclusive else value < upper):
            return name
    raise AssertionError(f"no bucket for margin={value}")


def subset_statistics(
    runs: list[dict[str, Any]],
    membership: dict[FaceKey, dict[str, Any]],
    *,
    std_ddof: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    raw = []
    for run in runs:
        for subset in ("hard", "ambiguity"):
            selected = [key for key, row in membership.items() if bool(row[subset])]
            rate = correct_rate(run, selected)
            raw.append(
                {
                    "dataset": run["dataset"],
                    "method": run["method"],
                    "variant": run["variant"],
                    "budget": run["budget"],
                    "seed": run["seed"],
                    "subset": subset,
                    "num_faces": len(selected),
                    "correct_rate": rate,
                    "correct_rate_percent": 100.0 * rate if rate is not None else None,
                }
            )
    grouped: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = {}
    for row in raw:
        key = (row["dataset"], row["method"], row["variant"], row["budget"], row["subset"])
        grouped.setdefault(key, []).append(row)
    aggregates = []
    for key, rows in sorted(grouped.items()):
        rows.sort(key=lambda row: int(row["seed"]))
        values = [float(row["correct_rate"]) for row in rows if row["correct_rate"] is not None]
        aggregates.append(
            {
                "dataset": key[0],
                "method": key[1],
                "variant": key[2],
                "budget": key[3],
                "subset": key[4],
                "num_faces": rows[0]["num_faces"],
                "seeds": [row["seed"] for row in rows],
                "correct_rate_values": [row["correct_rate"] for row in rows],
                "correct_rate_mean": sum(values) / len(values) if values else None,
                "correct_rate_std": standard_deviation(values, ddof=std_ddof) if values else None,
                "std_ddof": std_ddof,
            }
        )
    return raw, aggregates


def bucket_gain_statistics(
    runs: list[dict[str, Any]],
    membership: dict[FaceKey, dict[str, Any]],
    *,
    std_ddof: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    b0_by_seed: dict[int, dict[str, Any]] = {}
    for run in runs:
        if run["variant"] == "B0":
            seed = int(run["seed"])
            if seed in b0_by_seed:
                raise ValueError(f"multiple B0 runs found for seed={seed}")
            b0_by_seed[seed] = run
    raw = []
    for run in runs:
        seed = int(run["seed"])
        if seed not in b0_by_seed:
            raise ValueError(f"run {run['variant']} seed={seed} has no same-seed B0 reference")
        baseline = b0_by_seed[seed]
        for bucket, _, _, _ in BUCKETS:
            selected = [key for key, row in membership.items() if row["bucket"] == bucket]
            rate = correct_rate(run, selected)
            baseline_rate = correct_rate(baseline, selected)
            gain = rate - baseline_rate if rate is not None and baseline_rate is not None else None
            raw.append(
                {
                    "dataset": run["dataset"],
                    "method": run["method"],
                    "variant": run["variant"],
                    "budget": run["budget"],
                    "seed": seed,
                    "bucket": bucket,
                    "num_faces": len(selected),
                    "correct_rate": rate,
                    "b0_correct_rate": baseline_rate,
                    "gain": gain,
                    "gain_percentage_points": 100.0 * gain if gain is not None else None,
                }
            )
    grouped: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = {}
    for row in raw:
        key = (row["dataset"], row["method"], row["variant"], row["budget"], row["bucket"])
        grouped.setdefault(key, []).append(row)
    aggregates = []
    for key, rows in sorted(grouped.items()):
        rows.sort(key=lambda row: int(row["seed"]))
        values = [float(row["gain"]) for row in rows if row["gain"] is not None]
        aggregates.append(
            {
                "dataset": key[0],
                "method": key[1],
                "variant": key[2],
                "budget": key[3],
                "bucket": key[4],
                "num_faces": rows[0]["num_faces"],
                "seeds": [row["seed"] for row in rows],
                "gain_values": [row["gain"] for row in rows],
                "gain_mean": sum(values) / len(values) if values else None,
                "gain_std": standard_deviation(values, ddof=std_ddof) if values else None,
                "std_ddof": std_ddof,
            }
        )
    return raw, aggregates


def correct_rate(run: dict[str, Any], face_keys: list[FaceKey]) -> float | None:
    if not face_keys:
        return None
    return sum(int(run["faces"][key]["pred_after"]) == int(run["faces"][key]["target"]) for key in face_keys) / len(face_keys)


def standard_deviation(values: list[float], *, ddof: int) -> float:
    if len(values) <= ddof:
        return 0.0
    average = sum(values) / len(values)
    return math.sqrt(sum((value - average) ** 2 for value in values) / (len(values) - ddof))


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
