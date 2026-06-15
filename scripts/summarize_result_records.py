#!/usr/bin/env python3
"""Summarize raw mechanism records from full experiment runs.

This utility is intentionally lightweight. It does not run the full CAD
pipeline; it reads per-face inference and calibration JSONL records, then
writes intermediate CSV/JSON summaries for manual paper table preparation.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Iterable


DEFAULT_OUTPUT_ROOT = Path("/workspace/Gjj Local/data2/topocalib_uc_out/result_record_summary")
MARGIN_BINS = [
    ("very_small", 0.00, 0.05),
    ("small", 0.05, 0.10),
    ("medium", 0.10, 0.20),
    ("large", 0.20, 1.01),
]


@dataclass(frozen=True)
class Record:
    dataset: str
    method: str
    variant: str
    budget: str
    seed: int
    part_id: str
    face_id: int
    target: str
    pred_before: str
    pred_after: str
    candidate_a: str
    candidate_b: str
    margin_before: float
    margin_after: float
    gate: float
    local_evidence: float
    part_prior: float
    calibration_delta: float
    num_neighbors: int
    relation_types: list[str]

    @property
    def run_key(self) -> tuple[str, str, str, str, int]:
        return (self.dataset, self.method, self.variant, self.budget, self.seed)

    @property
    def pair_key(self) -> tuple[str, str]:
        return tuple(sorted((self.candidate_a, self.candidate_b)))

    @property
    def before_correct(self) -> bool:
        return self.pred_before == self.target

    @property
    def after_correct(self) -> bool:
        return self.pred_after == self.target

    @property
    def corrected_by_calibration(self) -> bool:
        return not self.before_correct and self.after_correct

    @property
    def damaged_by_calibration(self) -> bool:
        return self.before_correct and not self.after_correct


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--input-jsonl", type=Path, action="append", required=True)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    input_paths = list(args.input_jsonl)
    records = list(read_records(input_paths))
    if not records:
        raise SystemExit("No records found in --input-jsonl files.")

    summary_dir = output_root / "summaries"
    summary_dir.mkdir(parents=True, exist_ok=True)
    summaries = summarize_records(records)
    write_outputs(summary_dir, records, summaries, input_paths)

    print(
        json.dumps(
            {
                "records": len(records),
                "inputs": [str(path) for path in input_paths],
                "summary_dir": str(summary_dir),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


def read_records(paths: Iterable[Path]) -> Iterable[Record]:
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield normalize_record(json.loads(line))
                except Exception as exc:  # noqa: BLE001
                    raise ValueError(f"Invalid record in {path}:{line_no}: {exc}") from exc


def normalize_record(raw: dict[str, Any]) -> Record:
    return Record(
        dataset=str(raw["dataset"]),
        method=str(raw["method"]),
        variant=str(raw.get("variant", "")),
        budget=str(raw.get("budget", "")),
        seed=int(raw.get("seed", 0)),
        part_id=str(raw["part_id"]),
        face_id=int(raw["face_id"]),
        target=str(raw["target"]),
        pred_before=str(raw.get("pred_before", raw.get("pred", ""))),
        pred_after=str(raw.get("pred_after", raw.get("pred", ""))),
        candidate_a=str(raw["candidate_a"]),
        candidate_b=str(raw["candidate_b"]),
        margin_before=float(raw.get("margin_before", 0.0)),
        margin_after=float(raw.get("margin_after", raw.get("margin_before", 0.0))),
        gate=float(raw.get("gate", 0.0)),
        local_evidence=float(raw.get("local_evidence", 0.0)),
        part_prior=float(raw.get("part_prior", 0.0)),
        calibration_delta=float(raw.get("calibration_delta", 0.0)),
        num_neighbors=int(raw.get("num_neighbors", 0)),
        relation_types=[str(item) for item in raw.get("relation_types", [])],
    )


def summarize_records(records: list[Record]) -> dict[str, list[dict[str, Any]]]:
    return {
        "run_summary": run_summary(records),
        "candidate_pair_summary": candidate_pair_summary(records),
        "margin_bin_summary": margin_bin_summary(records),
        "evidence_source_summary": evidence_source_summary(records),
        "part_summary": part_summary(records),
    }


def run_summary(records: list[Record]) -> list[dict[str, Any]]:
    groups = group_by(records, lambda row: row.run_key)
    rows = []
    for key, items in sorted(groups.items()):
        dataset, method, variant, budget, seed = key
        rows.append(
            {
                "dataset": dataset,
                "method": method,
                "variant": variant,
                "budget": budget,
                "seed": seed,
                "num_faces": len(items),
                "num_parts": len({item.part_id for item in items}),
                "accuracy_before": ratio(item.before_correct for item in items),
                "accuracy_after": ratio(item.after_correct for item in items),
                "corrected_faces": sum(item.corrected_by_calibration for item in items),
                "damaged_faces": sum(item.damaged_by_calibration for item in items),
                "mean_margin_before": mean_value(item.margin_before for item in items),
                "mean_margin_after": mean_value(item.margin_after for item in items),
                "mean_abs_calibration_delta": mean_value(abs(item.calibration_delta) for item in items),
            }
        )
    return rows


def candidate_pair_summary(records: list[Record]) -> list[dict[str, Any]]:
    groups = group_by(records, lambda row: row.run_key + row.pair_key)
    rows = []
    for key, items in sorted(groups.items()):
        dataset, method, variant, budget, seed, class_a, class_b = key
        rows.append(
            {
                "dataset": dataset,
                "method": method,
                "variant": variant,
                "budget": budget,
                "seed": seed,
                "class_a": class_a,
                "class_b": class_b,
                "num_faces": len(items),
                "errors_before": sum(not item.before_correct for item in items),
                "errors_after": sum(not item.after_correct for item in items),
                "corrected_faces": sum(item.corrected_by_calibration for item in items),
                "damaged_faces": sum(item.damaged_by_calibration for item in items),
                "mean_margin_before": mean_value(item.margin_before for item in items),
                "mean_local_evidence": mean_value(item.local_evidence for item in items),
                "mean_part_prior": mean_value(item.part_prior for item in items),
            }
        )
    return rows


def margin_bin_summary(records: list[Record]) -> list[dict[str, Any]]:
    groups = group_by(records, lambda row: row.run_key + (margin_bin(row.margin_before),))
    rows = []
    for key, items in sorted(groups.items()):
        dataset, method, variant, budget, seed, bin_name = key
        rows.append(
            {
                "dataset": dataset,
                "method": method,
                "variant": variant,
                "budget": budget,
                "seed": seed,
                "margin_bin": bin_name,
                "num_faces": len(items),
                "accuracy_before": ratio(item.before_correct for item in items),
                "accuracy_after": ratio(item.after_correct for item in items),
                "corrected_faces": sum(item.corrected_by_calibration for item in items),
                "mean_gate": mean_value(item.gate for item in items),
                "mean_abs_calibration_delta": mean_value(abs(item.calibration_delta) for item in items),
            }
        )
    return rows


def evidence_source_summary(records: list[Record]) -> list[dict[str, Any]]:
    groups = group_by(records, lambda row: row.run_key)
    rows = []
    for key, items in sorted(groups.items()):
        dataset, method, variant, budget, seed = key
        rows.append(
            {
                "dataset": dataset,
                "method": method,
                "variant": variant,
                "budget": budget,
                "seed": seed,
                "num_faces": len(items),
                "mean_local_evidence_when_corrected": mean_value(
                    item.local_evidence for item in items if item.corrected_by_calibration
                ),
                "mean_part_prior_when_corrected": mean_value(
                    item.part_prior for item in items if item.corrected_by_calibration
                ),
                "mean_local_evidence_when_unchanged_error": mean_value(
                    item.local_evidence for item in items if not item.before_correct and not item.after_correct
                ),
                "mean_part_prior_when_unchanged_error": mean_value(
                    item.part_prior for item in items if not item.before_correct and not item.after_correct
                ),
                "mean_neighbors_when_corrected": mean_value(
                    item.num_neighbors for item in items if item.corrected_by_calibration
                ),
            }
        )
    return rows


def part_summary(records: list[Record]) -> list[dict[str, Any]]:
    groups = group_by(records, lambda row: row.run_key + (row.part_id,))
    rows = []
    for key, items in sorted(groups.items()):
        dataset, method, variant, budget, seed, part_id = key
        relation_counter: defaultdict[str, int] = defaultdict(int)
        for item in items:
            for relation_type in item.relation_types:
                relation_counter[relation_type] += 1
        rows.append(
            {
                "dataset": dataset,
                "method": method,
                "variant": variant,
                "budget": budget,
                "seed": seed,
                "part_id": part_id,
                "num_faces": len(items),
                "accuracy_before": ratio(item.before_correct for item in items),
                "accuracy_after": ratio(item.after_correct for item in items),
                "corrected_faces": sum(item.corrected_by_calibration for item in items),
                "damaged_faces": sum(item.damaged_by_calibration for item in items),
                "mean_neighbors": mean_value(item.num_neighbors for item in items),
                "relation_type_counts": json.dumps(dict(sorted(relation_counter.items())), sort_keys=True),
            }
        )
    return rows


def write_outputs(
    summary_dir: Path,
    records: list[Record],
    summaries: dict[str, list[dict[str, Any]]],
    input_paths: list[Path],
) -> None:
    manifest = {
        "input_jsonl": [str(path) for path in input_paths],
        "num_records": len(records),
        "outputs": {},
    }
    for name, rows in summaries.items():
        csv_path = summary_dir / f"{name}.csv"
        json_path = summary_dir / f"{name}.json"
        write_csv(csv_path, rows)
        json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        manifest["outputs"][name] = {"csv": str(csv_path), "json": str(json_path), "rows": len(rows)}
    manifest_path = summary_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def group_by(records: Iterable[Record], key_fn) -> dict[tuple[Any, ...], list[Record]]:
    groups: defaultdict[tuple[Any, ...], list[Record]] = defaultdict(list)
    for record in records:
        groups[key_fn(record)].append(record)
    return dict(groups)


def ratio(values: Iterable[bool]) -> float:
    items = list(values)
    if not items:
        return 0.0
    return round(sum(bool(item) for item in items) / len(items), 6)


def mean_value(values: Iterable[float]) -> float:
    items = [float(item) for item in values]
    if not items:
        return 0.0
    return round(mean(items), 6)


def margin_bin(value: float) -> str:
    for name, lo, hi in MARGIN_BINS:
        if lo <= value < hi:
            return name
    return "out_of_range"


if __name__ == "__main__":
    raise SystemExit(main())
