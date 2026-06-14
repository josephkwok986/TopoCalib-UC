"""Import external baseline metrics into the repository result schema."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from statistics import mean
from typing import Any

from data_protocol.io import write_json


METRIC_ALIASES = {
    "miou": {"miou", "m_iou", "mean_iou", "mean iou", "mean intersection over union"},
    "accuracy": {"accuracy", "acc", "overall_accuracy", "overall acc", "oa"},
    "num_faces": {"num_faces", "faces", "n_faces"},
}


def add_import_args(parser: argparse.ArgumentParser, *, method: str, dataset: str) -> None:
    parser.add_argument("--raw-output-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--budget", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--metrics-json", type=Path, required=True)
    parser.add_argument("--num-classes", type=int, required=True)


def import_external_result(args: argparse.Namespace) -> dict[str, Any]:
    source_files = _source_files(args.raw_output_dir, args.metrics_json)
    payloads = [_read_source(path) for path in source_files]
    metrics = _merge_metrics([_extract_metrics(payload) for payload in payloads])
    result = {
        "method": str(args.method),
        "dataset": str(args.dataset),
        "config": {
            "budget": _coerce_budget(args.budget),
            "labeled_part_budget": _coerce_budget(args.budget),
            "seed": int(args.seed),
            "split": str(args.split),
            "num_classes": args.num_classes,
        },
        "metrics": {
            str(args.split): metrics,
        },
        "external_baseline": {
            "raw_output_dir": str(args.raw_output_dir),
            "source_files": [str(path) for path in source_files],
        },
    }
    write_json(args.output, result)
    return result


def _source_files(raw_output_dir: Path, metrics_json: Path | None) -> list[Path]:
    if metrics_json is not None:
        return [metrics_json]
    preferred_names = {
        "metrics.json",
        "results.json",
        "result.json",
        "test_metrics.json",
        "test_results.json",
        "test_summary.json",
        "summary.json",
    }
    json_paths = sorted(path for path in raw_output_dir.rglob("*.json") if path.name in preferred_names)
    if json_paths:
        return json_paths
    text_paths = sorted(path for path in raw_output_dir.rglob("*") if path.suffix.lower() in {".log", ".txt"})
    if text_paths:
        return text_paths
    raise FileNotFoundError(f"No supported metric files found under {raw_output_dir}")


def _read_source(path: Path) -> Any:
    text = path.read_text(encoding="utf-8", errors="ignore")
    if path.suffix.lower() == ".json":
        return json.loads(text)
    return {"_text": text}


def _extract_metrics(payload: Any) -> dict[str, float]:
    flat = _flatten(payload)
    out: dict[str, float] = {}
    for target, aliases in METRIC_ALIASES.items():
        values = [value for key, value in flat.items() if _metric_key_match(key, aliases)]
        if values:
            out[target] = float(values[-1])
    if not out and isinstance(payload, dict) and "_text" in payload:
        out = _extract_metrics_from_text(str(payload["_text"]))
    if "miou" not in out:
        raise ValueError("Unable to locate mIoU in external baseline output.")
    return out


def _flatten(value: Any, prefix: str = "") -> dict[str, float]:
    rows: dict[str, float] = {}
    if isinstance(value, dict):
        for key, item in value.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            rows.update(_flatten(item, child))
    elif isinstance(value, list):
        for idx, item in enumerate(value):
            child = f"{prefix}.{idx}" if prefix else str(idx)
            rows.update(_flatten(item, child))
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        rows[prefix] = float(value)
    elif isinstance(value, str):
        try:
            rows[prefix] = float(value.strip().rstrip("%"))
        except ValueError:
            pass
    return rows


def _metric_key_match(key: str, aliases: set[str]) -> bool:
    tail = key.lower().replace("-", "_").replace(" ", "_").split(".")[-1]
    compact = tail.replace("_", "")
    return tail in aliases or compact in {alias.replace(" ", "").replace("_", "") for alias in aliases}


def _extract_metrics_from_text(text: str) -> dict[str, float]:
    out: dict[str, float] = {}
    patterns = {
        "miou": r"(?:mIoU|mean[_ -]?IoU)\s*[:=]\s*([0-9]*\.?[0-9]+%?)",
        "accuracy": r"(?:accuracy|acc|overall[_ -]?acc)\s*[:=]\s*([0-9]*\.?[0-9]+%?)",
        "num_faces": r"(?:num[_ -]?faces|faces)\s*[:=]\s*([0-9]*\.?[0-9]+)",
    }
    for name, pattern in patterns.items():
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        if matches:
            out[name] = _parse_number(matches[-1])
    return out


def _parse_number(value: str) -> float:
    text = str(value).strip()
    if text.endswith("%"):
        return float(text[:-1]) / 100.0
    return float(text)


def _merge_metrics(items: list[dict[str, float]]) -> dict[str, float]:
    keys = sorted({key for item in items for key in item})
    merged: dict[str, float] = {}
    for key in keys:
        values = [item[key] for item in items if key in item]
        merged[key] = float(mean(values))
    return merged


def _coerce_budget(value: str) -> int | str:
    return int(value) if str(value).isdigit() else str(value)
