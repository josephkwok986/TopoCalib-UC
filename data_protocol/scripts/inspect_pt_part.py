#!/usr/bin/env python3
"""Write a compact JSON preview for one cached PartGraph."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from data_protocol.io import load_part_graph


def shape_of(value: Any) -> list[int] | None:
    if hasattr(value, "shape"):
        return [int(x) for x in value.shape]
    if hasattr(value, "size") and callable(value.size):
        try:
            return [int(x) for x in value.size()]
        except Exception:
            return None
    return None


def to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def preview_array(value: Any, limit: int = 8) -> list[Any]:
    arr = to_numpy(value).reshape(-1)
    return [x.item() if hasattr(x, "item") else x for x in arr[:limit]]


def summarize_part(part: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "part_id": part.get("part_id"),
        "dataset": part.get("dataset"),
        "split": part.get("split"),
        "num_faces": int(part.get("num_faces", -1)),
        "source_paths": part.get("source_paths", {}),
        "meta": part.get("meta", {}),
        "fields": {},
    }
    for key in ["y", "surface_type", "edges", "edge_type", "face_features_raw"]:
        value = part.get(key)
        if value is None:
            continue
        arr = to_numpy(value)
        field = {
            "shape": shape_of(value),
            "dtype": str(arr.dtype),
            "preview": preview_array(value),
        }
        if arr.size and key in {"y", "surface_type", "edge_type"}:
            unique, counts = np.unique(arr, return_counts=True)
            field["unique_counts"] = {
                str(int(k)): int(v) for k, v in zip(unique[:32], counts[:32])
            }
        summary["fields"][key] = field
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect one cached PartGraph .pt file.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    summary = summarize_part(load_part_graph(args.input))
    text = json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True)
    args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
