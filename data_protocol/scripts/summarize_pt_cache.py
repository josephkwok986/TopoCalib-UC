#!/usr/bin/env python3
"""Summarize a cached PartGraph directory."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from data_protocol.io import load_part_graph


def to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize all PartGraph .pt files in a cache.")
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    parts_dir = args.cache_dir / "parts"
    paths = sorted(parts_dir.glob("*.pt"))
    label_counts: Counter[int] = Counter()
    surface_counts: Counter[int] = Counter()
    edge_type_counts: Counter[int] = Counter()
    total_faces = 0
    total_edges = 0
    serializations: Counter[str] = Counter()

    for path in paths:
        part = load_part_graph(path)
        y = to_numpy(part["y"]).astype(int)
        surface = to_numpy(part["surface_type"]).astype(int)
        edge_type = to_numpy(part["edge_type"]).astype(int)
        total_faces += int(y.shape[0])
        total_edges += int(to_numpy(part["edges"]).shape[0])
        label_counts.update(int(x) for x in y)
        surface_counts.update(int(x) for x in surface)
        edge_type_counts.update(int(x) for x in edge_type)
        serializations.update([str(part.get("meta", {}).get("serialization", "unknown"))])

    summary = {
        "cache_dir": str(args.cache_dir),
        "num_parts": len(paths),
        "total_faces": total_faces,
        "total_edges": total_edges,
        "label_counts": {str(k): v for k, v in sorted(label_counts.items())},
        "surface_type_counts": {str(k): v for k, v in sorted(surface_counts.items())},
        "edge_type_counts": {str(k): v for k, v in sorted(edge_type_counts.items())},
        "serializations": dict(serializations),
    }
    text = json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True)
    args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
