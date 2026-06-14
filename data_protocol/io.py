"""Serialization and manifest helpers for preprocessing caches."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Mapping

import numpy as np


def require_torch():
    try:
        import torch  # type: ignore

        return torch
    except Exception as exc:
        raise RuntimeError(
            "PyTorch is required for PartGraph .pt caches. "
            "Install torch in the preprocessing environment before running stage A/B."
        ) from exc


def as_serializable_part(part: Mapping[str, Any]) -> dict[str, Any]:
    """Convert nested arrays to a torch-save friendly plain dictionary."""

    result: dict[str, Any] = dict(part)
    result.setdefault("meta", {})
    result["meta"] = dict(result["meta"])
    return result


def save_part_graph(path: Path, part: Mapping[str, Any]) -> str:
    """Save one PartGraph as a torch-loadable .pt file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = as_serializable_part(part)
    tensor_keys = {"y", "surface_type", "edges", "edge_type", "face_features_raw"}
    torch = require_torch()
    for key in tensor_keys:
        if key in payload and not hasattr(payload[key], "device"):
            arr = np.asarray(payload[key])
            if key in {"y", "surface_type", "edges", "edge_type"}:
                payload[key] = torch.as_tensor(arr, dtype=torch.long)
            else:
                payload[key] = torch.as_tensor(arr, dtype=torch.float32)
    payload["meta"]["serialization"] = "torch"
    torch.save(payload, path)
    return "torch"


def load_part_graph(path: Path) -> dict[str, Any]:
    """Load a PartGraph saved by save_part_graph."""

    torch = require_torch()
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")
    except RuntimeError as exc:
        if "Invalid magic number" not in str(exc):
            raise
        with path.open("rb") as f:
            return pickle.load(f)


def append_jsonl(path: Path, record: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
        f.write("\n")


def write_json(path: Path, record: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")
