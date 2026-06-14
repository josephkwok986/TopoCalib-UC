"""Shared helpers for HybridBrep-based frozen representation scripts."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from typing import Any, Iterable

import torch


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def ensure_repo_imports() -> None:
    root = repo_root()
    for path in [root, root / "baselines" / "hybridbrep"]:
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)


def import_hybridbrep_symbols() -> tuple[Any, Any]:
    ensure_repo_imports()
    module = importlib.import_module("hybridbrep")
    return getattr(module, "HPart"), getattr(module, "GeneralConvEncDec")


def load_hybridbrep_graph(path: Path):
    return torch.load(path, map_location="cpu")


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(_json_safe(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def iter_partgraph_step_sources(cache_dir: Path) -> list[dict[str, Any]]:
    ensure_repo_imports()
    from topocalib_uc.train.partgraph_dataset import PartGraphCache

    cache = PartGraphCache(cache_dir)
    rows: list[dict[str, Any]] = []
    for record in cache.records:
        step_path = record.source_paths.get("step")
        if not step_path:
            continue
        rows.append(
            {
                "part_id": record.part_id,
                "step_path": str(step_path),
                "dataset": record.dataset,
                "split": record.split,
                "num_faces": record.num_faces,
                "source_paths": record.source_paths,
                "meta": record.meta,
            }
        )
    return rows


def iter_step_sources(step_paths: Iterable[Path]) -> list[dict[str, Any]]:
    rows = []
    for path in step_paths:
        rows.append(
            {
                "part_id": path.stem,
                "step_path": str(path),
                "dataset": "",
                "split": "",
                "num_faces": None,
                "source_paths": {"step": str(path)},
                "meta": {},
            }
        )
    return rows


def sanitize_tensors(data: Any) -> Any:
    for name, value in data:
        if isinstance(value, torch.Tensor) and value.is_floating_point():
            value.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
            setattr(data, name, value)
    return data


def move_to_device(data: Any, device: torch.device) -> Any:
    if hasattr(data, "to"):
        return data.to(device)
    for name, value in data:
        if isinstance(value, torch.Tensor):
            setattr(data, name, value.to(device))
    return data


def load_ssrl_model(ckpt_path: Path | None, *, embedding_dim: int | None, hidden_dim: int | None, layers: int | None, attn_heads: int | None, device: torch.device):
    _, GeneralConvEncDec = import_hybridbrep_symbols()
    config: dict[str, Any] = {}
    state_dict = None
    if ckpt_path is not None:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        if isinstance(ckpt, dict):
            config = dict(ckpt.get("config", {}))
            state_dict = ckpt.get("state_dict", ckpt)
        else:
            raise ValueError(f"Unsupported checkpoint payload type: {type(ckpt).__name__}")
    emb_dim = int(embedding_dim or config.get("embedding_dim", 256))
    dec_dim = int(hidden_dim or config.get("hidden_dim", 1024))
    dec_layers = int(layers or config.get("layers", 4))
    heads = int(attn_heads or config.get("attn_heads", 16))
    model = GeneralConvEncDec(emb_dim=emb_dim, dec_dim=dec_dim, dec_layers=dec_layers, attn_heads=heads)
    if state_dict is not None:
        model.load_state_dict(state_dict, strict=False)
    model.to(device)
    return model, {"embedding_dim": emb_dim, "hidden_dim": dec_dim, "layers": dec_layers, "attn_heads": heads}


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    return value
