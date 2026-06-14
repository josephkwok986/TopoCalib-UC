"""Frozen face embedding cache utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch


@dataclass(frozen=True)
class FrozenEmbeddingRecord:
    part_id: str
    encoder: str
    checkpoint: str | None
    embedding_dim: int
    z: torch.Tensor
    source_paths: dict[str, Any]
    meta: dict[str, Any]


class FrozenEmbeddingCache:
    """Read face-level frozen embeddings from a directory cache."""

    def __init__(self, cache_dir: str | Path):
        self.cache_dir = Path(cache_dir)
        self.parts_dir = self.cache_dir / "parts"
        if not self.parts_dir.exists():
            raise FileNotFoundError(f"Missing frozen embedding parts directory: {self.parts_dir}")
        self.records = self._load_records()
        if not self.records:
            raise ValueError(f"No frozen embedding .pt files found in {self.parts_dir}")
        self.part_ids = [record.part_id for record in self.records]
        self.by_id = {record.part_id: record for record in self.records}
        if len(self.by_id) != len(self.part_ids):
            raise ValueError("Frozen embedding part IDs must be unique inside one cache.")

    def _load_records(self) -> list[FrozenEmbeddingRecord]:
        records: list[FrozenEmbeddingRecord] = []
        for path in sorted(self.parts_dir.glob("*.pt")):
            payload = torch.load(path, map_location="cpu")
            records.append(_record_from_payload(path, payload))
        return records

    def embedding_for_part(self, part_id: str, *, expected_faces: int | None = None) -> torch.Tensor:
        if part_id not in self.by_id:
            raise KeyError(f"Missing frozen embedding for part_id={part_id!r}")
        z = self.by_id[part_id].z
        if expected_faces is not None and int(z.shape[0]) != int(expected_faces):
            raise ValueError(
                f"Frozen embedding face count mismatch for part_id={part_id!r}: "
                f"z rows {int(z.shape[0])} != expected_faces {int(expected_faces)}"
            )
        return z

    def transform_cache(self, partgraph_cache) -> dict[str, torch.Tensor]:
        return {
            record.part_id: self.embedding_for_part(record.part_id, expected_faces=record.num_faces)
            for record in partgraph_cache.records
        }


def save_embedding_record(path: str | Path, record: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(record)
    payload["z"] = torch.as_tensor(payload["z"], dtype=torch.float32).detach().cpu()
    payload["embedding_dim"] = int(payload.get("embedding_dim", payload["z"].shape[1]))
    payload.setdefault("source_paths", {})
    payload.setdefault("meta", {})
    torch.save(payload, path)


def _record_from_payload(path: Path, payload: Mapping[str, Any]) -> FrozenEmbeddingRecord:
    if "part_id" not in payload:
        raise ValueError(f"{path}: missing part_id")
    if "z" not in payload:
        raise ValueError(f"{path}: missing z")
    z = torch.as_tensor(payload["z"], dtype=torch.float32).detach().cpu()
    if z.ndim != 2:
        raise ValueError(f"{path}: z must be rank-2 [num_faces, embedding_dim], got shape {tuple(z.shape)}")
    embedding_dim = int(payload.get("embedding_dim", z.shape[1]))
    if embedding_dim != int(z.shape[1]):
        raise ValueError(f"{path}: embedding_dim {embedding_dim} != z.shape[1] {int(z.shape[1])}")
    return FrozenEmbeddingRecord(
        part_id=str(payload["part_id"]),
        encoder=str(payload.get("encoder", "")),
        checkpoint=str(payload["checkpoint"]) if payload.get("checkpoint") is not None else None,
        embedding_dim=embedding_dim,
        z=z,
        source_paths=dict(payload.get("source_paths", {})),
        meta=dict(payload.get("meta", {})),
    )
