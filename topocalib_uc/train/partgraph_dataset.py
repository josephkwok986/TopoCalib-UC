"""PartGraph cache loading utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from data_protocol.io import load_part_graph


@dataclass(frozen=True)
class PartRecord:
    index: int
    part_id: str
    dataset: str
    split: str
    path: Path
    num_faces: int
    y: torch.Tensor
    surface_type: torch.Tensor
    edges: torch.Tensor
    edge_type: torch.Tensor
    face_features_raw: torch.Tensor
    source_paths: dict[str, Any]
    meta: dict[str, Any]

    @property
    def class_set(self) -> set[int]:
        return {int(x) for x in torch.unique(self.y).tolist()}


class PartGraphCache:
    """Load a directory of PartGraph .pt files into plain torch tensors."""

    def __init__(self, cache_dir: str | Path):
        self.cache_dir = Path(cache_dir)
        self.parts_dir = self.cache_dir / "parts"
        self.records = self._load_records()
        if not self.records:
            raise ValueError(f"No .pt files found in {self.parts_dir}")
        self.part_ids = [record.part_id for record in self.records]
        self.id_to_index = {part_id: idx for idx, part_id in enumerate(self.part_ids)}
        if len(self.id_to_index) != len(self.part_ids):
            raise ValueError("Part IDs must be unique inside one cache.")
        self.classes = sorted({label for record in self.records for label in record.class_set})

    def _load_records(self) -> list[PartRecord]:
        records: list[PartRecord] = []
        for idx, path in enumerate(sorted(self.parts_dir.glob("*.pt"))):
            part = load_part_graph(path)
            y = _as_long_tensor(part["y"])
            surface_type = _as_long_tensor(part["surface_type"])
            edges = _as_long_tensor(part["edges"]).reshape(-1, 2)
            edge_type = _as_long_tensor(part["edge_type"]).reshape(-1)
            face_features_raw = _as_float_tensor(part["face_features_raw"])
            num_faces = int(part.get("num_faces", y.numel()))
            _validate_part(path, num_faces, y, surface_type, edges, edge_type, face_features_raw)
            records.append(
                PartRecord(
                    index=idx,
                    part_id=str(part["part_id"]),
                    dataset=str(part.get("dataset", "")),
                    split=str(part.get("split", "")),
                    path=path,
                    num_faces=num_faces,
                    y=y,
                    surface_type=surface_type,
                    edges=edges,
                    edge_type=edge_type,
                    face_features_raw=face_features_raw,
                    source_paths=dict(part.get("source_paths", {})),
                    meta=dict(part.get("meta", {})),
                )
            )
        return records

    def by_part_id(self, part_id: str) -> PartRecord:
        return self.records[self.id_to_index[part_id]]

    def labels_for_parts(self, part_ids: list[str] | set[str]) -> set[int]:
        return {label for part_id in part_ids for label in self.by_part_id(part_id).class_set}


def _as_long_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().long()
    return torch.as_tensor(value, dtype=torch.long)


def _as_float_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().float()
    return torch.as_tensor(value, dtype=torch.float32)


def _validate_part(
    path: Path,
    num_faces: int,
    y: torch.Tensor,
    surface_type: torch.Tensor,
    edges: torch.Tensor,
    edge_type: torch.Tensor,
    face_features_raw: torch.Tensor,
) -> None:
    if y.numel() != num_faces:
        raise ValueError(f"{path}: y length {y.numel()} != num_faces {num_faces}")
    if surface_type.numel() != num_faces:
        raise ValueError(f"{path}: surface_type length {surface_type.numel()} != num_faces {num_faces}")
    if face_features_raw.shape[0] != num_faces:
        raise ValueError(f"{path}: face_features_raw rows {face_features_raw.shape[0]} != num_faces {num_faces}")
    if edges.shape[0] != edge_type.numel():
        raise ValueError(f"{path}: edges rows {edges.shape[0]} != edge_type length {edge_type.numel()}")
    if edges.numel():
        if int(edges.min()) < 0 or int(edges.max()) >= num_faces:
            raise ValueError(f"{path}: edge index outside [0, {num_faces})")
