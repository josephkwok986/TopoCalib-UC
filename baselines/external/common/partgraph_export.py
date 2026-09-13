"""Shared PartGraph export helpers for external baseline adapters."""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from data_protocol.io import write_json
from topocalib_uc.train.partgraph_dataset import PartGraphCache, PartRecord


@dataclass(frozen=True)
class ExportedPart:
    part_id: str
    split: str
    num_faces: int
    labels: list[int]
    edges: list[list[int]]
    edge_type: list[int]
    surface_type: list[int]
    step_path: str | None


def load_exported_parts(cache_dir: str | Path) -> list[ExportedPart]:
    cache = PartGraphCache(cache_dir)
    return [record_to_exported_part(record) for record in cache.records]


def record_to_exported_part(record: PartRecord) -> ExportedPart:
    return ExportedPart(
        part_id=record.part_id,
        split=normalize_split(record.split),
        num_faces=record.num_faces,
        labels=[int(x) for x in record.y.tolist()],
        edges=[[int(a), int(b)] for a, b in record.edges.tolist()],
        edge_type=[int(x) for x in record.edge_type.tolist()],
        surface_type=[int(x) for x in record.surface_type.tolist()],
        step_path=_optional_str(record.source_paths.get("step")),
    )


def normalize_split(split: str) -> str:
    value = str(split).lower()
    if value in {"train", "training", "training_set"}:
        return "train"
    if value in {"val", "valid", "validation", "validation_set"}:
        return "val"
    if value in {"test", "testing", "test_set"}:
        return "test"
    return value or "train"


def split_part_ids(parts: Iterable[ExportedPart]) -> dict[str, list[str]]:
    result = {"train": [], "val": [], "test": []}
    for part in parts:
        result.setdefault(part.split, []).append(part.part_id)
    for key in result:
        result[key] = sorted(result[key])
    return result


def class_names(parts: Iterable[ExportedPart]) -> list[str]:
    labels = sorted({label for part in parts for label in part.labels})
    return [str(label) for label in labels]


def write_text_lines(path: Path, lines: Iterable[str]) -> None:
    text = "\n".join(str(line) for line in lines)
    if text:
        text += "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_export_manifest(path: Path, *, dataset: str, parts: list[ExportedPart], extra: dict[str, Any] | None = None) -> None:
    payload: dict[str, Any] = {
        "dataset": dataset,
        "num_parts": len(parts),
        "num_faces": int(sum(part.num_faces for part in parts)),
        "classes": class_names(parts),
        "splits": split_part_ids(parts),
        "parts": [
            {
                "part_id": part.part_id,
                "split": part.split,
                "num_faces": part.num_faces,
                "step_path": part.step_path,
            }
            for part in sorted(parts, key=lambda item: item.part_id)
        ],
    }
    if extra:
        payload.update(extra)
    write_json(path, payload)


def write_labels_txt(path: Path, labels: Iterable[int]) -> None:
    write_text_lines(path, [str(int(label)) for label in labels])


def write_labels_json_array(path: Path, labels: Iterable[int]) -> None:
    write_json(path, [int(label) for label in labels])


def link_or_record_step_files(parts: list[ExportedPart], step_dir: Path) -> list[dict[str, str | None]]:
    step_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, str | None]] = []
    for part in parts:
        target = step_dir / f"{part.part_id}.step"
        source = part.step_path
        if source is None:
            raise FileNotFoundError(f"Part {part.part_id!r} has no STEP source path")
        source_path = Path(source)
        if not source_path.is_file():
            raise FileNotFoundError(f"STEP source for part {part.part_id!r} does not exist: {source_path}")
        source_path = source_path.resolve()

        if target.exists() or target.is_symlink():
            if target.is_dir() and not target.is_symlink():
                raise IsADirectoryError(f"STEP target is a directory: {target}")
            try:
                if target.samefile(source_path):
                    records.append(
                        {"part_id": part.part_id, "source": str(source_path), "local_step": str(target)}
                    )
                    continue
            except OSError:
                pass
            target.unlink()

        try:
            os.symlink(source_path, target)
        except OSError:
            shutil.copy2(source_path, target)
        records.append({"part_id": part.part_id, "source": str(source_path), "local_step": str(target)})
    return records


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)
