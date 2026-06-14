"""MFCAD++ PartGraph cache preprocessing."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Iterator

import h5py
import numpy as np

from .constants import (
    EDGE_TYPE_CONCAVE,
    EDGE_TYPE_CONVEX,
    EDGE_TYPE_OTHER,
    EDGE_TYPE_SMOOTH,
)
from .error_log import append_error
from .io import append_jsonl, save_part_graph, write_json
from .ssrl_filter import check_step_geometry, find_step_files


def decode_cad_model(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.bytes_):
        return bytes(value).decode("utf-8")
    return str(value)


def split_name_from_h5(path: Path) -> str:
    name = path.name.lower()
    if name.startswith("training"):
        return "train"
    if name.startswith("val"):
        return "val"
    if name.startswith("test"):
        return "test"
    return path.stem


def group_part_ranges(group) -> Iterator[tuple[str, int, int]]:
    model_ids = [decode_cad_model(x) for x in group["CAD_model"][()]]
    labels = group["labels"][()]
    idx = group["idx"][()]
    face_base = int(idx[-1, 0]) - len(labels) + 1
    prev_face_end = face_base - 1
    for model_id, row in zip(model_ids, idx):
        face_end = int(row[0])
        local_start = prev_face_end + 1 - face_base
        local_end = face_end + 1 - face_base
        prev_face_end = face_end
        yield model_id, int(local_start), int(local_end)


def _local_edges(indices: np.ndarray, start: int, end: int) -> np.ndarray:
    if indices.size == 0:
        return np.zeros((0, 2), dtype=np.int64)
    arr = np.asarray(indices, dtype=np.int64)
    mask = (
        (arr[:, 0] >= start)
        & (arr[:, 0] < end)
        & (arr[:, 1] >= start)
        & (arr[:, 1] < end)
        & (arr[:, 0] != arr[:, 1])
    )
    local = arr[mask] - start
    if local.size == 0:
        return np.zeros((0, 2), dtype=np.int64)
    local = np.sort(local, axis=1)
    return np.unique(local, axis=0).astype(np.int64)


def _edge_dict_from_group(group, start: int, end: int) -> dict[tuple[int, int], int]:
    edges: dict[tuple[int, int], int] = {}
    for edge in _local_edges(group["A_1_idx"][()], start, end):
        edges[tuple(int(x) for x in edge)] = EDGE_TYPE_OTHER
    for name, edge_type in [
        ("E_1_idx", EDGE_TYPE_CONVEX),
        ("E_2_idx", EDGE_TYPE_CONCAVE),
        ("E_3_idx", EDGE_TYPE_SMOOTH),
    ]:
        if name not in group:
            continue
        for edge in _local_edges(group[name][()], start, end):
            edges[tuple(int(x) for x in edge)] = edge_type
    return edges


def _surface_type_from_v1(v1: np.ndarray) -> np.ndarray:
    # The provided HDF5 V_1[:, 4] values are normalized discrete surface IDs.
    # Multiplying by 11 recovers the observed primitive IDs; normalize to
    # plane/cylinder/cone/sphere/torus = 0..4 for the paper adapter.
    raw = np.rint(np.asarray(v1[:, 4], dtype=np.float32) * 11.0).astype(np.int64)
    if raw.size == 0:
        return raw
    if int(raw.min()) >= 1 and int(raw.max()) <= 5:
        return raw - 1
    if int(raw.min()) >= 0 and int(raw.max()) <= 4:
        return raw
    raise ValueError(f"Unsupported MFCAD++ surface type ids: {sorted(set(int(x) for x in raw.tolist()))}")


def _part_from_group(
    *,
    group,
    model_id: str,
    split: str,
    start: int,
    end: int,
    h5_path: Path,
    step_path: Path,
) -> dict[str, Any]:
    labels = np.asarray(group["labels"][()][start:end], dtype=np.int64)
    v1 = np.asarray(group["V_1"][()][start:end], dtype=np.float32)
    edge_dict = _edge_dict_from_group(group, start, end)
    if edge_dict:
        edges = np.asarray(sorted(edge_dict.keys()), dtype=np.int64)
        edge_type = np.asarray([edge_dict[tuple(edge)] for edge in edges], dtype=np.int64)
    else:
        edges = np.zeros((0, 2), dtype=np.int64)
        edge_type = np.zeros((0,), dtype=np.int64)

    return {
        "part_id": model_id,
        "dataset": "mfcadpp",
        "split": split,
        "num_faces": int(labels.shape[0]),
        "y": labels,
        "surface_type": _surface_type_from_v1(v1),
        "edges": edges,
        "edge_type": edge_type,
        "face_features_raw": v1,
        "source_paths": {"h5": str(h5_path), "step": str(step_path)},
        "meta": {
            "ssrl_filter_passed": True,
            "h5_group": group.name,
            "surface_type_note": "Recovered from normalized V_1[:, 4] by round(value * 11), normalized to primitive ids 0..4.",
        },
    }


def iter_h5_paths(graph_dir: Path) -> list[Path]:
    preferred = ["training_MFCAD++.h5", "val_MFCAD++.h5", "test_MFCAD++.h5"]
    paths = [graph_dir / name for name in preferred if (graph_dir / name).exists()]
    seen = {path.name for path in paths}
    paths.extend(path for path in sorted(graph_dir.glob("*.h5")) if path.name not in seen)
    return paths


def build_mfcadpp_partgraph_cache(
    *,
    dataset_root: Path,
    output_root: Path,
    cache_name: str | None,
    error_log: Path,
) -> dict[str, int | str]:
    graph_dir = dataset_root / "hierarchical_graphs"
    step_dir = dataset_root / "step"
    cache_dir = output_root / (cache_name or "mfcadpp_partgraph")
    parts_dir = cache_dir / "parts"
    manifest_path = cache_dir / "manifest.jsonl"
    summary_path = cache_dir / "summary.json"

    if manifest_path.exists():
        manifest_path.unlink()

    step_files = find_step_files(step_dir)
    print(f"[mfcadpp] start dataset_root={dataset_root} graph_dir={graph_dir} step_dir={step_dir}", flush=True)
    print(f"[mfcadpp] output cache_dir={cache_dir} parts_dir={parts_dir} error_log={error_log}", flush=True)
    print(f"[mfcadpp] discovered_step_files={len(step_files)}", flush=True)
    scanned = 0
    written = 0
    rejected = 0

    for h5_path in iter_h5_paths(graph_dir):
        split = split_name_from_h5(h5_path)
        with h5py.File(h5_path, "r") as h5:
            group_names = sorted(h5.keys(), key=lambda value: int(value) if value.isdigit() else value)
            for group_name in group_names:
                group = h5[group_name]
                for model_id, start, end in group_part_ranges(group):
                    scanned += 1

                    step_path = step_files.get(model_id)
                    if step_path is None:
                        rejected += 1
                        record = {
                            "dataset": "mfcadpp",
                            "part_id": model_id,
                            "split": split,
                            "status": "rejected",
                            "reason": "missing_step",
                            "h5_path": str(h5_path),
                        }
                        append_jsonl(manifest_path, record)
                        append_error("MFCAD++ part missing STEP file", error_log=error_log, context=record)
                        print(f"[mfcadpp] rejected part_id={model_id} split={split} reason=missing_step", flush=True)
                        continue

                    expected_faces = int(end - start)
                    geometry_check = check_step_geometry(step_path, expected_faces=expected_faces)
                    if not geometry_check.passed:
                        rejected += 1
                        record = {
                            "dataset": "mfcadpp",
                            "part_id": model_id,
                            "split": split,
                            "status": "rejected",
                            "reason": geometry_check.reason,
                            "step_path": str(step_path),
                            "h5_path": str(h5_path),
                        }
                        append_jsonl(manifest_path, record)
                        append_error("MFCAD++ part rejected by SSRL filter", error_log=error_log, context=record)
                        print(f"[mfcadpp] rejected part_id={model_id} split={split} reason={geometry_check.reason}", flush=True)
                        continue

                    try:
                        part = _part_from_group(
                            group=group,
                            model_id=model_id,
                            split=split,
                            start=start,
                            end=end,
                            h5_path=h5_path,
                            step_path=step_path,
                        )
                        out_path = parts_dir / f"{model_id}.pt"
                        serialization = save_part_graph(out_path, part)
                        written += 1
                        append_jsonl(
                            manifest_path,
                            {
                                "dataset": "mfcadpp",
                                "part_id": model_id,
                                "split": split,
                                "status": "written",
                                "path": str(out_path),
                                "num_faces": int(part["num_faces"]),
                                "num_edges": int(part["edges"].shape[0]),
                                "serialization": serialization,
                            },
                        )
                        print(
                            f"[mfcadpp] written part_id={model_id} split={split} "
                            f"faces={int(part['num_faces'])} edges={int(part['edges'].shape[0])}",
                            flush=True,
                        )
                    except Exception as exc:
                        rejected += 1
                        record = {
                            "dataset": "mfcadpp",
                            "part_id": model_id,
                            "split": split,
                            "status": "error",
                            "reason": f"{type(exc).__name__}: {exc}",
                            "step_path": str(step_path),
                            "h5_path": str(h5_path),
                        }
                        append_jsonl(manifest_path, record)
                        append_error("MFCAD++ preprocessing error", error_log=error_log, context=record)
                        print(f"[mfcadpp] error part_id={model_id} split={split} reason={type(exc).__name__}: {exc}", flush=True)

    summary = {
        "dataset": "mfcadpp",
        "cache_dir": str(cache_dir),
        "scanned_parts": scanned,
        "written_parts": written,
        "rejected_or_error_parts": rejected,
        "requested_parts": "all",
    }
    write_json(summary_path, summary)
    print(f"[mfcadpp] done summary={summary}", flush=True)
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build MFCAD++ PartGraph cache.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--cache-name", type=str, required=True)
    parser.add_argument("--error-log", type=Path, required=True)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    summary = build_mfcadpp_partgraph_cache(
        dataset_root=args.dataset_root,
        output_root=args.output_root,
        cache_name=args.cache_name,
        error_log=args.error_log,
    )
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
