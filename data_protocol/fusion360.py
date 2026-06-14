"""Fusion360 PartGraph cache preprocessing."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterator

from .cad_geometry import extract_cad_geometry
from .error_log import append_error
from .io import append_jsonl, save_part_graph, write_json
from .ssrl_filter import check_shape_geometry, find_step_files, read_step_shape


def parse_seg_file(path: Path) -> list[int]:
    labels: list[int] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            value = line.strip()
            if not value:
                continue
            try:
                labels.append(int(float(value)))
            except ValueError as exc:
                raise ValueError(f"Invalid label in {path} line {line_no}: {value!r}") from exc
    return labels


def iter_candidate_part_ids(step_dir: Path, seg_dir: Path) -> Iterator[str]:
    step_ids = set(find_step_files(step_dir))
    seg_ids = {path.stem for path in seg_dir.glob("*.seg")}
    for part_id in sorted(step_ids & seg_ids):
        yield part_id


def build_fusion360_partgraph_cache(
    *,
    dataset_root: Path,
    output_root: Path,
    cache_name: str | None,
    error_log: Path,
) -> dict[str, int | str]:
    step_dir = dataset_root / "breps" / "step"
    seg_dir = dataset_root / "breps" / "seg"
    cache_dir = output_root / (cache_name or "fusion360_partgraph")
    parts_dir = cache_dir / "parts"
    manifest_path = cache_dir / "manifest.jsonl"
    summary_path = cache_dir / "summary.json"

    if manifest_path.exists():
        manifest_path.unlink()

    step_files = find_step_files(step_dir)
    print(f"[fusion360] start dataset_root={dataset_root} step_dir={step_dir} seg_dir={seg_dir}", flush=True)
    print(f"[fusion360] output cache_dir={cache_dir} parts_dir={parts_dir} error_log={error_log}", flush=True)
    print(f"[fusion360] discovered_step_files={len(step_files)}", flush=True)
    scanned = 0
    written = 0
    rejected = 0

    for part_id in iter_candidate_part_ids(step_dir, seg_dir):
        scanned += 1

        step_path = step_files[part_id]
        seg_path = seg_dir / f"{part_id}.seg"
        try:
            labels = parse_seg_file(seg_path)
            shape = read_step_shape(step_path)
            geometry_check = check_shape_geometry(shape, expected_faces=len(labels))
            if not geometry_check.passed:
                rejected += 1
                record = {
                    "dataset": "fusion360",
                    "part_id": part_id,
                    "status": "rejected",
                    "reason": geometry_check.reason,
                    "step_path": str(step_path),
                    "seg_path": str(seg_path),
                }
                append_jsonl(manifest_path, record)
                append_error("Fusion360 part rejected by SSRL filter", error_log=error_log, context=record)
                print(f"[fusion360] rejected part_id={part_id} reason={geometry_check.reason}", flush=True)
                continue

            cad = extract_cad_geometry(shape)
            if len(labels) != int(cad.surface_type.shape[0]):
                rejected += 1
                reason = f"face_count_mismatch extracted={cad.surface_type.shape[0]} labels={len(labels)}"
                record = {
                    "dataset": "fusion360",
                    "part_id": part_id,
                    "status": "rejected",
                    "reason": reason,
                    "step_path": str(step_path),
                    "seg_path": str(seg_path),
                }
                append_jsonl(manifest_path, record)
                append_error("Fusion360 extracted face count mismatch", error_log=error_log, context=record)
                print(f"[fusion360] rejected part_id={part_id} reason={reason}", flush=True)
                continue

            out_path = parts_dir / f"{part_id}.pt"
            serialization = save_part_graph(
                out_path,
                {
                    "part_id": part_id,
                    "dataset": "fusion360",
                    "split": "subset",
                    "num_faces": len(labels),
                    "y": labels,
                    "surface_type": cad.surface_type,
                    "edges": cad.edges,
                    "edge_type": cad.edge_type,
                    "face_features_raw": cad.face_features_raw,
                    "source_paths": {"step": str(step_path), "seg": str(seg_path)},
                    "meta": {
                        "ssrl_filter_passed": True,
                        "num_solids": geometry_check.num_solids,
                        **cad.meta,
                    },
                },
            )
            written += 1
            append_jsonl(
                manifest_path,
                {
                    "dataset": "fusion360",
                    "part_id": part_id,
                    "status": "written",
                    "path": str(out_path),
                    "num_faces": len(labels),
                    "num_edges": int(cad.edges.shape[0]),
                    "serialization": serialization,
                },
            )
            print(f"[fusion360] written part_id={part_id} faces={len(labels)} edges={int(cad.edges.shape[0])}", flush=True)
        except Exception as exc:
            rejected += 1
            record = {
                "dataset": "fusion360",
                "part_id": part_id,
                "status": "error",
                "reason": f"{type(exc).__name__}: {exc}",
                "step_path": str(step_path),
                "seg_path": str(seg_path),
            }
            append_jsonl(manifest_path, record)
            append_error("Fusion360 preprocessing error", error_log=error_log, context=record)
            print(f"[fusion360] error part_id={part_id} reason={type(exc).__name__}: {exc}", flush=True)

    summary = {
        "dataset": "fusion360",
        "cache_dir": str(cache_dir),
        "scanned_parts": scanned,
        "written_parts": written,
        "rejected_or_error_parts": rejected,
        "requested_parts": "all",
    }
    write_json(summary_path, summary)
    print(f"[fusion360] done summary={summary}", flush=True)
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build Fusion360 PartGraph cache.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--cache-name", type=str, required=True)
    parser.add_argument("--error-log", type=Path, required=True)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    summary = build_fusion360_partgraph_cache(
        dataset_root=args.dataset_root,
        output_root=args.output_root,
        cache_name=args.cache_name,
        error_log=args.error_log,
    )
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
