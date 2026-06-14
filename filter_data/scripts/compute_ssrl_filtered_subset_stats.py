#!/usr/bin/env python3
"""Compute SSRL-compatible filtered subset statistics.

The filter is applied at part level:
  * exactly one solid body
  * all face surface types are plane/cylinder/cone/sphere/torus
  * all edge curve types are line/circle/ellipse

Full Fusion360 command:
  python scripts/compute_ssrl_filtered_subset_stats.py \
    --dataset_name fusion360 \
    --dataset_root "/workspace/Gjj Local/data2/s2.0.1" \
    --step_dir "/workspace/Gjj Local/data2/s2.0.1/breps/step" \
    --label_dir "/workspace/Gjj Local/data2/s2.0.1/breps/seg" \
    --output_dir "/workspace/Gjj Local/data2/ssrl_filtered_stats/fusion360"

Full MFCAD++ command:
  python scripts/compute_ssrl_filtered_subset_stats.py \
    --dataset_name mfcadpp \
    --dataset_root "/workspace/Gjj Local/data2/MFCAD++_dataset" \
    --step_dir "/workspace/Gjj Local/data2/MFCAD++_dataset/step" \
    --label_dir "/workspace/Gjj Local/data2/MFCAD++_dataset/hierarchical_graphs" \
    --output_dir "/workspace/Gjj Local/data2/ssrl_filtered_stats/mfcadpp"

"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple

import h5py
import numpy as np
from OCC.Core.BRepAdaptor import BRepAdaptor_Curve, BRepAdaptor_Surface
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_SOLID
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopoDS import topods
from OCC.Core.GeomAbs import (
    GeomAbs_Circle,
    GeomAbs_Cone,
    GeomAbs_Cylinder,
    GeomAbs_Ellipse,
    GeomAbs_Line,
    GeomAbs_Plane,
    GeomAbs_Sphere,
    GeomAbs_Torus,
)


ALLOWED_SURFACE_TYPES = {
    GeomAbs_Plane,
    GeomAbs_Cylinder,
    GeomAbs_Cone,
    GeomAbs_Sphere,
    GeomAbs_Torus,
}

ALLOWED_CURVE_TYPES = {
    GeomAbs_Line,
    GeomAbs_Circle,
    GeomAbs_Ellipse,
}

STEP_EXTENSIONS = (".step", ".stp", ".STEP", ".STP")


@dataclass(frozen=True)
class PartLabels:
    part_id: str
    labels: Tuple[int, ...]
    step_path: Optional[Path]


@dataclass(frozen=True)
class GeometryCheck:
    passed: bool
    num_faces: int
    num_solids: int
    reason: str = ""


def normalize_dataset_name(name: str) -> str:
    value = name.strip().lower().replace("-", "").replace("_", "")
    if value in {"fusion360", "fusion", "f360"}:
        return "fusion360"
    if value in {"mfcadpp", "mfcad++", "mfcadplusplus"}:
        return "mfcadpp"
    raise ValueError(f"Unsupported dataset_name: {name!r}. Use fusion360 or mfcadpp.")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute SSRL-compatible filtered subset retention and class distribution stats."
    )
    parser.add_argument("--dataset_name", required=True, help="fusion360 or mfcadpp")
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--step_dir", required=True)
    parser.add_argument("--label_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--progress_interval",
        type=int,
        required=True,
        help="Print progress every N processed parts.",
    )
    return parser


def resolve_paths(args: argparse.Namespace) -> argparse.Namespace:
    dataset_name = normalize_dataset_name(args.dataset_name)
    args.dataset_name = dataset_name
    args.dataset_root = Path(args.dataset_root)
    args.step_dir = Path(args.step_dir)
    args.label_dir = Path(args.label_dir)
    args.output_dir = Path(args.output_dir)
    return args


def read_step_shape(step_path: Path):
    reader = STEPControl_Reader()
    status = reader.ReadFile(str(step_path))
    if status != IFSelect_RetDone:
        raise RuntimeError(f"STEP read failed with status {status}")
    reader.TransferRoots()
    return reader.OneShape()


def count_shape_items(shape, kind) -> int:
    count = 0
    explorer = TopExp_Explorer(shape, kind)
    while explorer.More():
        count += 1
        explorer.Next()
    return count


def check_geometry(step_path: Path) -> GeometryCheck:
    try:
        shape = read_step_shape(step_path)
        num_solids = count_shape_items(shape, TopAbs_SOLID)
        num_faces = count_shape_items(shape, TopAbs_FACE)

        if num_solids != 1:
            return GeometryCheck(False, num_faces, num_solids, f"num_solids={num_solids}")

        face_explorer = TopExp_Explorer(shape, TopAbs_FACE)
        while face_explorer.More():
            face = topods.Face(face_explorer.Current())
            surface_type = BRepAdaptor_Surface(face).GetType()
            if surface_type not in ALLOWED_SURFACE_TYPES:
                return GeometryCheck(False, num_faces, num_solids, f"surface_type={surface_type}")
            face_explorer.Next()

        edge_explorer = TopExp_Explorer(shape, TopAbs_EDGE)
        while edge_explorer.More():
            edge = topods.Edge(edge_explorer.Current())
            curve_type = BRepAdaptor_Curve(edge).GetType()
            if curve_type not in ALLOWED_CURVE_TYPES:
                return GeometryCheck(False, num_faces, num_solids, f"curve_type={curve_type}")
            edge_explorer.Next()

        return GeometryCheck(True, num_faces, num_solids)
    except Exception as exc:  # Keep whole-dataset stats moving past corrupt geometry.
        return GeometryCheck(False, 0, 0, f"geometry_error={type(exc).__name__}: {exc}")


def find_step_files(step_dir: Path) -> Dict[str, Path]:
    step_files: Dict[str, Path] = {}
    for path in step_dir.rglob("*"):
        if path.is_file() and path.suffix in STEP_EXTENSIONS:
            step_files.setdefault(path.stem, path)
    return step_files


def parse_seg_file(path: Path) -> Tuple[int, ...]:
    labels: List[int] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            value = line.strip()
            if not value:
                continue
            try:
                labels.append(int(float(value)))
            except ValueError as exc:
                raise ValueError(f"Invalid label in {path} line {line_no}: {value!r}") from exc
    return tuple(labels)


def iter_fusion360_parts(step_files: Dict[str, Path], label_dir: Path) -> Iterator[PartLabels]:
    for seg_path in sorted(label_dir.glob("*.seg")):
        labels = parse_seg_file(seg_path)
        yield PartLabels(seg_path.stem, labels, step_files.get(seg_path.stem))


def decode_cad_model(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.bytes_):
        return bytes(value).decode("utf-8")
    return str(value)


def labels_for_mfcad_group(group) -> Iterator[Tuple[str, Tuple[int, ...]]]:
    model_ids = [decode_cad_model(x) for x in group["CAD_model"][()]]
    labels = group["labels"][()]
    idx = group["idx"][()]

    # MFCAD++ idx stores inclusive end indices in the batched graph index space.
    # The labels dataset is local to this H5 group, so translate global/batch
    # indices back to local offsets before slicing.
    face_base = int(idx[-1, 0]) - len(labels) + 1
    prev_face_end = face_base - 1
    for model_id, row in zip(model_ids, idx):
        face_end = int(row[0])
        local_start = prev_face_end + 1 - face_base
        local_end = face_end + 1 - face_base
        part_labels = tuple(int(x) for x in labels[local_start:local_end])
        prev_face_end = face_end
        yield model_id, part_labels


def iter_mfcadpp_parts(step_files: Dict[str, Path], label_dir: Path) -> Iterator[PartLabels]:
    h5_paths = sorted(label_dir.glob("*.h5"))
    for h5_path in h5_paths:
        with h5py.File(h5_path, "r") as h5:
            for group_name in sorted(h5.keys(), key=lambda x: int(x) if x.isdigit() else x):
                group = h5[group_name]
                for model_id, labels in labels_for_mfcad_group(group):
                    yield PartLabels(model_id, labels, step_files.get(model_id))


def load_class_names(dataset_name: str, dataset_root: Path) -> Dict[int, str]:
    if dataset_name == "fusion360":
        path = dataset_root / "segment_names.json"
        if not path.exists():
            return {}
        with path.open("r", encoding="utf-8") as f:
            names = json.load(f)
        if isinstance(names, list):
            return {idx: str(name) for idx, name in enumerate(names)}
        if isinstance(names, dict):
            return {int(k): str(v) for k, v in names.items()}
        return {}

    if dataset_name == "mfcadpp":
        path = dataset_root / "feature_labels.txt"
        if not path.exists():
            return {}
        mapping: Dict[int, str] = {}
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                match = re.match(r"\s*(\d+)\s*-\s*(.+?)\s*$", line)
                if match:
                    mapping[int(match.group(1))] = match.group(2)
        return mapping

    return {}


def safe_ratio(numerator: int, denominator: int) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def format_duration(seconds: Optional[float]) -> str:
    if seconds is None:
        return "unknown"
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:d}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes:d}m{secs:02d}s"
    return f"{secs:d}s"


def progress_line(
    dataset_name: str,
    processed_step_paths: int,
    total_step_paths: int,
    filtered_num_parts: int,
    start_time: float,
) -> str:
    elapsed = time.monotonic() - start_time
    progress = safe_ratio(processed_step_paths, total_step_paths)
    rate = processed_step_paths / elapsed if elapsed > 0 and processed_step_paths > 0 else 0.0
    remaining_steps = max(total_step_paths - processed_step_paths, 0)
    eta = remaining_steps / rate if rate > 0 else None
    return (
        f"[{dataset_name}] processed_step_paths={processed_step_paths}/{total_step_paths} "
        f"({progress:.2%}) filtered_parts={filtered_num_parts} "
        f"elapsed={format_duration(elapsed)} eta={format_duration(eta)}"
    )


def compute_stats(
    dataset_name: str,
    dataset_root: Path,
    parts: Iterable[PartLabels],
    total_step_paths: int,
    output_dir: Path,
    progress_interval: int,
) -> None:
    before_counts: Counter[int] = Counter()
    after_counts: Counter[int] = Counter()

    original_num_parts = 0
    filtered_num_parts = 0
    original_num_faces = 0
    filtered_num_faces = 0
    missing_step_parts = 0
    geometry_rejected_parts = 0
    face_count_mismatch_parts = 0
    processed_step_paths = 0
    seen_step_paths: Set[Path] = set()
    last_reported_step_paths = 0
    start_time = time.monotonic()

    for part in parts:
        original_num_parts += 1
        label_count = len(part.labels)
        original_num_faces += label_count
        before_counts.update(part.labels)

        if part.step_path is None:
            missing_step_parts += 1
            passed = False
        else:
            geometry = check_geometry(part.step_path)
            if part.step_path not in seen_step_paths:
                seen_step_paths.add(part.step_path)
                processed_step_paths += 1
            passed = geometry.passed
            if not passed:
                geometry_rejected_parts += 1
            if geometry.num_faces and geometry.num_faces != label_count:
                face_count_mismatch_parts += 1

        if passed:
            filtered_num_parts += 1
            filtered_num_faces += label_count
            after_counts.update(part.labels)

        if (
            progress_interval > 0
            and processed_step_paths > 0
            and processed_step_paths != last_reported_step_paths
            and processed_step_paths % progress_interval == 0
        ):
            print(
                progress_line(dataset_name, processed_step_paths, total_step_paths, filtered_num_parts, start_time),
                flush=True,
            )
            last_reported_step_paths = processed_step_paths

    summary = {
        "dataset_name": dataset_name,
        "original_num_parts": original_num_parts,
        "filtered_num_parts": filtered_num_parts,
        "original_num_faces": original_num_faces,
        "filtered_num_faces": filtered_num_faces,
        "part_retention": safe_ratio(filtered_num_parts, original_num_parts),
        "face_retention": safe_ratio(filtered_num_faces, original_num_faces),
        "missing_step_parts": missing_step_parts,
        "geometry_rejected_parts": geometry_rejected_parts,
        "face_count_mismatch_parts": face_count_mismatch_parts,
        "total_step_paths": total_step_paths,
        "processed_step_paths": processed_step_paths,
    }

    validate_stats(summary, before_counts, after_counts)
    write_outputs(output_dir, summary, before_counts, after_counts, load_class_names(dataset_name, dataset_root))
    print(progress_line(dataset_name, processed_step_paths, total_step_paths, filtered_num_parts, start_time), flush=True)
    print(
        f"[{dataset_name}] done original_parts={original_num_parts} "
        f"filtered_parts={filtered_num_parts} output_dir={output_dir}",
        flush=True,
    )


def validate_stats(summary: Dict[str, object], before: Counter[int], after: Counter[int]) -> None:
    checks = [
        (sum(before.values()) == summary["original_num_faces"], "before_face_count sum mismatch"),
        (sum(after.values()) == summary["filtered_num_faces"], "after_face_count sum mismatch"),
        (summary["filtered_num_parts"] <= summary["original_num_parts"], "part count ordering failed"),
        (summary["filtered_num_faces"] <= summary["original_num_faces"], "face count ordering failed"),
    ]
    failed = [message for passed, message in checks if not passed]
    if failed:
        raise AssertionError("; ".join(failed))


def write_outputs(
    output_dir: Path,
    summary: Dict[str, object],
    before_counts: Counter[int],
    after_counts: Counter[int],
    class_names: Dict[int, str],
) -> None:
    summary_path = output_dir / "retention_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
        f.write("\n")

    csv_path = output_dir / "class_distribution_before_after.csv"
    all_class_ids = sorted(set(before_counts) | set(after_counts))
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "class_id",
                "class_name",
                "before_face_count",
                "after_face_count",
                "retention_ratio",
            ],
        )
        writer.writeheader()
        for class_id in all_class_ids:
            before = before_counts[class_id]
            after = after_counts[class_id]
            writer.writerow(
                {
                    "class_id": class_id,
                    "class_name": class_names.get(class_id, f"class_{class_id}"),
                    "before_face_count": before,
                    "after_face_count": after,
                    "retention_ratio": safe_ratio(after, before),
                }
            )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = resolve_paths(parser.parse_args(argv))

    step_files = find_step_files(args.step_dir)
    print(f"[{args.dataset_name}] total_step_paths={len(step_files)} step_dir={args.step_dir}", flush=True)

    if args.dataset_name == "fusion360":
        parts = iter_fusion360_parts(step_files, args.label_dir)
    elif args.dataset_name == "mfcadpp":
        parts = iter_mfcadpp_parts(step_files, args.label_dir)
    else:
        raise AssertionError(args.dataset_name)

    compute_stats(
        dataset_name=args.dataset_name,
        dataset_root=args.dataset_root,
        parts=parts,
        total_step_paths=len(step_files),
        output_dir=args.output_dir,
        progress_interval=args.progress_interval,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
