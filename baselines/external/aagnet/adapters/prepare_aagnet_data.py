"""Prepare AAGNet-compatible external baseline directories from PartGraph cache."""

from __future__ import annotations

import argparse
from pathlib import Path

from data_protocol.io import write_json
from baselines.external.common.partgraph_export import (
    link_or_record_step_files,
    load_exported_parts,
    split_part_ids,
    write_export_manifest,
    write_labels_json_array,
    write_text_lines,
)


def prepare_aagnet_data(*, partgraph_cache_dir: str | Path, output_dir: str | Path, dataset: str) -> dict[str, str]:
    output = Path(output_dir)
    parts = load_exported_parts(partgraph_cache_dir)
    step_records = link_or_record_step_files(parts, output / "steps")
    labels_dir = output / "labels"
    aag_dir = output / "aag"
    graphs = []
    for part in parts:
        write_labels_json_array(labels_dir / f"{part.part_id}.json", part.labels)
        graphs.append([part.part_id, _aag_graph_record(part)])

    splits = split_part_ids(parts)
    write_text_lines(output / "train.txt", splits.get("train", []))
    write_text_lines(output / "val.txt", splits.get("val", []))
    write_text_lines(output / "test.txt", splits.get("test", []))
    write_json(aag_dir / "graphs.json", graphs)
    write_json(aag_dir / "attr_stat.json", _attr_stat())
    write_json(output / "step_manifest.json", {"steps": step_records})
    write_export_manifest(output / "manifest.json", dataset=dataset, parts=parts, extra={"format": "aagnet"})
    return {
        "converted_data_dir": str(output),
        "step_dir": str(output / "steps"),
        "aag_dir": str(aag_dir),
        "labels_dir": str(labels_dir),
    }


def _aag_graph_record(part) -> dict:
    edges = part.edges or [[0, 0]]
    edge_count = len(edges)
    return {
        "graph": {"edges": ([edge[0] for edge in edges], [edge[1] for edge in edges]), "num_nodes": part.num_faces},
        "graph_face_attr": [[1.0 if idx == (surf % 7) else 0.0 for idx in range(10)] for surf in part.surface_type],
        "graph_face_grid": [_face_grid(face_idx) for face_idx in range(part.num_faces)],
        "graph_edge_attr": [[1.0 if idx == (edge_type % 12) else 0.0 for idx in range(12)] for edge_type in (part.edge_type or [0] * edge_count)],
        "graph_edge_grid": [_edge_grid(edge_idx) for edge_idx in range(edge_count)],
    }


def _face_grid(seed: int) -> list:
    return [
        [[float(seed) for _ in range(10)] for _ in range(10)],
        [[float(u) for _ in range(10)] for u in range(10)],
        [[float(v) for v in range(10)] for _ in range(10)],
        [[0.0 for _ in range(10)] for _ in range(10)],
        [[0.0 for _ in range(10)] for _ in range(10)],
        [[1.0 for _ in range(10)] for _ in range(10)],
        [[1.0 for _ in range(10)] for _ in range(10)],
    ]


def _edge_grid(seed: int) -> list:
    return [
        [float(seed) for _ in range(10)],
        [float(u) for u in range(10)],
        [0.0 for _ in range(10)],
        [1.0 for _ in range(10)],
        [0.0 for _ in range(10)],
        [0.0 for _ in range(10)],
        [0.0 for _ in range(10)],
        [float(u) for u in range(10)],
        [0.0 for _ in range(10)],
        [0.0 for _ in range(10)],
        [1.0 for _ in range(10)],
        [0.0 for _ in range(10)],
    ]


def _attr_stat() -> dict:
    return {
        "mean_face_attr": [0.0] * 10,
        "std_face_attr": [1.0] * 10,
        "mean_edge_attr": [0.0] * 12,
        "std_edge_attr": [1.0] * 12,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partgraph-cache-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset", required=True)
    args = parser.parse_args()
    result = prepare_aagnet_data(
        partgraph_cache_dir=args.partgraph_cache_dir,
        output_dir=args.output_dir,
        dataset=args.dataset,
    )
    write_json(Path(args.output_dir) / "prepare_result.json", result)


if __name__ == "__main__":
    main()
