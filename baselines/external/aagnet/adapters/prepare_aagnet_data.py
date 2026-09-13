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
    aag_dir.mkdir(parents=True, exist_ok=True)
    for part in parts:
        write_labels_json_array(labels_dir / f"{part.part_id}.json", part.labels)

    splits = split_part_ids(parts)
    write_text_lines(output / "train.txt", splits.get("train", []))
    write_text_lines(output / "val.txt", splits.get("val", []))
    write_text_lines(output / "test.txt", splits.get("test", []))
    write_json(output / "step_manifest.json", {"steps": step_records})
    write_export_manifest(output / "manifest.json", dataset=dataset, parts=parts, extra={"format": "aagnet"})
    return {
        "converted_data_dir": str(output),
        "step_dir": str(output / "steps"),
        "aag_dir": str(aag_dir),
        "labels_dir": str(labels_dir),
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
