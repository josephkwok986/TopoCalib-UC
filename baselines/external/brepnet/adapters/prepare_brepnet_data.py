"""Prepare BRepNet-compatible external baseline directories from PartGraph cache."""

from __future__ import annotations

import argparse
from pathlib import Path

from data_protocol.io import write_json
from baselines.external.common.partgraph_export import (
    class_names,
    link_or_record_step_files,
    load_exported_parts,
    split_part_ids,
    write_export_manifest,
    write_labels_txt,
)


def prepare_brepnet_data(*, partgraph_cache_dir: str | Path, output_dir: str | Path, dataset: str) -> dict[str, str]:
    output = Path(output_dir)
    parts = load_exported_parts(partgraph_cache_dir)
    step_records = link_or_record_step_files(parts, output / "steps")
    seg_dir = output / "seg"
    for part in parts:
        write_labels_txt(seg_dir / f"{part.part_id}.seg", part.labels)

    splits = split_part_ids(parts)
    write_json(output / "train_test.json", {"train": splits.get("train", []) + splits.get("val", []), "test": splits.get("test", [])})
    write_json(output / "fixed_split.json", {"train": splits.get("train", []), "validation": splits.get("val", []), "test": splits.get("test", [])})
    write_json(output / "segment_names.json", class_names(parts))
    write_json(output / "step_manifest.json", {"steps": step_records})
    write_export_manifest(output / "manifest.json", dataset=dataset, parts=parts, extra={"format": "brepnet"})
    return {
        "converted_data_dir": str(output),
        "step_dir": str(output / "steps"),
        "label_dir": str(seg_dir),
        "train_test": str(output / "train_test.json"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partgraph-cache-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset", required=True)
    args = parser.parse_args()
    result = prepare_brepnet_data(
        partgraph_cache_dir=args.partgraph_cache_dir,
        output_dir=args.output_dir,
        dataset=args.dataset,
    )
    write_json(Path(args.output_dir) / "prepare_result.json", result)


if __name__ == "__main__":
    main()
