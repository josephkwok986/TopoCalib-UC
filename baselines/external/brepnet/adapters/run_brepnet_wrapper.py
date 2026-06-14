"""Generate BRepNet external baseline command plans."""

from __future__ import annotations

import argparse
from typing import Any

from baselines.external.brepnet.adapters.prepare_brepnet_data import prepare_brepnet_data
from baselines.external.common.wrapper import add_common_args, run_wrapper


def build_commands(config: dict[str, Any]) -> list[str]:
    upstream = config["upstream_dir"]
    converted = config["converted_data_dir"]
    processed = f"{converted}/processed"
    dataset_file = f"{converted}/brepnet_dataset.json"
    log_dir = config["raw_output_dir"]
    feature_list = config.get("feature_list", f"{upstream}/feature_lists/all.json")
    kernel = config.get("kernel", f"{upstream}/kernels/winged_edge.json")
    max_epochs = int(config.get("max_epochs", 200))
    batch_size = int(config.get("batch_size", 64))
    num_workers = int(config.get("num_workers", 5))
    num_classes = int(config["num_classes"])
    validation_split = float(config.get("validation_split", 0.2))
    return [
        f"cd {upstream}",
        " ".join(
            [
                "python -m pipeline.extract_brepnet_data_from_step",
                f"--step_path {converted}/steps",
                f"--output {processed}",
                f"--feature_list {feature_list}",
                f"--seg_dir {converted}/seg",
                f"--num_workers {num_workers}",
            ]
        ),
        " ".join(
            [
                "python -m pipeline.build_dataset_file",
                f"--npz_folder {processed}",
                f"--train_test {converted}/train_test.json",
                f"--validation_split {validation_split}",
                f"--dataset_file {dataset_file}",
            ]
        ),
        " ".join(
            [
                "python -m train.train",
                f"--dataset_file {dataset_file}",
                f"--dataset_dir {processed}",
                f"--label_dir {converted}/seg",
                f"--input_features {feature_list}",
                f"--kernel {kernel}",
                f"--num_classes {num_classes}",
                f"--max_epochs {max_epochs}",
                f"--batch_size {batch_size}",
                f"--num_workers {num_workers}",
                f"--log_dir {log_dir}",
            ]
        ),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    args = parser.parse_args()
    run_wrapper(args=args, method="brepnet_external", command_builder=build_commands, prepare=prepare_brepnet_data)


if __name__ == "__main__":
    main()
