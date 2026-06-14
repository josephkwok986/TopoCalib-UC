"""Generate UV-Net external baseline command plans."""

from __future__ import annotations

import argparse
from typing import Any

from baselines.external.common.wrapper import add_common_args, run_wrapper
from baselines.external.uvnet.adapters.prepare_uvnet_data import prepare_uvnet_data


def build_commands(config: dict[str, Any]) -> list[str]:
    upstream = config["upstream_dir"]
    converted = config["converted_data_dir"]
    experiment_name = config.get("experiment_name", "fusion360_gallery")
    max_epochs = int(config.get("max_epochs", 100))
    batch_size = int(config.get("batch_size", 64))
    num_workers = int(config.get("num_workers", 8))
    return [
        f"cd {upstream}",
        f"python -m process.solid_to_graph {converted}/steps {converted}/graph --num_processes {num_workers}",
        " ".join(
            [
                "python segmentation.py train",
                "--dataset fusiongallery",
                f"--dataset_path {converted}",
                f"--max_epochs {max_epochs}",
                f"--batch_size {batch_size}",
                f"--num_workers {num_workers}",
                f"--experiment_name {experiment_name}",
            ]
        ),
        " ".join(
            [
                "python segmentation.py test",
                "--dataset fusiongallery",
                f"--dataset_path {converted}",
                f"--checkpoint {config['raw_output_dir']}/segmentation/best.ckpt",
                f"--batch_size {batch_size}",
                f"--num_workers {num_workers}",
                f"--experiment_name {experiment_name}_test",
            ]
        ),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    args = parser.parse_args()
    run_wrapper(args=args, method="uvnet_external", command_builder=build_commands, prepare=prepare_uvnet_data)


if __name__ == "__main__":
    main()
