"""Generate AAGNet external baseline command plans."""

from __future__ import annotations

import argparse
from typing import Any

from baselines.external.aagnet.adapters.prepare_aagnet_data import prepare_aagnet_data
from baselines.external.common.wrapper import add_common_args, run_wrapper


def build_commands(config: dict[str, Any]) -> list[str]:
    upstream = config["upstream_dir"]
    converted = config["converted_data_dir"]
    output_dir = config["raw_output_dir"]
    num_workers = int(config.get("num_workers", 8))
    return [
        f"cd {upstream}",
        f"python dataset/AAGExtractor.py --step_path {converted}/steps --output {converted}/aag --num_workers {num_workers}",
        " ".join(
            [
                "python -m baselines.external.aagnet.adapters.train_aagnet_from_config",
                f"--upstream-dir {upstream}",
                f"--dataset-dir {converted}",
                f"--output-dir {output_dir}",
                f"--epochs {int(config.get('epochs', 100))}",
                f"--batch-size {int(config.get('batch_size', 256))}",
                f"--num-classes {int(config['num_classes'])}",
            ]
        ),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    args = parser.parse_args()
    run_wrapper(args=args, method="aagnet_external", command_builder=build_commands, prepare=prepare_aagnet_data)


if __name__ == "__main__":
    main()
