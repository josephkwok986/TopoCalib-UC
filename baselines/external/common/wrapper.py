"""Command-plan helpers for external baseline wrappers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

from data_protocol.io import write_json


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def write_command_plan(
    *,
    config: dict[str, Any],
    method: str,
    commands: list[str],
    prepare_result: dict[str, str] | None,
) -> dict[str, Any]:
    results_dir = Path(config["results_dir"])
    output_path = results_dir / config.get("command_plan_name", f"{method}_commands.json")
    payload = {
        "method": method,
        "dataset": config["dataset"],
        "num_classes": int(config["num_classes"]),
        "partgraph_cache_dir": config["partgraph_cache_dir"],
        "converted_data_dir": config["converted_data_dir"],
        "raw_output_dir": config["raw_output_dir"],
        "results_dir": config["results_dir"],
        "prepare_result": prepare_result or {},
        "commands": commands,
        "source": {"config": config},
    }
    write_json(output_path, payload)
    return {"output": str(output_path), "commands": commands, "prepare_result": prepare_result or {}}


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True)
    parser.add_argument("--prepare-data", action="store_true")


def run_wrapper(
    *,
    args: argparse.Namespace,
    method: str,
    command_builder: Callable[[dict[str, Any]], list[str]],
    prepare: Callable[..., dict[str, str]],
) -> dict[str, Any]:
    config = load_config(args.config)
    prepare_result = None
    if args.prepare_data:
        prepare_result = prepare(
            partgraph_cache_dir=config["partgraph_cache_dir"],
            output_dir=config["converted_data_dir"],
            dataset=config["dataset"],
        )
    commands = command_builder(config)
    return write_command_plan(config=config, method=method, commands=commands, prepare_result=prepare_result)
