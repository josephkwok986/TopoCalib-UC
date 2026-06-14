#!/usr/bin/env python3
"""Build a HybridBrep preprocessed graph cache from STEP sources."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import torch


def _bootstrap_repo() -> None:
    root = Path(__file__).resolve().parents[2]
    for path in [root, root / "baselines" / "hybridbrep"]:
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Preprocess STEP files into HybridBrep graph .pt files.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--partgraph-cache-dir", type=Path, help="PartGraph cache whose source_paths.step entries should be processed.")
    source.add_argument("--step-path", type=Path, action="append", help="STEP file to process. Can be repeated.")
    parser.add_argument("--output-cache-dir", type=Path, required=True)
    parser.add_argument("--n-samples", type=int, required=True)
    parser.add_argument("--n-ref-samples", type=int, required=True)
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--sort-frac", type=float, required=True)
    parser.add_argument("--kernel-feats", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser


def main() -> int:
    _bootstrap_repo()
    args = build_arg_parser().parse_args()
    from frozen_representation.hybridbrep_backend import (
        import_hybridbrep_symbols,
        iter_partgraph_step_sources,
        iter_step_sources,
        save_json,
    )

    HPart, _ = import_hybridbrep_symbols()
    if args.partgraph_cache_dir is not None:
        sources = iter_partgraph_step_sources(args.partgraph_cache_dir)
    else:
        step_paths = args.step_path or []
        sources = iter_step_sources(step_paths)
    result = build_cache(args, HPart, sources)
    save_json(args.output_cache_dir / "manifest.json", result)
    print(result)
    return 0 if result["failed"] == 0 else 1


def build_cache(args: argparse.Namespace, HPart, sources: list[dict[str, Any]]) -> dict[str, Any]:
    parts_dir = args.output_cache_dir / "parts"
    rows = []
    failed = 0
    started = time.time()
    print(f"[hybridbrep_cache] start output={args.output_cache_dir} requested={len(sources)}", flush=True)
    for source in sources:
        step_path = Path(source["step_path"])
        out_path = parts_dir / f"{source['part_id']}.pt"
        try:
            print(f"[hybridbrep_cache] processing part_id={source['part_id']} step={step_path}", flush=True)
            data = HPart(
                str(step_path),
                n_samples=args.n_samples,
                n_ref_samples=args.n_ref_samples,
                normalize=args.normalize,
                sort_frac=args.sort_frac,
                kernel_feats=args.kernel_feats,
            ).data
            payload = {
                "part_id": str(source["part_id"]),
                "dataset": str(source.get("dataset", "")),
                "split": str(source.get("split", "")),
                "hybridbrep_data": data,
                "num_faces": int(getattr(data, "faces").shape[0]),
                "source_paths": dict(source.get("source_paths", {"step": str(step_path)})),
                "meta": {
                    "preprocess": {
                        "n_samples": args.n_samples,
                        "n_ref_samples": args.n_ref_samples,
                        "normalize": args.normalize,
                        "sort_frac": args.sort_frac,
                        "kernel_feats": args.kernel_feats,
                    },
                    "source_meta": dict(source.get("meta", {})),
                },
            }
            torch.save(payload, out_path)
            rows.append({"part_id": source["part_id"], "status": "ok", "path": str(out_path), "num_faces": payload["num_faces"]})
            print(f"[hybridbrep_cache] written part_id={source['part_id']} faces={payload['num_faces']} path={out_path}", flush=True)
        except Exception as exc:
            failed += 1
            rows.append({"part_id": source.get("part_id", step_path.stem), "status": "failed", "error_type": type(exc).__name__, "error": str(exc)})
            print(f"[hybridbrep_cache] failed part_id={source.get('part_id', step_path.stem)} reason={type(exc).__name__}: {exc}", flush=True)
            if not args.continue_on_error:
                break
    return {
        "cache_type": "hybridbrep_preprocessed",
        "output_cache_dir": str(args.output_cache_dir),
        "requested": len(sources),
        "processed": sum(1 for row in rows if row["status"] == "ok"),
        "failed": failed,
        "elapsed_sec": time.time() - started,
        "parts": rows,
    }


if __name__ == "__main__":
    raise SystemExit(main())
