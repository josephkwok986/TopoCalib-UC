#!/usr/bin/env python3
"""Export face-level SSRL embeddings from a HybridBrep preprocessed cache."""

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
    parser = argparse.ArgumentParser(description="Export HybridBrep SSRL face embeddings into a frozen embedding cache.")
    parser.add_argument("--preprocessed-cache-dir", type=Path, required=True)
    parser.add_argument("--output-cache-dir", type=Path, required=True)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--embedding-dim", type=int, required=True)
    parser.add_argument("--hidden-dim", type=int, required=True)
    parser.add_argument("--layers", type=int, required=True)
    parser.add_argument("--attn-heads", type=int, required=True)
    parser.add_argument("--device", required=True)
    return parser


def main() -> int:
    _bootstrap_repo()
    args = build_arg_parser().parse_args()
    from frozen_representation.hybridbrep_backend import save_json

    result = export(args)
    save_json(args.output_cache_dir / "manifest.json", result)
    print(result)
    return 0


@torch.no_grad()
def export(args: argparse.Namespace) -> dict[str, Any]:
    from frozen_representation.cache import save_embedding_record
    from frozen_representation.hybridbrep_backend import load_hybridbrep_graph, load_ssrl_model, move_to_device, sanitize_tensors

    device = torch.device(args.device)
    model, model_config = load_ssrl_model(
        args.ckpt,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        layers=args.layers,
        attn_heads=args.attn_heads,
        device=device,
    )
    model.eval()
    part_paths = sorted((args.preprocessed_cache_dir / "parts").glob("*.pt"))
    if not part_paths:
        raise ValueError(f"No preprocessed HybridBrep parts found in {args.preprocessed_cache_dir / 'parts'}")
    parts_dir = args.output_cache_dir / "parts"
    rows = []
    started = time.time()
    print(
        f"[export_ssrl] start preprocessed={args.preprocessed_cache_dir} checkpoint={args.ckpt} "
        f"parts={len(part_paths)} output={args.output_cache_dir}",
        flush=True,
    )
    for path in part_paths:
        payload = load_hybridbrep_graph(path)
        data = move_to_device(sanitize_tensors(payload["hybridbrep_data"]), device)
        z = model.encode_faces(data).detach().cpu().float()
        part_id = str(payload["part_id"])
        out_path = parts_dir / f"{part_id}.pt"
        save_embedding_record(
            out_path,
            {
                "part_id": part_id,
                "encoder": "hybridbrep_ssrl",
                "checkpoint": str(args.ckpt) if args.ckpt is not None else None,
                "embedding_dim": int(z.shape[1]),
                "z": z,
                "source_paths": dict(payload.get("source_paths", {})),
                "meta": {
                    "model_config": model_config,
                    "preprocessed_path": str(path),
                    "preprocessed_meta": dict(payload.get("meta", {})),
                },
            },
        )
        rows.append({"part_id": part_id, "path": str(out_path), "num_faces": int(z.shape[0]), "embedding_dim": int(z.shape[1])})
        print(f"[export_ssrl] written part_id={part_id} faces={int(z.shape[0])} dim={int(z.shape[1])} path={out_path}", flush=True)
    result = {
        "cache_type": "frozen_face_embeddings",
        "encoder": "hybridbrep_ssrl",
        "checkpoint": str(args.ckpt) if args.ckpt is not None else None,
        "output_cache_dir": str(args.output_cache_dir),
        "num_parts": len(rows),
        "embedding_dim": rows[0]["embedding_dim"] if rows else None,
        "elapsed_sec": time.time() - started,
        "parts": rows,
    }
    print(f"[export_ssrl] done num_parts={len(rows)} elapsed_sec={result['elapsed_sec']:.2f}", flush=True)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
