#!/usr/bin/env python3
"""Train the HybridBrep SSRL encoder on a preprocessed graph cache."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


def _bootstrap_repo() -> None:
    root = Path(__file__).resolve().parents[2]
    for path in [root, root / "baselines" / "hybridbrep"]:
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train HybridBrep GeneralConvEncDec and save a checkpoint.")
    parser.add_argument("--preprocessed-cache-dir", type=Path, required=True)
    parser.add_argument("--output-ckpt", type=Path, required=True)
    parser.add_argument("--embedding-dim", type=int, required=True)
    parser.add_argument("--hidden-dim", type=int, required=True)
    parser.add_argument("--layers", type=int, required=True)
    parser.add_argument("--attn-heads", type=int, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--weight-decay", type=float, required=True)
    parser.add_argument("--device", required=True)
    return parser


def main() -> int:
    _bootstrap_repo()
    args = build_arg_parser().parse_args()
    result = train(args)
    print(result)
    return 0


def train(args: argparse.Namespace) -> dict[str, Any]:
    from frozen_representation.hybridbrep_backend import import_hybridbrep_symbols, load_hybridbrep_graph, move_to_device, sanitize_tensors

    try:
        from torch_geometric.loader import DataLoader
    except Exception as exc:
        raise RuntimeError("torch_geometric is required for SSRL training. Run this inside the HybridBrep CAD environment.") from exc

    _, GeneralConvEncDec = import_hybridbrep_symbols()
    device = torch.device(args.device)
    part_paths = sorted((args.preprocessed_cache_dir / "parts").glob("*.pt"))
    if not part_paths:
        raise ValueError(f"No preprocessed HybridBrep parts found in {args.preprocessed_cache_dir / 'parts'}")
    dataset = [sanitize_tensors(load_hybridbrep_graph(path)["hybridbrep_data"]) for path in part_paths]
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    print(
        f"[train_ssrl] start cache={args.preprocessed_cache_dir} parts={len(part_paths)} "
        f"epochs={args.epochs} batch_size={args.batch_size} device={device}",
        flush=True,
    )
    model = GeneralConvEncDec(
        emb_dim=args.embedding_dim,
        dec_dim=args.hidden_dim,
        dec_layers=args.layers,
        attn_heads=args.attn_heads,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history = []
    started = time.time()
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        steps = 0
        for batch in loader:
            batch = move_to_device(sanitize_tensors(batch), device)
            optimizer.zero_grad()
            loss = _reconstruction_loss(model, batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            epoch_loss += float(loss.detach().cpu())
            steps += 1
        avg_loss = epoch_loss / max(1, steps)
        history.append({"epoch": epoch, "loss": avg_loss, "steps": steps})
        print(f"[train_ssrl] epoch={epoch} loss={avg_loss:.6f} steps={steps}", flush=True)

    torch.save(
        {
            "model": "GeneralConvEncDec",
            "encoder": "hybridbrep_ssrl",
            "state_dict": model.state_dict(),
            "config": {
                "embedding_dim": args.embedding_dim,
                "hidden_dim": args.hidden_dim,
                "layers": args.layers,
                "attn_heads": args.attn_heads,
            },
            "train_args": vars(args),
            "history": history,
        },
        args.output_ckpt,
    )
    print(f"[train_ssrl] done checkpoint={args.output_ckpt} elapsed_sec={time.time() - started:.2f}", flush=True)
    return {
        "checkpoint": str(args.output_ckpt),
        "num_parts": len(part_paths),
        "epochs": args.epochs,
        "history": history,
        "elapsed_sec": time.time() - started,
    }


def _reconstruction_loss(model, batch) -> torch.Tensor:
    n_curves, n_curve_samples, _ = batch.curve_samples.shape
    edge_coords = torch.linspace(0.0, 1.0, n_curve_samples, device=batch.faces.device).repeat(n_curves).reshape((n_curves, -1))
    _, face_preds, _, edge_preds = model(batch, batch.surface_coords, edge_coords)
    face_target = batch.surface_samples
    edge_target = batch.curve_samples[:, :, :3].reshape((-1, 3))
    face_pred_xyz = face_preds[:, :3]
    face_pred_mask = face_preds[:, 3]
    face_target_xyz = face_target[:, :, :3].reshape((-1, 3))
    face_target_mask = face_target[:, :, -1].flatten()
    loss = F.mse_loss(face_pred_xyz, face_target_xyz) + F.mse_loss(face_pred_mask, face_target_mask)
    if edge_preds.numel() and edge_target.numel():
        loss = loss + F.mse_loss(edge_preds, edge_target)
    return loss


if __name__ == "__main__":
    raise SystemExit(main())
