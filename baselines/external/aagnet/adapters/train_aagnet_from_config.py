"""Parameterized AAGNet training entry for the external baseline workflow."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-dir", required=True)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--num-classes", type=int, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--weight-decay", type=float, required=True)
    parser.add_argument("--num-threads", type=int, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    upstream = Path(args.upstream_dir).resolve()
    dataset_dir = Path(args.dataset_dir).resolve()
    output_root = Path(args.output_dir).resolve()
    sys.path.insert(0, str(upstream))
    _train(args=args, dataset_dir=dataset_dir, output_root=output_root)


def _train(*, args: argparse.Namespace, dataset_dir: Path, output_root: Path) -> None:
    import numpy as np
    import torch
    from torch import nn
    from torch_ema import ExponentialMovingAverage
    from torchmetrics.classification import MulticlassAccuracy, MulticlassJaccardIndex
    from tqdm import tqdm

    from dataloader.mfcad2 import MFCAD2Dataset
    from models.segmentors import AAGNetSegmentor
    from utils.misc import init_logger, print_num_params, seed_torch

    os.environ.setdefault("WANDB_MODE", "offline")
    torch.set_float32_matmul_precision("high")
    seed_torch(args.seed)

    device = args.device
    n_classes = int(args.num_classes)
    save_path = output_root / str(args.run_name)
    logger = init_logger(str(save_path / "log.txt"))

    model = AAGNetSegmentor(
        num_classes=n_classes,
        arch="AAGNetGraphEncoder",
        edge_attr_dim=12,
        node_attr_dim=10,
        edge_attr_emb=64,
        node_attr_emb=64,
        edge_grid_dim=0,
        node_grid_dim=7,
        edge_grid_emb=0,
        node_grid_emb=64,
        num_layers=3,
        delta=2,
        mlp_ratio=2,
        drop=0.25,
        drop_path=0.25,
        head_hidden_dim=64,
        conv_on_edge=False,
        use_uv_gird=True,
        use_edge_attr=True,
        use_face_attr=True,
    ).to(device)
    logger.info(f"total_params: {print_num_params(model)}")

    train_dataset = MFCAD2Dataset(
        root_dir=str(dataset_dir),
        split="train",
        center_and_scale=False,
        normalize=True,
        random_rotate=False,
        num_threads=args.num_threads,
    )
    graphs = train_dataset.graphs()
    val_dataset = MFCAD2Dataset(
        root_dir=str(dataset_dir),
        graphs=graphs,
        split="val",
        center_and_scale=False,
        normalize=True,
        num_threads=args.num_threads,
    )
    test_dataset = MFCAD2Dataset(
        root_dir=str(dataset_dir),
        graphs=graphs,
        split="test",
        center_and_scale=False,
        normalize=True,
        random_rotate=False,
        num_threads=args.num_threads,
    )

    train_loader = train_dataset.get_dataloader(batch_size=args.batch_size, pin_memory=True)
    val_loader = val_dataset.get_dataloader(batch_size=args.batch_size, shuffle=False, drop_last=False, pin_memory=True)
    test_loader = test_dataset.get_dataloader(batch_size=args.batch_size, shuffle=False, drop_last=False, pin_memory=True)

    seg_loss = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=0)
    iters = max(1, len(train_loader))
    ema = ExponentialMovingAverage(model.parameters(), decay=(1.0 / 2.0) ** (1 / iters))

    best_iou = -1.0
    for epoch in range(args.epochs):
        logger.info(f"------------- epoch {epoch} -------------")
        train_loss, train_acc, train_iou = _run_epoch(
            model=model,
            loader=train_loader,
            device=device,
            n_classes=n_classes,
            loss_fn=seg_loss,
            opt=opt,
            ema=ema,
            train=True,
            progress=True,
        )
        scheduler.step()
        logger.info(f"train_loss: {train_loss}, train_seg_acc: {train_acc}, train_seg_iou: {train_iou}")

        with torch.no_grad(), ema.average_parameters():
            val_loss, val_acc, val_iou = _run_epoch(
                model=model,
                loader=val_loader,
                device=device,
                n_classes=n_classes,
                loss_fn=seg_loss,
                opt=None,
                ema=None,
                train=False,
                progress=True,
            )
        logger.info(f"val_loss: {val_loss}, val_seg_acc: {val_acc}, val_seg_iou: {val_iou}")
        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(model.state_dict(), save_path / f"weight_{epoch}-epoch.pth")

    with torch.no_grad():
        test_loss, test_acc, test_iou = _run_epoch(
            model=model,
            loader=test_loader,
            device=device,
            n_classes=n_classes,
            loss_fn=seg_loss,
            opt=None,
            ema=None,
            train=False,
            progress=True,
        )
    logger.info(f"test_loss: {test_loss}, test_seg_acc: {test_acc}, test_seg_iou: {test_iou}")


def _run_epoch(*, model, loader, device, n_classes, loss_fn, opt, ema, train: bool, progress: bool):
    import numpy as np
    from torchmetrics.classification import MulticlassAccuracy, MulticlassJaccardIndex
    from tqdm import tqdm

    if train:
        model.train()
    else:
        model.eval()
    acc = MulticlassAccuracy(num_classes=n_classes).to(device)
    iou = MulticlassJaccardIndex(num_classes=n_classes).to(device)
    losses = []
    iterator = tqdm(loader) if progress else loader
    for data in iterator:
        graphs = data["graph"].to(device, non_blocking=True)
        labels = graphs.ndata["y"]
        if train:
            opt.zero_grad()
        pred = model(graphs)
        loss = loss_fn(pred, labels)
        losses.append(loss.item())
        if train:
            loss.backward()
            opt.step()
            ema.update()
        acc.update(pred, labels)
        iou.update(pred, labels)
    return float(np.mean(losses).item()) if losses else 0.0, float(acc.compute().item()), float(iou.compute().item())


if __name__ == "__main__":
    main()
