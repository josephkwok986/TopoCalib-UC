"""Batch complete CAD parts without mixing their topology or context."""
from __future__ import annotations

import argparse
import random
from typing import Sequence, TypeVar

import torch

from topocalib_uc.evidence.readout import FaceBatchMeta
from topocalib_uc.train.partgraph_dataset import PartRecord

T = TypeVar('T')


def positive_part_count(value: str) -> int:
    try:
        result = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError('Batch part count must be a positive integer') from exc
    if result <= 0:
        raise argparse.ArgumentTypeError('Batch part count must be a positive integer')
    return result


def validate_part_count(batch_part_count: int) -> None:
    if isinstance(batch_part_count, bool) or not isinstance(batch_part_count, int) or batch_part_count <= 0:
        raise ValueError('batch_part_count must be a positive integer')


def part_batches(items: Sequence[T], batch_part_count: int, *, shuffle_seed: int | None = None):
    validate_part_count(batch_part_count)
    ordered = list(items)
    if shuffle_seed is not None:
        random.Random(shuffle_seed).shuffle(ordered)
    for start in range(0, len(ordered), batch_part_count):
        yield ordered[start:start + batch_part_count]


def collate_parts(
    records: Sequence[PartRecord],
    z_by_part: dict[str, torch.Tensor],
    device: torch.device | str = 'cpu',
) -> dict:
    if not records:
        raise ValueError('Cannot collate an empty part batch')
    edge_chunks, relation_chunks, part_index = [], [], []
    slices = []
    offset = 0
    for index, record in enumerate(records):
        end = offset + record.num_faces
        slices.append(slice(offset, end))
        part_index.append(torch.full((record.num_faces,), index, dtype=torch.long, device=device))
        if record.edges.numel():
            edge_chunks.append(record.edges.to(device) + offset)
            relation_chunks.append(record.edge_type.to(device))
        offset = end
    edges = torch.cat(edge_chunks) if edge_chunks else torch.empty((0, 2), dtype=torch.long, device=device)
    relations = torch.cat(relation_chunks) if relation_chunks else torch.empty(0, dtype=torch.long, device=device)
    return {
        'z': torch.cat([z_by_part[r.part_id].to(device) for r in records]),
        'surface_type': torch.cat([r.surface_type.to(device) for r in records]),
        'y': torch.cat([r.y.to(device) for r in records]),
        'meta': FaceBatchMeta(torch.cat(part_index), edges, relations),
        'slices': slices,
    }
