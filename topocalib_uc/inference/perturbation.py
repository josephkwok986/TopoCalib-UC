"""Deterministic undirected adjacency deletion for downstream inference."""
import hashlib
import torch
from topocalib_uc.evidence.readout import FaceBatchMeta


def delete_adjacencies(meta, probability, seed, part_id):
    if not 0 <= probability <= 1:
        raise ValueError('Deletion probability must be in [0, 1]')
    if probability == 0 or not meta.edges.numel():
        return meta
    edges = meta.edges.detach().cpu().reshape(-1, 2)
    pairs = sorted({tuple(sorted(edge)) for edge in edges.tolist()})
    digest = hashlib.sha256(f'{seed}:{part_id}'.encode()).digest()
    generator = torch.Generator().manual_seed(int.from_bytes(digest[:8], 'little') % (2**63 - 1))
    draws = torch.rand(len(pairs), generator=generator).tolist()
    keep = {pair: draw >= probability for pair, draw in zip(pairs, draws)}
    mask = torch.tensor([keep[tuple(sorted(edge))] for edge in edges.tolist()], device=meta.edges.device, dtype=torch.bool)
    return FaceBatchMeta(meta.part_index, meta.edges[mask], meta.edge_type[mask])
