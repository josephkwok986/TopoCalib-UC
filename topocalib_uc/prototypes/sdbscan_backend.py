"""Adapter between PyTorch support tokens and the upstream sDBSCAN extension."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import importlib
from typing import Any

import numpy as np
import torch


SDBSCAN_EPS_SEARCH_GRID = (0.025, 0.05, 0.075, 0.10, 0.125, 0.15, 0.175, 0.20)
SDBSCAN_MIN_PTS_SEARCH_GRID = (3, 5, 10)
SDBSCAN_SELECTED_EPS = {"fusion360": 0.15, "mfcadpp": 0.10}
SDBSCAN_DEFAULT_MIN_PTS = 5
SDBSCAN_DEFAULT_N_PROJ = 1024
SDBSCAN_DEFAULT_TOP_K = 10
SDBSCAN_DEFAULT_TOP_M = 10
SDBSCAN_DEFAULT_N_THREADS = 1


@dataclass(frozen=True)
class SDBSCANConfig:
    eps: float
    min_pts: int
    n_proj: int
    top_k: int
    top_m: int
    n_threads: int
    random_seed: int

    def as_dict(self) -> dict[str, int | float]:
        return asdict(self)


def sdbscan_labels(
    tokens: torch.Tensor,
    config: SDBSCANConfig,
    *,
    backend_module: Any | None = None,
) -> tuple[np.ndarray, dict[str, int | float]]:
    """Cluster one class and return one label per support in original order."""

    if tokens.ndim != 2 or tokens.shape[0] == 0:
        raise ValueError("sDBSCAN tokens must have shape [num_supports, hidden_dim] with num_supports > 0")
    _validate_config(config)
    num_supports, hidden_dim = int(tokens.shape[0]), int(tokens.shape[1])
    effective = {
        "eps": float(config.eps),
        "min_pts": min(config.min_pts, num_supports),
        "n_proj": int(config.n_proj),
        "top_k": min(config.top_k, num_supports),
        "top_m": min(config.top_m, num_supports),
        "n_threads": int(config.n_threads),
        "random_seed": int(config.random_seed),
    }
    matrix = np.ascontiguousarray(
        tokens.detach().cpu().to(torch.float32).numpy().T,
        dtype=np.float32,
    )
    if matrix.shape != (hidden_dim, num_supports) or matrix.dtype != np.float32 or not matrix.flags.c_contiguous:
        raise AssertionError("sDBSCAN input conversion failed")
    module = backend_module if backend_module is not None else _import_backend()
    estimator = module.sDbscan(num_supports, hidden_dim)
    estimator.set_params(
        effective["n_proj"],
        effective["top_k"],
        effective["top_m"],
        "Cosine",
        effective["n_proj"],
        1.0,
        0.4,
        0,
        0.01,
        False,
        "",
        effective["n_threads"],
        effective["random_seed"],
    )
    estimator.fit_sDbscan(matrix, effective["eps"], effective["min_pts"])
    labels = np.asarray(estimator.labels_, dtype=np.int64).reshape(-1)
    if labels.shape != (num_supports,):
        raise RuntimeError(
            f"sDBSCAN returned labels with shape {labels.shape}; expected ({num_supports},)"
        )
    return labels, effective


def _validate_config(config: SDBSCANConfig) -> None:
    if not 0 < config.eps <= 2:
        raise ValueError("sDBSCAN cosine eps must be in (0, 2]")
    for name in ("min_pts", "n_proj", "top_k", "top_m", "n_threads"):
        if int(getattr(config, name)) <= 0:
            raise ValueError(f"sDBSCAN {name} must be positive")
    if config.random_seed < 0:
        raise ValueError("sDBSCAN random_seed must be non-negative and explicit")


def _import_backend() -> Any:
    try:
        return importlib.import_module("sDbscan")
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "Could not import the sDbscan extension. Build the pinned upstream source as documented, "
            "then rerun with the same sDBSCAN parameters."
        ) from exc
