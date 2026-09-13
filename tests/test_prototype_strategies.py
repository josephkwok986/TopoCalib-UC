from __future__ import annotations

import json
from pathlib import Path
import unittest

import numpy as np
import torch

from topocalib_uc.prototypes.matching import build_prototypes
from topocalib_uc.prototypes.sdbscan_backend import (
    SDBSCANConfig,
    SDBSCAN_DEFAULT_MIN_PTS,
    SDBSCAN_DEFAULT_N_PROJ,
    SDBSCAN_DEFAULT_N_THREADS,
    SDBSCAN_DEFAULT_TOP_K,
    SDBSCAN_DEFAULT_TOP_M,
    SDBSCAN_EPS_SEARCH_GRID,
    SDBSCAN_MIN_PTS_SEARCH_GRID,
    SDBSCAN_SELECTED_EPS,
    sdbscan_labels,
)
from topocalib_uc.prototypes.strategies import (
    PrototypeStrategyConfig,
    build_prototypes_with_strategy,
    leave_one_out_logits_with_strategy,
    medoid_prototype,
)
from topocalib_uc.train.run_frozen_proto import (
    _ensure_defaults,
    _prototype_config_from_args,
    build_arg_parser,
)


class _StubModule:
    labels: list[int] = []
    last_matrix: np.ndarray | None = None
    last_fit: tuple[float, int] | None = None
    last_params: tuple | None = None

    class sDbscan:
        def __init__(self, n_points: int, n_features: int) -> None:
            self.n_points = n_points
            self.n_features = n_features
            self.labels_: list[int] = []

        def set_params(self, *args) -> None:
            _StubModule.last_params = args

        def fit_sDbscan(self, matrix: np.ndarray, eps: float, min_pts: int) -> None:
            _StubModule.last_matrix = matrix
            _StubModule.last_fit = (eps, min_pts)
            labels = _StubModule.labels or [0] * self.n_points
            self.labels_ = labels[: self.n_points]


def _sdbscan_config() -> SDBSCANConfig:
    return SDBSCANConfig(
        eps=0.2,
        min_pts=5,
        n_proj=32,
        top_k=10,
        top_m=20,
        n_threads=2,
        random_seed=7,
    )


class PrototypeStrategiesTest(unittest.TestCase):
    def setUp(self) -> None:
        _StubModule.labels = []
        _StubModule.last_matrix = None
        _StubModule.last_fit = None
        _StubModule.last_params = None

    def test_mean_matches_legacy_implementation(self) -> None:
        tokens = torch.tensor([[1.0, 0.0], [0.8, 0.2], [0.0, 1.0], [0.2, 0.8]])
        labels = torch.tensor([0, 0, 1, 1])
        expected, expected_counts = build_prototypes(tokens, labels, 2)
        actual, actual_counts, info = build_prototypes_with_strategy(
            tokens,
            labels,
            2,
            PrototypeStrategyConfig(method="mean"),
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(actual_counts, expected_counts, rtol=0, atol=0)
        self.assertEqual(info["fallback_count"], 0)

    def test_table_b6_protocol_matches_code_constants(self) -> None:
        path = Path(__file__).resolve().parents[1] / "experiment_protocols" / "sdbscan_table_b6.json"
        protocol = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(tuple(protocol["search_grid"]["eps"]), SDBSCAN_EPS_SEARCH_GRID)
        self.assertEqual(tuple(protocol["search_grid"]["min_pts"]), SDBSCAN_MIN_PTS_SEARCH_GRID)
        self.assertEqual(protocol["selected"]["fusion360"]["eps"], SDBSCAN_SELECTED_EPS["fusion360"])
        self.assertEqual(protocol["selected"]["mfcadpp"]["eps"], SDBSCAN_SELECTED_EPS["mfcadpp"])
        self.assertEqual(protocol["selected"]["fusion360"]["min_pts"], SDBSCAN_DEFAULT_MIN_PTS)
        self.assertEqual(protocol["selected"]["mfcadpp"]["min_pts"], SDBSCAN_DEFAULT_MIN_PTS)
        fixed = protocol["fixed_approximation_parameters"]
        self.assertEqual(fixed["n_proj"], SDBSCAN_DEFAULT_N_PROJ)
        self.assertEqual(fixed["top_k"], SDBSCAN_DEFAULT_TOP_K)
        self.assertEqual(fixed["top_m"], SDBSCAN_DEFAULT_TOP_M)
        self.assertEqual(protocol["execution"]["n_threads"], SDBSCAN_DEFAULT_N_THREADS)

    def test_sdbscan_cli_uses_fixed_parameters_and_run_seed(self) -> None:
        args = build_arg_parser().parse_args(
            [
                "--cache-dir",
                "/data/cache",
                "--output",
                "/results/run.json",
                "--labeled-part-budget",
                "1000",
                "--seed",
                "2",
                "--prototype-method",
                "sdbscan",
                "--sdbscan-eps",
                "0.10",
            ]
        )
        _ensure_defaults(args)
        config = _prototype_config_from_args(args).sdbscan
        self.assertIsNotNone(config)
        assert config is not None
        self.assertEqual(config.eps, 0.10)
        self.assertEqual(config.min_pts, 5)
        self.assertEqual(config.n_proj, 1024)
        self.assertEqual(config.top_k, 10)
        self.assertEqual(config.top_m, 10)
        self.assertEqual(config.n_threads, 1)
        self.assertEqual(config.random_seed, 2)

    def test_medoid_is_an_original_support_and_ties_choose_first(self) -> None:
        tokens = torch.tensor([[1.0, 0.0], [1.0, 0.0], [-1.0, 0.0]])
        medoid = medoid_prototype(tokens, chunk_size=1)
        torch.testing.assert_close(medoid, tokens[0])
        self.assertTrue(any(torch.equal(medoid, row) for row in tokens))

    def test_sdbscan_input_contract_and_small_class_caps(self) -> None:
        tokens = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float64)
        _StubModule.labels = [0, -1]
        labels, effective = sdbscan_labels(tokens, _sdbscan_config(), backend_module=_StubModule)
        self.assertEqual(labels.tolist(), [0, -1])
        self.assertEqual(_StubModule.last_matrix.shape, (3, 2))  # type: ignore[union-attr]
        self.assertEqual(_StubModule.last_matrix.dtype, np.float32)  # type: ignore[union-attr]
        self.assertTrue(_StubModule.last_matrix.flags.c_contiguous)  # type: ignore[union-attr]
        self.assertEqual(effective["min_pts"], 2)
        self.assertEqual(effective["top_k"], 2)
        self.assertEqual(effective["top_m"], 2)
        self.assertEqual(_StubModule.last_params[3], "Cosine")  # type: ignore[index]

    def test_noise_exclusion_and_all_noise_fallback(self) -> None:
        tokens = torch.tensor([[1.0, 0.0], [0.8, 0.2], [-1.0, 0.0]])
        labels = torch.tensor([0, 0, 0])
        config = PrototypeStrategyConfig(method="sdbscan", sdbscan=_sdbscan_config())
        _StubModule.labels = [0, 0, -1]
        prototype, _, info = build_prototypes_with_strategy(
            tokens,
            labels,
            1,
            config,
            sdbscan_backend_module=_StubModule,
        )
        torch.testing.assert_close(prototype[0], tokens[:2].mean(dim=0))
        self.assertEqual(info["total_noise"], 1)
        self.assertEqual(info["fallback_count"], 0)

        _StubModule.labels = [-1, -1, -1]
        prototype, _, info = build_prototypes_with_strategy(
            tokens,
            labels,
            1,
            config,
            sdbscan_backend_module=_StubModule,
        )
        torch.testing.assert_close(prototype[0], tokens.mean(dim=0))
        self.assertEqual(info["fallback_count"], 1)

    def test_all_strategies_support_leave_one_out_with_stub_backend(self) -> None:
        tokens = torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]], requires_grad=True)
        labels = torch.tensor([0, 0, 1, 1])
        configs = [
            PrototypeStrategyConfig(method="mean"),
            PrototypeStrategyConfig(method="medoid", medoid_chunk_size=1),
            PrototypeStrategyConfig(method="sdbscan", sdbscan=_sdbscan_config()),
        ]
        for config in configs:
            _StubModule.labels = []
            logits = leave_one_out_logits_with_strategy(
                tokens,
                labels,
                num_classes=2,
                tau=0.2,
                config=config,
                sdbscan_backend_module=_StubModule,
            )
            self.assertEqual(tuple(logits.shape), (4, 2))
            self.assertTrue(torch.isfinite(logits).all())


if __name__ == "__main__":
    unittest.main()
