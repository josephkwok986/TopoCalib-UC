from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import torch

from topocalib_uc.calibration.margin import (
    apply_margin_calibration,
    apply_pairwise_logit_update,
    compute_margin_calibration,
)
from topocalib_uc.candidate_pair.pairs import inference_candidate_pairs
from topocalib_uc.evidence.readout import FaceBatchMeta
from topocalib_uc.inference.records import build_face_prediction_records
from topocalib_uc.prototypes.matching import ClassIndex
from topocalib_uc.train.partgraph_dataset import PartRecord
from topocalib_uc.train.splitting import (
    PartSplit,
    load_labeled_parts_manifest,
    load_split_manifest,
    write_labeled_parts_manifest,
    write_split_manifest,
)


class _Cache:
    def __init__(self) -> None:
        self.records = [
            _CacheRecord("train_a", "unit_fixture", {0, 1}),
            _CacheRecord("train_b", "unit_fixture", {0, 1}),
            _CacheRecord("val_a", "unit_fixture", {0}),
            _CacheRecord("test_a", "unit_fixture", {0, 1}),
        ]
        self.part_ids = [item.part_id for item in self.records]
        self.classes = [0, 1]
        self._by_id = {item.part_id: item for item in self.records}

    def by_part_id(self, part_id: str) -> "_CacheRecord":
        return self._by_id[part_id]

    def labels_for_parts(self, part_ids: set[str]) -> set[int]:
        return {label for part_id in part_ids for label in self.by_part_id(part_id).class_set}


class _CacheRecord:
    def __init__(self, part_id: str, dataset: str, class_set: set[int]) -> None:
        self.part_id = part_id
        self.dataset = dataset
        self.class_set = class_set


class ProtocolManifestTest(unittest.TestCase):
    def test_split_and_labeled_parts_round_trip(self) -> None:
        cache = _Cache()
        split = PartSplit(train=["train_a", "train_b"], val=["val_a"], test=["test_a"])
        with tempfile.TemporaryDirectory() as tmp:
            split_path = Path(tmp) / "split.json"
            labeled_path = Path(tmp) / "labeled.json"
            write_split_manifest(split_path, cache, split)  # type: ignore[arg-type]
            write_labeled_parts_manifest(labeled_path, cache, split, ["train_a"])  # type: ignore[arg-type]
            self.assertEqual(load_split_manifest(split_path, cache), split)  # type: ignore[arg-type]
            self.assertEqual(
                load_labeled_parts_manifest(labeled_path, cache, split, expected_budget=1),  # type: ignore[arg-type]
                ["train_a"],
            )

    def test_manifest_rejects_changed_dataset(self) -> None:
        cache = _Cache()
        split = PartSplit(train=["train_a"], val=[], test=["test_a"])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "split.json"
            write_split_manifest(path, cache, split)  # type: ignore[arg-type]
            raw = json.loads(path.read_text(encoding="utf-8"))
            raw["dataset"] = "different"
            path.write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "does not match cache dataset"):
                load_split_manifest(path, cache)  # type: ignore[arg-type]


class CalibrationDiagnosticsTest(unittest.TestCase):
    def test_diagnostic_update_matches_compatibility_entry_point(self) -> None:
        logits = torch.tensor([[2.0, 1.5], [0.1, 0.4]], dtype=torch.float32)
        probs = torch.softmax(logits, dim=1)
        a, b, margin = inference_candidate_pairs(probs)
        meta = FaceBatchMeta(
            part_index=torch.zeros(2, dtype=torch.long),
            edges=torch.tensor([[0, 1]], dtype=torch.long),
            edge_type=torch.tensor([0], dtype=torch.long),
        )
        kwargs = dict(
            use_local_evidence=True,
            use_ambiguity_gate=True,
            use_part_prior=True,
            beta=0.5,
            lambda_cal=0.7,
            lambda_part=0.2,
            top_k=1,
        )
        diagnostics = compute_margin_calibration(probs, a, b, margin, meta, **kwargs)
        split_result = apply_pairwise_logit_update(logits, a, b, diagnostics.delta)
        compatibility_result = apply_margin_calibration(logits, probs, a, b, margin, meta, **kwargs)
        torch.testing.assert_close(split_result, compatibility_result)
        rows = torch.arange(2)
        before = logits[rows, a] - logits[rows, b]
        after = split_result[rows, a] - split_result[rows, b]
        torch.testing.assert_close(after, before + 2.0 * diagnostics.delta)


class FaceRecordTest(unittest.TestCase):
    def test_b0_and_b5_face_keys_align(self) -> None:
        record = PartRecord(
            index=0,
            part_id="part_1",
            dataset="unit_fixture",
            split="test",
            path=Path("part_1.pt"),
            num_faces=3,
            y=torch.tensor([10, 20, 10]),
            surface_type=torch.zeros(3, dtype=torch.long),
            edges=torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
            edge_type=torch.tensor([0, 1]),
            face_features_raw=torch.zeros((3, 2)),
            source_paths={},
            meta={},
        )
        class_index = ClassIndex([10, 20])
        base_logits = torch.tensor([[2.0, 1.0], [0.2, 0.8], [0.5, 0.4]])
        probs = torch.softmax(base_logits, dim=1)
        a, b, margin = inference_candidate_pairs(probs)
        meta = FaceBatchMeta(
            part_index=torch.zeros(3, dtype=torch.long),
            edges=record.edges,
            edge_type=record.edge_type,
        )
        diagnostics = compute_margin_calibration(
            probs,
            a,
            b,
            margin,
            meta,
            use_local_evidence=True,
            use_ambiguity_gate=True,
            use_part_prior=True,
            beta=0.5,
            lambda_cal=0.5,
            lambda_part=0.2,
            top_k=1,
        )
        calibrated = apply_pairwise_logit_update(base_logits, a, b, diagnostics.delta)
        b0 = build_face_prediction_records(
            record,
            class_index,
            base_logits,
            base_logits,
            a,
            b,
            margin,
            None,
            method="frozen_proto",
            variant="B0",
            budget="1",
            seed=0,
        )
        b5 = build_face_prediction_records(
            record,
            class_index,
            base_logits,
            calibrated,
            a,
            b,
            margin,
            diagnostics,
            method="topocalib_uc",
            variant="B5",
            budget="1",
            seed=0,
        )
        self.assertEqual([row["face_key"] for row in b0], [row["face_key"] for row in b5])
        for row in b5:
            self.assertAlmostEqual(
                row["pair_logit_margin_after"],
                row["pair_logit_margin_before"] + 2.0 * row["calibration_delta"],
                places=6,
            )
        self.assertEqual(b5[1]["neighbor_face_ids"], [0, 2])


if __name__ == "__main__":
    unittest.main()
