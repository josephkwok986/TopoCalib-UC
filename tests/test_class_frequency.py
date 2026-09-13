from __future__ import annotations

import unittest

import torch

from scripts.analyze_class_frequency import build_frequency_manifest
from topocalib_uc.metrics.segmentation import grouped_mean_iou, mean_iou, per_class_iou
from topocalib_uc.train.splitting import PartSplit


class _Record:
    def __init__(self, part_id: str, labels: list[int]) -> None:
        self.part_id = part_id
        self.dataset = "unit_fixture"
        self.y = torch.tensor(labels)


class _Cache:
    def __init__(self) -> None:
        self.classes = [0, 1, 2, 3, 4, 5]
        self.records = [
            _Record("train_a", [0, 0, 0, 1, 1, 2, 3, 4, 5]),
            _Record("train_b", [0, 0, 1, 2, 3, 4, 5]),
            _Record("test_a", [0, 1, 2, 3, 4, 5]),
        ]
        self._by_id = {record.part_id: record for record in self.records}

    def by_part_id(self, part_id: str) -> _Record:
        return self._by_id[part_id]


class ClassFrequencyTest(unittest.TestCase):
    def test_frequency_ranking_tie_break_and_partition(self) -> None:
        cache = _Cache()
        split = PartSplit(train=["train_a", "train_b"], val=[], test=["test_a"])
        manifest = build_frequency_manifest(cache, split, [2, 2, 2], {0: "zero"})  # type: ignore[arg-type]
        ranked = [row["class_id"] for row in manifest["classes"]]
        self.assertEqual(ranked, [0, 1, 2, 3, 4, 5])
        assigned = [class_id for values in manifest["groups"].values() for class_id in values]
        self.assertEqual(sorted(assigned), cache.classes)
        self.assertEqual(len(assigned), len(set(assigned)))
        self.assertAlmostEqual(sum(row["training_face_share"] for row in manifest["group_summary"]), 1.0)

    def test_group_miou_recomputes_from_per_class_rows(self) -> None:
        target = torch.tensor([0, 0, 1, 1, 2, 2])
        pred = torch.tensor([0, 1, 1, 1, 2, 0])
        rows = per_class_iou(pred, target, [0, 1, 2])
        grouped = grouped_mean_iou(rows, {"Head": [0], "Body": [1], "Tail": [2]})
        self.assertEqual(grouped["Head"], rows[0]["iou"])
        self.assertAlmostEqual(mean_iou(pred, target, [0, 1, 2]), sum(grouped.values()) / 3)


if __name__ == "__main__":
    unittest.main()
