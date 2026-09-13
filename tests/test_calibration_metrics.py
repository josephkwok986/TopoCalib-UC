from __future__ import annotations

import unittest

import torch

from topocalib_uc.metrics.calibration import (
    calibration_metrics_from_logits,
    calibration_metrics_from_probabilities,
)


class CalibrationMetricsTest(unittest.TestCase):
    def test_perfect_confidence(self) -> None:
        probabilities = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float64)
        result = calibration_metrics_from_probabilities(probabilities, torch.tensor([0, 1]))
        self.assertEqual(result["ece"], 0.0)
        self.assertEqual(result["nll"], 0.0)

    def test_high_confidence_error(self) -> None:
        probabilities = torch.tensor([[0.99, 0.01]], dtype=torch.float64)
        result = calibration_metrics_from_probabilities(probabilities, torch.tensor([1]))
        self.assertAlmostEqual(result["ece"], 0.99)
        self.assertGreater(result["nll"], 4.0)

    def test_internal_boundary_enters_upper_bin(self) -> None:
        probabilities = torch.tensor([[0.6, 0.4]], dtype=torch.float64)
        result = calibration_metrics_from_probabilities(probabilities, torch.tensor([0]), num_bins=5)
        populated = [row for row in result["bins"] if row["count"]]
        self.assertEqual(len(populated), 1)
        self.assertEqual(populated[0]["index"], 3.0)

    def test_logits_and_probabilities_agree(self) -> None:
        logits = torch.tensor([[1.2, -0.2, 0.4], [-0.5, 0.3, 1.0]], dtype=torch.float64)
        targets = torch.tensor([10, 30])
        classes = [10, 20, 30]
        from_logits = calibration_metrics_from_logits(logits, targets, classes=classes)
        from_probs = calibration_metrics_from_probabilities(
            torch.softmax(logits, dim=1),
            targets,
            classes=classes,
        )
        self.assertAlmostEqual(from_logits["ece"], from_probs["ece"], places=12)
        self.assertAlmostEqual(from_logits["nll"], from_probs["nll"], places=12)

    def test_empty_input_and_unknown_class(self) -> None:
        empty = calibration_metrics_from_logits(torch.empty((0, 2)), torch.empty((0,), dtype=torch.long))
        self.assertEqual(empty["num_faces"], 0.0)
        with self.assertRaisesRegex(ValueError, "absent from classes"):
            calibration_metrics_from_logits(torch.zeros((1, 2)), torch.tensor([3]), classes=[1, 2])


if __name__ == "__main__":
    unittest.main()
