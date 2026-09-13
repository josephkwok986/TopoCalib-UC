from __future__ import annotations

import unittest

from scripts.analyze_fixed_subsets import (
    BUCKETS,
    build_membership,
    margin_bucket,
    validate_aligned_runs,
)


def _run(seed: int, margins: list[float]) -> dict:
    targets = [0, 1, 2, 1, 0]
    neighbors = [[1, 2], [0], [0], [4], [3]]
    faces = {}
    for face_id, (target, margin) in enumerate(zip(targets, margins)):
        key = ("unit_fixture", "part", face_id)
        faces[key] = {
            "target": target,
            "pred_after": target,
            "neighbor_face_ids": neighbors[face_id],
            "probability_margin_before": margin,
        }
    return {
        "path": f"seed_{seed}.jsonl",
        "dataset": "unit_fixture",
        "method": "frozen_proto",
        "variant": "B0",
        "budget": "1000",
        "seed": seed,
        "classes": [0, 1, 2],
        "faces": faces,
    }


class FixedSubsetTest(unittest.TestCase):
    def test_membership_and_fixed_buckets(self) -> None:
        margins = [0.01, 0.03, 0.07, 0.10, 1.0]
        runs = [_run(seed, margins) for seed in (0, 1, 2)]
        validate_aligned_runs(runs)
        membership = build_membership(runs, beta=0.10)
        ordered = [membership[("unit_fixture", "part", index)] for index in range(5)]
        self.assertTrue(ordered[0]["hard"])
        self.assertFalse(ordered[1]["hard"])
        self.assertEqual([row["bucket"] for row in ordered], [item[0] for item in BUCKETS])
        ambiguity_keys = {tuple(row["face_key"]) for row in ordered if row["ambiguity"]}
        first_three = {tuple(row["face_key"]) for row in ordered if row["bucket"] in {item[0] for item in BUCKETS[:3]}}
        self.assertEqual(ambiguity_keys, first_three)

    def test_bucket_boundaries(self) -> None:
        self.assertEqual(margin_bucket(0.02), "[0.02,0.05)")
        self.assertEqual(margin_bucket(0.10), "[0.10,0.20)")
        self.assertEqual(margin_bucket(1.0), "[0.20,1.00]")

    def test_alignment_rejects_changed_target(self) -> None:
        runs = [_run(0, [0.1] * 5), _run(1, [0.1] * 5)]
        runs[1]["faces"][("unit_fixture", "part", 0)]["target"] = 2
        with self.assertRaisesRegex(ValueError, "differ from the fixed test records"):
            validate_aligned_runs(runs)


if __name__ == "__main__":
    unittest.main()
