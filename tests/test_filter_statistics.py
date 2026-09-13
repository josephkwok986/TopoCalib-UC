from __future__ import annotations

from collections import Counter
import unittest

from filter_data.scripts.compute_ssrl_filtered_subset_stats import build_table_a2_statistics


class TableA2StatisticsTest(unittest.TestCase):
    def test_shares_retention_and_extrema(self) -> None:
        summary = {
            "dataset_name": "unit_fixture",
            "original_num_parts": 10,
            "filtered_num_parts": 7,
            "original_num_faces": 100,
            "filtered_num_faces": 65,
            "part_retention": 0.7,
            "face_retention": 0.65,
        }
        enriched, rows = build_table_a2_statistics(
            summary,
            Counter({0: 50, 1: 30, 2: 20}),
            Counter({0: 40, 1: 15, 2: 10}),
            {0: "zero", 1: "one", 2: "two"},
        )
        by_id = {int(row["class_id"]): row for row in rows}
        self.assertAlmostEqual(float(by_id[0]["before_share"]), 0.5)
        self.assertAlmostEqual(float(by_id[0]["after_share"]), 40 / 65)
        self.assertAlmostEqual(float(by_id[0]["share_change_pp"]), 100 * (40 / 65 - 0.5))
        self.assertAlmostEqual(float(by_id[1]["retention_ratio"]), 0.5)
        table = enriched["table_a2"]
        self.assertAlmostEqual(float(table["class_retention_min"]), 0.5)
        self.assertEqual(table["class_retention_min_class_id"], 1)
        self.assertAlmostEqual(float(table["class_retention_max"]), 0.8)
        self.assertEqual(table["largest_class_share_change_class_name"], "zero")
        self.assertAlmostEqual(
            float(table["largest_absolute_class_share_change_pp"]),
            abs(100 * (40 / 65 - 0.5)),
        )


if __name__ == "__main__":
    unittest.main()
