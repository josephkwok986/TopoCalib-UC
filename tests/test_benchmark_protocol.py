from __future__ import annotations

import unittest

from scripts.benchmark_inference import build_workload_manifest, measure_runner


class _Record:
    def __init__(self, part_id: str, num_faces: int) -> None:
        self.part_id = part_id
        self.num_faces = num_faces
        self.dataset = "unit_fixture"


class _Cache:
    def __init__(self) -> None:
        self.records = [_Record(f"part_{index}", index + 3) for index in range(6)]
        self.part_ids = [record.part_id for record in self.records]
        self._by_id = {record.part_id: record for record in self.records}

    def by_part_id(self, part_id: str) -> _Record:
        return self._by_id[part_id]


class BenchmarkProtocolTest(unittest.TestCase):
    def test_workload_range_and_explicit_repeat_policy(self) -> None:
        cache = _Cache()
        with self.assertRaisesRegex(ValueError, "allow-repeats"):
            build_workload_manifest(
                cache,  # type: ignore[arg-type]
                cache.part_ids,
                name="small",
                num_parts=10,
                min_faces=3,
                max_faces=8,
                seed=0,
                allow_repeats=False,
            )
        manifest = build_workload_manifest(
            cache,  # type: ignore[arg-type]
            cache.part_ids,
            name="small",
            num_parts=10,
            min_faces=3,
            max_faces=8,
            seed=0,
            allow_repeats=True,
        )
        self.assertEqual(len(manifest["part_ids"]), 10)
        self.assertTrue(manifest["has_repeats"])
        self.assertEqual(manifest["actual_face_range"], [3, 8])
        self.assertEqual(manifest["total_faces"], sum(cache.by_part_id(item).num_faces for item in manifest["part_ids"]))

    def test_cpu_smoke_timing_marks_memory_unavailable(self) -> None:
        calls = 0

        def runner() -> None:
            nonlocal calls
            calls += 1

        import torch

        result = measure_runner(
            runner,
            num_parts=4,
            device=torch.device("cpu"),
            warmup_repetitions=2,
            repetitions=3,
            std_ddof=1,
        )
        self.assertEqual(calls, 5)
        self.assertEqual(len(result["workload_time_ms_values"]), 3)
        self.assertIsNone(result["peak_allocated_bytes"])


if __name__ == "__main__":
    unittest.main()
