# Fixed-protocol analyses and inference benchmarks

This document covers the additional experiment entries used for Tables A2, 5, 6, 8, B4, B5, and B6 and Fig. 2. Paths below are examples; keep raw data, checkpoints, and generated results outside the repository.

## Fixed split, labeled parts, and face records

Freeze a split and one labeled-part selection during the first run:

```bash
python -m topocalib_uc.train.run_frozen_proto \
  --cache-dir /data/fusion360_filtered \
  --ssrl-cache-dir /data/fusion360_ssrl_faces \
  --output /results/B0_seed0.json \
  --split-source random_filtered \
  --split-seed 0 \
  --seed 0 \
  --labeled-part-budget 1000 \
  --variant B0 \
  --write-split-manifest /results/protocol/fusion360_split.json \
  --write-labeled-parts-manifest /results/protocol/fusion360_budget1000_seed0.json
```

Reuse the fixed split for every method and seed. Reuse a labeled-parts manifest when the compared runs are intended to share the same labeled set:

```bash
python -m topocalib_uc.train.run_frozen_proto \
  --cache-dir /data/fusion360_filtered \
  --ssrl-cache-dir /data/fusion360_ssrl_faces \
  --output /results/B5_seed0.json \
  --split-manifest /results/protocol/fusion360_split.json \
  --labeled-parts-manifest /results/protocol/fusion360_budget1000_seed0.json \
  --seed 0 \
  --labeled-part-budget 1000 \
  --variant B5
```

The Frozen + Linear and SSRL-FE + MR-GCN training entries accept the same `--split-manifest`, `--split-seed`, `--labeled-parts-manifest`, and manifest-writing options.

Export the fixed-test prediction records used by the analyses:

```bash
python -m topocalib_uc.inference.run \
  --checkpoint /results/B5_seed0.pt \
  --cache-dir /data/fusion360_filtered \
  --ssrl-cache-dir /data/fusion360_ssrl_faces \
  --split-manifest /results/protocol/fusion360_split.json \
  --split-group test \
  --records-output /results/records/B5_seed0.jsonl \
  --output /results/predictions/B5_seed0.json \
  --device cuda
```

The JSONL face key is `(dataset, part_id, face_id)`, where `face_id` is the zero-based local face index. Diagnostic export is disabled unless `--records-output` is supplied.

## Table A2: filtering statistics

```bash
python filter_data/scripts/compute_ssrl_filtered_subset_stats.py \
  --dataset_name fusion360 \
  --dataset_root /data/s2.0.1 \
  --step_dir /data/s2.0.1/breps/step \
  --label_dir /data/s2.0.1/breps/seg \
  --output_dir /results/table_a2/fusion360 \
  --progress_interval 1000
```

The output directory contains `retention_summary.json`, `table_a2_summary.csv`, and `class_distribution_before_after.csv`. Use `dataset_name=mfcadpp` with the corresponding MFCAD++ paths for the second dataset.

## Table 8: ECE and NLL

```bash
python scripts/aggregate_confidence_metrics.py \
  --input-jsonl /results/records/B0_seed0.jsonl \
  --input-jsonl /results/records/B0_seed1.jsonl \
  --input-jsonl /results/records/B0_seed2.jsonl \
  --input-jsonl /results/records/B5_seed0.jsonl \
  --input-jsonl /results/records/B5_seed1.jsonl \
  --input-jsonl /results/records/B5_seed2.jsonl \
  --num-bins 15 \
  --expected-seeds 0 1 2 \
  --std-ddof 1 \
  --output /results/table8/fusion360.json \
  --csv-output /results/table8/fusion360.csv
```

ECE is retained internally in `[0,1]` and is also exported in percent. NLL is computed from final logits with `log_softmax`.

## Table 6 and Table A3: class-frequency groups

Provide a JSON list or `{class_id: class_name}` object for display names. Fusion360 uses group sizes `3 2 3`; MFCAD++ uses `8 9 8`.

```bash
python scripts/analyze_class_frequency.py \
  --cache-dir /data/fusion360_filtered \
  --split-manifest /results/protocol/fusion360_split.json \
  --class-names-json /data/fusion360_class_names.json \
  --group-sizes 3 2 3 \
  --input-jsonl /results/records/B0_seed0.jsonl \
  --input-jsonl /results/records/B0_seed1.jsonl \
  --input-jsonl /results/records/B0_seed2.jsonl \
  --input-jsonl /results/records/B5_seed0.jsonl \
  --input-jsonl /results/records/B5_seed1.jsonl \
  --input-jsonl /results/records/B5_seed2.jsonl \
  --expected-seeds 0 1 2 \
  --group-manifest-output /results/table6/fusion360_groups.json \
  --output /results/table6/fusion360.json
```

Frequencies are calculated from the full fixed training split before labeled-part sampling. Group mIoU is the unweighted mean of saved per-class IoUs.

## Fig. 2 and Table 5: fixed subsets and buckets

```bash
python scripts/analyze_fixed_subsets.py \
  --input-jsonl /results/records/B0_seed0.jsonl \
  --input-jsonl /results/records/B0_seed1.jsonl \
  --input-jsonl /results/records/B0_seed2.jsonl \
  --input-jsonl /results/records/B3_seed0.jsonl \
  --input-jsonl /results/records/B3_seed1.jsonl \
  --input-jsonl /results/records/B3_seed2.jsonl \
  --input-jsonl /results/records/B4_seed0.jsonl \
  --input-jsonl /results/records/B4_seed1.jsonl \
  --input-jsonl /results/records/B4_seed2.jsonl \
  --input-jsonl /results/records/B5_seed0.jsonl \
  --input-jsonl /results/records/B5_seed1.jsonl \
  --input-jsonl /results/records/B5_seed2.jsonl \
  --reference-seeds 0 1 2 \
  --beta 0.10 \
  --membership-output /results/fig2/fusion360_membership.json \
  --output /results/fig2/fusion360_summary.json
```

The script rejects duplicate, missing, or inconsistent face records before computing any statistics.

## Table B6: prototype construction

Mean is the default. Medoid and sDBSCAN change only prototype construction:

```bash
python -m topocalib_uc.train.run_frozen_proto \
  --cache-dir /data/fusion360_filtered \
  --ssrl-cache-dir /data/fusion360_ssrl_faces \
  --split-manifest /results/protocol/fusion360_split.json \
  --labeled-parts-manifest /results/protocol/fusion360_budget1000_seed0.json \
  --labeled-part-budget 1000 \
  --seed 0 \
  --variant B5 \
  --prototype-method medoid \
  --medoid-chunk-size 1024 \
  --output /results/table_b6/medoid_seed0.json
```

Build the vendored extension in place:

```bash
cd baselines/external/sDbscan
python setup.py build_ext --inplace
python -c "import sDbscan; print(sDbscan.__file__)"
```

For Table B6, the validation grid is `eps={0.025, 0.05, 0.075, 0.10, 0.125, 0.15, 0.175, 0.20}` and `minPts={3, 5, 10}`. Selection uses the highest mean validation mIoU at the 1000-labeled-part setting over seeds `{0, 1, 2}`. The approximation parameters are fixed to `D=1024`, `k=10`, and `m=10`. The selected Fusion360 configuration is `eps=0.15`, `minPts=5`:

```bash
python -m topocalib_uc.train.run_frozen_proto \
  --cache-dir /data/fusion360_filtered \
  --ssrl-cache-dir /data/fusion360_ssrl_faces \
  --split-manifest /results/protocol/fusion360_split.json \
  --labeled-parts-manifest /results/protocol/fusion360_budget1000_seed0.json \
  --labeled-part-budget 1000 \
  --seed 0 \
  --variant B5 \
  --prototype-method sdbscan \
  --sdbscan-eps 0.15 \
  --sdbscan-min-pts 5 \
  --sdbscan-n-proj 1024 \
  --sdbscan-top-k 10 \
  --sdbscan-top-m 10 \
  --sdbscan-n-threads 1 \
  --sdbscan-random-seed 0 \
  --output /results/table_b6/sdbscan_seed0.json
```

For MFCAD++, use `--sdbscan-eps 0.10`; the remaining selected and fixed parameters are unchanged. For seeds 1 and 2, use the corresponding labeled-parts manifest and set both `--seed` and `--sdbscan-random-seed` to the run seed. The complete protocol is also stored in `experiment_protocols/sdbscan_table_b6.json`. Checkpoints and result JSON record the strategy, every parameter, and per-class retained/noise/fallback counts.

## Tables B4 and B5: cached-feature inference benchmark

Build the fixed workload before timing. Repeated sampling is rejected unless `--allow-repeats` is explicitly supplied.

```bash
python scripts/benchmark_inference.py build-workload \
  --cache-dir /data/fusion360_filtered \
  --split-manifest /results/protocol/fusion360_split.json \
  --name table_b4 \
  --num-parts 1000 \
  --min-faces 13 \
  --max-faces 13 \
  --seed 0 \
  --output /results/benchmark/table_b4_workload.json
```

For Table B5, build four manifests with face ranges `3 8`, `9 17`, `18 38`, and `44 130`. Then run all compared checkpoints on the same manifest:

```bash
python scripts/benchmark_inference.py run \
  --cache-dir /data/fusion360_filtered \
  --ssrl-cache-dir /data/fusion360_ssrl_faces \
  --split-manifest /results/protocol/fusion360_split.json \
  --workload-manifest /results/benchmark/table_b4_workload.json \
  --checkpoint Frozen+Proto=/results/B0_seed0.pt \
  --checkpoint TopoCalib-UC=/results/B5_seed0.pt \
  --checkpoint SSRL-MRGCN=/results/ssrl_mrgcn_seed0.pt \
  --batch-size 16 \
  --warmup-repetitions 1 \
  --repetitions 5 \
  --std-ddof 1 \
  --device cuda \
  --output /results/benchmark/table_b4.json
```

The CUDA path synchronizes before and after every timed workload. The output contains every raw repetition, amortized time, allocated-memory baseline, absolute peak, incremental peak, parameter count, workload IDs, and software/hardware metadata.
