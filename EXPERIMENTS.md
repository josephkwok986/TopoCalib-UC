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

## Targeted structural controls and candidate selection (Tables 11 and A6)

The controls reuse the existing B5 training entry, leave-one-out prototype logits,
full-class supervision, and validation checkpoint selection. The default
`--calibration-mode pairwise --training-candidates gt_aware` preserves B0–B5.

| Control | Training options | Inference |
| --- | --- | --- |
| Structural probability smoothing | `--variant B5 --calibration-mode smoothing --smoothing-alpha 0.5` | Uses the saved smoothing rule |
| All-class structural adjustment | `--variant B5 --calibration-mode all_class --lambda-all 1` | Uses the saved all-class rule |
| Predicted training candidates | `--variant B5 --training-candidates predicted_top2` | Top-1/top-2, as for ordinary B5 |
| Frequency adjustment of B0 | Reuse a selected B0 checkpoint | `--frequency-eta 0.5` |

Smoothing normalizes detached local-plus-part evidence and mixes it with base
probabilities. Faces with zero evidence retain their base distribution. Its
training loss is the negative log-probability of the true class. The all-class
control adds `2 * gate * lambda_all * (evidence - class_mean_evidence)` to every
logit, matching the candidate-pair margin change at `lambda_all = lambda_cal`.
Frequency adjustment subtracts `eta * log(class_prior)`; counts come exclusively
from the checkpoint's labeled training parts. New checkpoints save these counts.
For older checkpoints, inference recovers counts from saved `labeled_parts`, or
from `--labeled-parts-manifest` together with `--split-manifest`. Missing classes
cause an explicit error rather than an invented prior.

Configuration files are under `experiment_protocols/configs/`. The default
strengths reproduce the settings specified in Appendix B. To repeat strength
selection, use the corresponding `*_validation.json` configuration: smoothing
uses `{0, 0.25, 0.5, 0.75}`, all-class adjustment `{0, 0.5, 1, 2}`, and frequency
adjustment `{0, 0.25, 0.5, 1, 2}`. Structural search configurations enable `--validation-only` to skip test evaluation. Selection uses mean validation mIoU over seeds
0, 1, and 2 at budget 1000; reuse the selected strength for both budgets.

Run the following from the repository root. The planner writes `commands.json`;
add `--execute` to run the commands. Protocol files must already exist and be
shared with the reference B0/B5 runs. Output directories should be distinct for
each dataset, control, and validation/final stage.

```bash
python -m scripts.run_targeted_experiments \
  --config experiment_protocols/configs/structural_smoothing_validation.json \
  --dataset fusion360 \
  --cache-dir /data/fusion360_filtered \
  --ssrl-cache-dir /data/fusion360_ssrl_faces \
  --split-manifest /results/protocol/fusion360_split.json \
  --labeled-parts-template '/results/protocol/fusion360_budget{budget}_seed{seed}.json' \
  --output-root /results/smoothing_validation \
  --execute

python -m scripts.select_control_strength \
  --input /results/smoothing_validation/fusion360_budget1000_seed*_strength*.json \
  --output /results/protocol/fusion360_smoothing_strength.json

python -m scripts.run_targeted_experiments \
  --config experiment_protocols/configs/structural_smoothing.json \
  --selected-strength /results/protocol/fusion360_smoothing_strength.json \
  --dataset fusion360 \
  --cache-dir /data/fusion360_filtered \
  --ssrl-cache-dir /data/fusion360_ssrl_faces \
  --split-manifest /results/protocol/fusion360_split.json \
  --labeled-parts-template '/results/protocol/fusion360_budget{budget}_seed{seed}.json' \
  --output-root /results/smoothing_final \
  --export-predictions test \
  --execute
```

Use `all_class_adjustment_validation.json` and `all_class_adjustment.json` for
the all-class control. Use `predicted_candidates.json` for Table A6; it has no
strength search. Replace `fusion360` and all input/output paths for MFCAD++.
Additional training tokens may be supplied with repeated `--train-arg`, for
example `--train-arg=--epochs --train-arg=200`. Do not override the planner's
checkpoint or output paths when requesting prediction export.

For frequency adjustment, first run `frequency_adjustment_validation.json`
with `--split-group val`, then select its strength from the resulting inference
JSON files. Use `frequency_adjustment.json`, `--selected-strength`, and
`--split-group test` for final evaluation. These are inference jobs and require
`--checkpoint-template '/results/B0_budget{budget}_seed{seed}.pt'` instead of a
labeled-parts template. Every B0 checkpoint must correspond to that budget and
seed. No retraining is performed.

Export reference B5 test predictions with `topocalib_uc.inference.run` using the
same split manifest. Pair them with the control's `*_predictions.json` files at
one budget and dataset at a time:

```bash
python -m scripts.summarize_targeted_experiments \
  --left /results/reference/B5_budget200_seed0.json /results/reference/B5_budget200_seed1.json /results/reference/B5_budget200_seed2.json \
  --right /results/smoothing_final/fusion360_budget200_seed0_strength0.5_predictions.json /results/smoothing_final/fusion360_budget200_seed1_strength0.5_predictions.json /results/smoothing_final/fusion360_budget200_seed2_strength0.5_predictions.json \
  --resamples 10000 --seed 0 \
  --output /results/table11/fusion360_budget200_smoothing.json
```

The difference is **left minus right**. For Table A6, put predicted-candidate
training on the left and GT-aware B5 on the right, following the table's sign
convention. Bootstrap draws sample whole test parts with replacement and share
the same draw across methods and seeds. mIoU is recomputed from pooled confusion
matrices before averaging paired differences over training seeds. Confidence
intervals are unadjusted percentile intervals conditional on those training
runs. Individual method mIoUs include the sample standard deviation over seeds.

## Controlled adjacency deletion (Table 10)

`adjacency_deletion.json` specifies budget 1000, training seeds 0–2, deletion
probabilities 0, 0.1, 0.3, and 0.5, and perturbation seeds 0–2 for nonzero
probabilities. The perturbation seed is combined with the part ID through a
stable hash, so each part has a reproducible draw independent of iteration
order. One keep/delete draw is shared by both directions and all relation
records of an undirected face pair. The realization seed choice is explicit in
the configuration; the manuscript specifies three realizations without naming
numerical realization seeds.

```bash
python -m scripts.run_targeted_experiments \
  --config experiment_protocols/configs/adjacency_deletion.json \
  --dataset fusion360 \
  --cache-dir /data/fusion360_filtered \
  --ssrl-cache-dir /data/fusion360_ssrl_faces \
  --split-manifest /results/protocol/fusion360_split.json \
  --checkpoint-template '/results/B5_budget{budget}_seed{seed}.pt' \
  --output-root /results/topology/fusion360 \
  --execute

python -m scripts.summarize_targeted_experiments \
  --topology \
  --left /results/topology/fusion360/fusion360_budget1000_seed*_delete*.json \
  --output /results/table10/fusion360.json
```

The same checkpoint is reused for all conditions. All faces, adapter parameters,
embeddings, prototypes, and same-part evidence remain fixed; only local evidence
uses the remaining adjacencies. Results average the three realizations within
each training seed before reporting mean and sample standard deviation across
training seeds. Changes from clean and calibration gains are calculated within
seeds. Prediction files retain initial labels as `base_pred` for this comparison.
Structural-control and frequency-adjustment predictions use JSON output with
`--save-logits`; the pairwise diagnostic JSONL schema is reserved for pairwise
calibration.

## Correction outcomes and paired component increments (Tables 9, A4, A5)

Export the unperturbed B5 face records with `--records-output` using the fixed
test split, then summarize the three seed files for one dataset:

```bash
python -m scripts.analyze_correction_outcomes \
  --input-jsonl /results/records/B5_seed0.jsonl /results/records/B5_seed1.jsonl /results/records/B5_seed2.jsonl \
  --output /results/correction_outcomes/fusion360.json
```

Transition percentages use all test faces. Top-2 coverage uses initially
incorrect faces. Evidence groups use gate-active faces and the signs of the
local and part terms: both positive, both negative, opposite signs, or at least
one zero. Recovery and damage use the initially incorrect and initially correct
faces in each group, respectively. An undefined rate in any seed yields `null`
for the aggregate, corresponding to N/A in a table.

For Table A5, run `scripts.summarize_targeted_experiments` with `--randomization`
and `--resamples 10000` for B4 minus B3 and B5 minus B4 on each dataset. Each
randomization draw swaps whole-part predictions between methods, with the same
swap shared across seeds. The two-sided Monte Carlo p-value includes a plus-one
correction. Apply Holm correction jointly to the four saved results:

```bash
python -m scripts.adjust_component_pvalues \
  --input /results/paired/fusion360_B4_B3.json /results/paired/fusion360_B5_B4.json /results/paired/mfcadpp_B4_B3.json /results/paired/mfcadpp_B5_B4.json \
  --output /results/table_a5.json
```
