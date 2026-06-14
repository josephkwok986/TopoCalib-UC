# TopoCalib-UC

TopoCalib-UC is a lightweight downstream learning method for low-label CAD B-Rep face-level segmentation. The method builds a token adapter and class prototypes on top of a frozen SSRL B-Rep face encoder. For the currently most confusing candidate class pair, it introduces shared-boundary local evidence and same-part prototype evidence to perform ambiguity-aware calibration on the prototype decision margin. Training and inference share the same candidate-pair calibration rule.

The experiments cover Fusion360 Gallery Segmentation and MFCAD++. This repository only contains code, adaptation layers, and experiment entry points; raw data, preprocessed outputs, checkpoints, logs, and result files should all be stored in external directories outside the repository.

## Environment

The repository provides three Dockerfiles, respectively for CAD/B-Rep preprocessing, TopoCalib-UC downstream training, and HybridBrep/SSRL representation training.

### CAD/B-Rep Preprocessing Environment

`Dockerfile.cad_preprocess` is used for OpenCascade/pythonocc-related STEP reading, B-Rep geometry parsing, PartGraph construction, cache checking, and auxiliary chart processing.

This command builds the CAD/B-Rep preprocessing image.

```bash
docker build -t topocalib_cad_preprocess:1.0 -f Dockerfile.cad_preprocess .
```

This command enters the CAD/B-Rep preprocessing container.

```bash
docker run -it --rm \
  --name topocalib_cad_preprocess \
  -v "/abs/path/TopoCalib-UC":"/workspace/TopoCalib-UC" \
  -v "/abs/data":"/data" \
  -v "/abs/out":"/out" \
  topocalib_cad_preprocess:1.0 \
  bash -lc 'cd /workspace/TopoCalib-UC && exec bash'
```

### TopoCalib-UC Training Environment

`Dockerfile.topocalib_train` is used for PartGraph loading, TopoCalib-UC, Frozen + Linear, Frozen + Proto, SSRL + MR-GCN, result organization, and external baseline adaptation layers.

This command builds the TopoCalib-UC downstream training image.

```bash
docker build -t topocalib_train:1.0 -f Dockerfile.topocalib_train .
```

This command enters the TopoCalib-UC downstream training container.

```bash
docker run -it --rm --gpus all \
  --name topocalib_train \
  -v "/abs/path/TopoCalib-UC":"/workspace/TopoCalib-UC" \
  -v "/abs/data":"/data" \
  -v "/abs/out":"/out" \
  topocalib_train:1.0 \
  bash -lc 'cd /workspace/TopoCalib-UC && exec bash'
```

### SSRL Training Environment

`Dockerfile.hybridbrep_cad1.1` is used for HybridBrep preprocessing, SSRL encoder training, and SSRL face representation export. This environment is maintained separately from the downstream training environment.

This command builds the HybridBrep/SSRL training image.

```bash
docker build -t topocalib_hybridbrep:1.1 -f Dockerfile.hybridbrep_cad1.1 .
```

This command enters the HybridBrep/SSRL training container.

```bash
docker run -it --rm --gpus all \
  --name topocalib_hybridbrep \
  -v "/abs/path/TopoCalib-UC":"/workspace/TopoCalib-UC" \
  -v "/abs/data":"/data" \
  -v "/abs/out":"/out" \
  topocalib_hybridbrep:1.1 \
  bash -lc 'cd /workspace/TopoCalib-UC && exec bash'
```

After entering `topocalib_hybridbrep`, run the following first if the HybridBrep C++ extension has not yet been built:

This command compiles the HybridBrep C++ extension inside the container.

```bash
build_hybridbrep_cpp
```

## Data Preparation

Data inputs and outputs should not be placed inside the repository. The following paths are placeholder examples; replace them with real absolute paths before execution.

```text
/data/fusion360_gallery/
  breps/
    step/
    seg/

/data/mfcadpp/
  hierarchical_graphs/
  step/

/out/topocalib_uc/
  fusion360_filtered/
  mfcadpp_filtered/
  fusion360_hybridbrep/
  mfcadpp_hybridbrep/
  fusion360_ssrl_faces/
  mfcadpp_ssrl_faces/
  checkpoints/
  results/
  external_baselines/
```

## Full Reproduction

This workflow includes CAD parsing, SSRL training, downstream training for multiple budget/seed combinations, and external baselines. The full workflows for Fusion360 Gallery and MFCAD++ are usually time-consuming. CAD preprocessing and HybridBrep preprocessing are strongly affected by STEP parsing speed, while SSRL and graph-model training are strongly affected by GPU memory and batch settings.

### 1. Fusion360 PartGraph

Input: raw Fusion360 Gallery `breps/step` and `breps/seg`. Output: filtered PartGraph `.pt` files, manifest, and summary.

This command reads the STEP and face label files from Fusion360 Gallery and generates the Fusion360 filtered PartGraph cache.

```bash
python data_protocol/scripts/build_fusion360_cache.py \
  --dataset-root /data/fusion360_gallery \
  --output-root /out/topocalib_uc \
  --cache-name fusion360_filtered \
  --error-log /out/topocalib_uc/fusion360_filtered/error.jsonl
```

This command summarizes the parts, faces, edges, and label distribution in the Fusion360 PartGraph cache.

```bash
python data_protocol/scripts/summarize_pt_cache.py \
  --cache-dir /out/topocalib_uc/fusion360_filtered \
  --output /out/topocalib_uc/results/fusion360_partgraph_summary.json
```

### 2. MFCAD++ PartGraph

Input: MFCAD++ `hierarchical_graphs` and `step`. Output: filtered PartGraph `.pt` files, manifest, and summary.

This command reads the hierarchical graph and STEP files from MFCAD++ and generates the MFCAD++ filtered PartGraph cache.

```bash
python data_protocol/scripts/build_mfcadpp_cache.py \
  --dataset-root /data/mfcadpp \
  --output-root /out/topocalib_uc \
  --cache-name mfcadpp_filtered \
  --error-log /out/topocalib_uc/mfcadpp_filtered/error.jsonl
```

This command summarizes the parts, faces, edges, and label distribution in the MFCAD++ PartGraph cache.

```bash
python data_protocol/scripts/summarize_pt_cache.py \
  --cache-dir /out/topocalib_uc/mfcadpp_filtered \
  --output /out/topocalib_uc/results/mfcadpp_partgraph_summary.json
```

### 3. HybridBrep Preprocessing

Run the following commands in the `Dockerfile.hybridbrep_cad1.1` environment. Input: STEP paths recorded in the PartGraph cache. Output: HybridBrep graph `.pt` files and manifest.

This command converts the STEP files recorded in the Fusion360 PartGraph cache into a HybridBrep graph cache.

```bash
python frozen_representation/scripts/build_hybridbrep_cache.py \
  --partgraph-cache-dir /out/topocalib_uc/fusion360_filtered \
  --output-cache-dir /out/topocalib_uc/fusion360_hybridbrep \
  --n-samples 500 \
  --n-ref-samples 5000 \
  --sort-frac 0.5
```

This command converts the STEP files recorded in the MFCAD++ PartGraph cache into a HybridBrep graph cache.

```bash
python frozen_representation/scripts/build_hybridbrep_cache.py \
  --partgraph-cache-dir /out/topocalib_uc/mfcadpp_filtered \
  --output-cache-dir /out/topocalib_uc/mfcadpp_hybridbrep \
  --n-samples 500 \
  --n-ref-samples 5000 \
  --sort-frac 0.5
```

### 4. SSRL Encoder Training

Input: HybridBrep graph cache. Output: SSRL encoder checkpoint. During training, losses and process logs are continuously printed to stdout.

This command trains the SSRL encoder on the Fusion360 HybridBrep graph cache.

```bash
python frozen_representation/scripts/train_ssrl.py \
  --preprocessed-cache-dir /out/topocalib_uc/fusion360_hybridbrep \
  --output-ckpt /out/topocalib_uc/checkpoints/fusion360_ssrl_encoder.pt \
  --embedding-dim 256 \
  --hidden-dim 1024 \
  --layers 4 \
  --attn-heads 16 \
  --epochs 200 \
  --batch-size 1 \
  --lr 0.0005 \
  --weight-decay 0.0 \
  --device cuda
```

This command trains the SSRL encoder on the MFCAD++ HybridBrep graph cache.

```bash
python frozen_representation/scripts/train_ssrl.py \
  --preprocessed-cache-dir /out/topocalib_uc/mfcadpp_hybridbrep \
  --output-ckpt /out/topocalib_uc/checkpoints/mfcadpp_ssrl_encoder.pt \
  --embedding-dim 256 \
  --hidden-dim 1024 \
  --layers 4 \
  --attn-heads 16 \
  --epochs 200 \
  --batch-size 1 \
  --lr 0.0005 \
  --weight-decay 0.0 \
  --device cuda
```

### 5. SSRL Face Representation

Input: HybridBrep graph cache and SSRL checkpoint. Output: face representation `.pt` files for each part and a manifest.

This command loads the Fusion360 SSRL checkpoint and exports the Fusion360 face-level representation cache.

```bash
python frozen_representation/scripts/export_ssrl_embeddings.py \
  --preprocessed-cache-dir /out/topocalib_uc/fusion360_hybridbrep \
  --output-cache-dir /out/topocalib_uc/fusion360_ssrl_faces \
  --ckpt /out/topocalib_uc/checkpoints/fusion360_ssrl_encoder.pt \
  --embedding-dim 256 \
  --hidden-dim 1024 \
  --layers 4 \
  --attn-heads 16 \
  --device cuda
```

This command loads the MFCAD++ SSRL checkpoint and exports the MFCAD++ face-level representation cache.

```bash
python frozen_representation/scripts/export_ssrl_embeddings.py \
  --preprocessed-cache-dir /out/topocalib_uc/mfcadpp_hybridbrep \
  --output-cache-dir /out/topocalib_uc/mfcadpp_ssrl_faces \
  --ckpt /out/topocalib_uc/checkpoints/mfcadpp_ssrl_encoder.pt \
  --embedding-dim 256 \
  --hidden-dim 1024 \
  --layers 4 \
  --attn-heads 16 \
  --device cuda
```

### 6. Fusion360 Main Experiments

Input: Fusion360 PartGraph and SSRL face representation. Output: one JSON result file per command, containing the split, labeled parts, tail of the training log, and test metrics.

Frozen + Linear:

This command runs the Frozen + Linear baseline on Fusion360.

```bash
python -m topocalib_uc.train.run_frozen_linear \
  --cache-dir /out/topocalib_uc/fusion360_filtered \
  --output /out/topocalib_uc/results/fusion360_main/frozen_linear_budget100_seed0.json \
  --split-source random_filtered \
  --train-ratio 0.7 \
  --val-ratio 0.15 \
  --test-ratio 0.15 \
  --seed 0 \
  --min-parts-per-class 0 \
  --labeled-part-budget 100 \
  --ssrl-cache-dir /out/topocalib_uc/fusion360_ssrl_faces \
  --embedding-dim 256 \
  --hidden-dim 128 \
  --epochs 200 \
  --patience 20 \
  --lr 0.001 \
  --weight-decay 0.0001 \
  --num-surface-types 5
```

Frozen + Proto:

This command runs the Frozen + Proto baseline on Fusion360.

```bash
python -m topocalib_uc.train.run_frozen_proto \
  --cache-dir /out/topocalib_uc/fusion360_filtered \
  --output /out/topocalib_uc/results/fusion360_main/frozen_proto_B0_budget100_seed0.json \
  --split-source random_filtered \
  --train-ratio 0.7 \
  --val-ratio 0.15 \
  --test-ratio 0.15 \
  --seed 0 \
  --min-parts-per-class 0 \
  --labeled-part-budget 100 \
  --ssrl-cache-dir /out/topocalib_uc/fusion360_ssrl_faces \
  --embedding-dim 256 \
  --hidden-dim 128 \
  --epochs 200 \
  --patience 20 \
  --lr 0.001 \
  --weight-decay 0.0001 \
  --tau 0.07 \
  --num-surface-types 5 \
  --variant B0 \
  --beta 0.10 \
  --part-top-k 5 \
  --lambda-cal 1.0 \
  --lambda-part 0.5
```

TopoCalib-UC:

This command runs the full TopoCalib-UC model on Fusion360.

```bash
python -m topocalib_uc.train.run_frozen_proto \
  --cache-dir /out/topocalib_uc/fusion360_filtered \
  --output /out/topocalib_uc/results/fusion360_main/topocalib_uc_B5_budget100_seed0.json \
  --split-source random_filtered \
  --train-ratio 0.7 \
  --val-ratio 0.15 \
  --test-ratio 0.15 \
  --seed 0 \
  --min-parts-per-class 0 \
  --labeled-part-budget 100 \
  --ssrl-cache-dir /out/topocalib_uc/fusion360_ssrl_faces \
  --embedding-dim 256 \
  --hidden-dim 128 \
  --epochs 200 \
  --patience 20 \
  --lr 0.001 \
  --weight-decay 0.0001 \
  --tau 0.07 \
  --num-surface-types 5 \
  --variant B5 \
  --beta 0.10 \
  --part-top-k 5 \
  --lambda-cal 1.0 \
  --lambda-part 0.5 \
  --checkpoint-output /out/topocalib_uc/checkpoints/topocalib_uc_B5_budget100_seed0.pt
```

Different budget and seed combinations correspond to different values of `--labeled-part-budget`, `--seed`, and `--output`.

Checkpoint and inference:

The training entry point saves the downstream checkpoint. If `--checkpoint-output` is not explicitly provided, it is saved by default to the path with the same name as `--output` and the `.pt` suffix. The TopoCalib-UC / Frozen + Proto checkpoint contains adapter weights, class prototypes, the class list, the variant, and calibration hyperparameters; the Frozen + Linear checkpoint contains the adapter and linear head weights.

This command loads the Fusion360 TopoCalib-UC checkpoint and runs inference on all parts in the Fusion360 PartGraph cache. The output JSON contains the predicted labels for each part; if the cache contains GT labels, it also reports overall and per-part accuracy / mIoU.

```bash
python -m topocalib_uc.inference.run \
  --checkpoint /out/topocalib_uc/checkpoints/topocalib_uc_B5_budget100_seed0.pt \
  --cache-dir /out/topocalib_uc/fusion360_filtered \
  --ssrl-cache-dir /out/topocalib_uc/fusion360_ssrl_faces \
  --output /out/topocalib_uc/results/fusion360_main/topocalib_uc_B5_budget100_seed0_inference.json \
  --device cuda
```

This command runs inference only on specified parts. `--part-id` can be passed repeatedly; add `--save-logits` to save logits for each face.

```bash
python -m topocalib_uc.inference.run \
  --checkpoint /out/topocalib_uc/checkpoints/topocalib_uc_B5_budget100_seed0.pt \
  --cache-dir /out/topocalib_uc/fusion360_filtered \
  --ssrl-cache-dir /out/topocalib_uc/fusion360_ssrl_faces \
  --output /out/topocalib_uc/results/fusion360_main/topocalib_uc_B5_budget100_seed0_part_inference.json \
  --part-id PART_ID_TO_INFER \
  --save-logits \
  --device cuda
```

### 7. MFCAD++ Main Experiments

Input: MFCAD++ PartGraph and SSRL face representation. Output: one JSON result file per command.

This command runs the full TopoCalib-UC model on MFCAD++.

```bash
python -m topocalib_uc.train.run_frozen_proto \
  --cache-dir /out/topocalib_uc/mfcadpp_filtered \
  --output /out/topocalib_uc/results/mfcadpp_main/topocalib_uc_B5_budget100_seed0.json \
  --split-source random_filtered \
  --train-ratio 0.7 \
  --val-ratio 0.15 \
  --test-ratio 0.15 \
  --seed 0 \
  --min-parts-per-class 0 \
  --labeled-part-budget 100 \
  --ssrl-cache-dir /out/topocalib_uc/mfcadpp_ssrl_faces \
  --embedding-dim 256 \
  --hidden-dim 128 \
  --epochs 200 \
  --patience 20 \
  --lr 0.001 \
  --weight-decay 0.0001 \
  --tau 0.07 \
  --num-surface-types 5 \
  --variant B5 \
  --beta 0.10 \
  --part-top-k 5 \
  --lambda-cal 1.0 \
  --lambda-part 0.5
```

For Frozen + Linear and Frozen + Proto, use the corresponding commands from the previous section and replace `--cache-dir`, `--ssrl-cache-dir`, and `--output` with MFCAD++ paths. Use corresponding output files for different budget and seed combinations.

### 8. SSRL + MR-GCN

Input: PartGraph and SSRL face representation. Output: a JSON result file and a `.pt` checkpoint with the same base name.

This command runs the SSRL + MR-GCN baseline on Fusion360.

```bash
python -m baselines.ssrl_mrgcn.run \
  --cache-dir /out/topocalib_uc/fusion360_filtered \
  --ssrl-cache-dir /out/topocalib_uc/fusion360_ssrl_faces \
  --output /out/topocalib_uc/results/fusion360_ssrl_mrgcn/ssrl_mrgcn_budget100_seed0.json \
  --split-source random_filtered \
  --train-ratio 0.7 \
  --val-ratio 0.15 \
  --test-ratio 0.15 \
  --seed 0 \
  --min-parts-per-class 0 \
  --labeled-part-budget 100 \
  --hidden-dim 64 \
  --mp-layers 2 \
  --mlp-hidden-dim 64 \
  --dropout 0.0 \
  --epochs 200 \
  --patience 20 \
  --lr 0.001 \
  --weight-decay 0.0001 \
  --batch-part-count 16 \
  --undirected true
```

For MFCAD++, use the same command structure, replace the input and output paths with MFCAD++ paths, and set the corresponding budget and seed.

### 9. B0-B5 Ablation

Input: PartGraph and SSRL face representation. Output: one JSON file for each variant, budget, and seed.

This command runs the B0 ablation configuration on Fusion360.

```bash
python -m topocalib_uc.train.run_frozen_proto \
  --cache-dir /out/topocalib_uc/fusion360_filtered \
  --output /out/topocalib_uc/results/fusion360_ablation/proto_B0_budget1000_seed0.json \
  --split-source random_filtered \
  --train-ratio 0.7 \
  --val-ratio 0.15 \
  --test-ratio 0.15 \
  --seed 0 \
  --min-parts-per-class 0 \
  --labeled-part-budget 1000 \
  --ssrl-cache-dir /out/topocalib_uc/fusion360_ssrl_faces \
  --embedding-dim 256 \
  --hidden-dim 128 \
  --epochs 200 \
  --patience 20 \
  --lr 0.001 \
  --weight-decay 0.0001 \
  --tau 0.07 \
  --num-surface-types 5 \
  --variant B0 \
  --beta 0.10 \
  --part-top-k 5 \
  --lambda-cal 1.0 \
  --lambda-part 0.5
```

`B1`, `B2`, `B3`, `B4`, and `B5` use the corresponding `--variant` and `--output`. For MFCAD++, use the same command structure and replace the input and output directories with MFCAD++ directories.

### 10. External Baselines

The upstream code for external baselines is located in `baselines/external/*/upstream`, and the adaptation layers are located in `baselines/external/*/adapters`. Intermediate data, raw outputs, and unified results are stored in the `converted_data`, `raw_outputs`, and `results` directories, respectively.

BRepNet on Fusion360:

This command converts the Fusion360 PartGraph cache into the STEP, seg, and split files required by BRepNet.

```bash
python -m baselines.external.brepnet.adapters.prepare_brepnet_data \
  --partgraph-cache-dir /out/topocalib_uc/fusion360_filtered \
  --output-dir /out/topocalib_uc/external_baselines/brepnet/converted_data/fusion360_gallery \
  --dataset fusion360_gallery
```

This command enters the BRepNet upstream directory and extracts BRepNet training features from STEP files.

```bash
cd baselines/external/brepnet/upstream/BRepNet
python -m pipeline.extract_brepnet_data_from_step \
  --step_path /out/topocalib_uc/external_baselines/brepnet/converted_data/fusion360_gallery/steps \
  --output /out/topocalib_uc/external_baselines/brepnet/converted_data/fusion360_gallery/processed \
  --feature_list feature_lists/all.json \
  --seg_dir /out/topocalib_uc/external_baselines/brepnet/converted_data/fusion360_gallery/seg \
  --num_workers 8
```

This command builds the BRepNet dataset file from the BRepNet processed feature directory and the train/test split.

```bash
python -m pipeline.build_dataset_file \
  --npz_folder /out/topocalib_uc/external_baselines/brepnet/converted_data/fusion360_gallery/processed \
  --train_test /out/topocalib_uc/external_baselines/brepnet/converted_data/fusion360_gallery/train_test.json \
  --validation_split 0.2 \
  --dataset_file /out/topocalib_uc/external_baselines/brepnet/converted_data/fusion360_gallery/brepnet_dataset.json
```

This command trains BRepNet on the converted Fusion360 data.

```bash
python -m train.train \
  --dataset_file /out/topocalib_uc/external_baselines/brepnet/converted_data/fusion360_gallery/brepnet_dataset.json \
  --dataset_dir /out/topocalib_uc/external_baselines/brepnet/converted_data/fusion360_gallery/processed \
  --label_dir /out/topocalib_uc/external_baselines/brepnet/converted_data/fusion360_gallery/seg \
  --input_features feature_lists/all.json \
  --kernel kernels/winged_edge.json \
  --num_classes 8 \
  --max_epochs 200 \
  --batch_size 64 \
  --num_workers 8 \
  --log_dir /out/topocalib_uc/external_baselines/brepnet/raw_outputs/fusion360_gallery
```

UV-Net on Fusion360:

This command converts the Fusion360 PartGraph cache into the data directory required by UV-Net.

```bash
cd /workspace/TopoCalib-UC
python -m baselines.external.uvnet.adapters.prepare_uvnet_data \
  --partgraph-cache-dir /out/topocalib_uc/fusion360_filtered \
  --output-dir /out/topocalib_uc/external_baselines/uvnet/converted_data/fusion360_gallery \
  --dataset fusion360_gallery
```

This command converts STEP files into UV-Net graph data.

```bash
cd baselines/external/uvnet/upstream/UV-Net
python -m process.solid_to_graph \
  /out/topocalib_uc/external_baselines/uvnet/converted_data/fusion360_gallery/steps \
  /out/topocalib_uc/external_baselines/uvnet/converted_data/fusion360_gallery/graph \
  --num_processes 8
```

This command trains the UV-Net segmentation model on the converted Fusion360 data.

```bash
python segmentation.py train \
  --dataset fusiongallery \
  --dataset_path /out/topocalib_uc/external_baselines/uvnet/converted_data/fusion360_gallery \
  --max_epochs 100 \
  --batch_size 64 \
  --num_workers 8 \
  --experiment_name fusion360_gallery
```

AAGNet on MFCAD++:

This command converts the MFCAD++ PartGraph cache into the data directory required by AAGNet.

```bash
cd /workspace/TopoCalib-UC
python -m baselines.external.aagnet.adapters.prepare_aagnet_data \
  --partgraph-cache-dir /out/topocalib_uc/mfcadpp_filtered \
  --output-dir /out/topocalib_uc/external_baselines/aagnet/converted_data/mfcadpp \
  --dataset mfcadpp
```

This command extracts AAG representations from STEP files.

```bash
cd baselines/external/aagnet/upstream/AAGNet
python dataset/AAGExtractor.py \
  --step_path /out/topocalib_uc/external_baselines/aagnet/converted_data/mfcadpp/steps \
  --output /out/topocalib_uc/external_baselines/aagnet/converted_data/mfcadpp/aag \
  --num_workers 8
```

This command trains AAGNet on the converted MFCAD++ data.

```bash
cd /workspace/TopoCalib-UC
python -m baselines.external.aagnet.adapters.train_aagnet_from_config \
  --upstream-dir /workspace/TopoCalib-UC/baselines/external/aagnet/upstream/AAGNet \
  --dataset-dir /out/topocalib_uc/external_baselines/aagnet/converted_data/mfcadpp \
  --output-dir /out/topocalib_uc/external_baselines/aagnet/raw_outputs/mfcadpp \
  --run-name aagnet_seed0 \
  --epochs 100 \
  --batch-size 256 \
  --num-classes 25 \
  --device cuda \
  --seed 0 \
  --lr 0.01 \
  --weight-decay 0.01 \
  --num-threads 8
```

### 11. Result Tables and Mechanism Records

Each training, ablation, and comparison command above outputs independent JSON files, checkpoints, and related artifacts. Mechanism analysis and result tables are compiled from the output files of each experiment.


## Output Notes

The PartGraph cache contains the face labels, surface type, face adjacency, edge relation type, original path records, and manifest for each part. The HybridBrep directories contain the graph data required for SSRL training. The SSRL face representation directories contain face-level tensors aligned with PartGraph `part_id`. Downstream training results are recorded as individual JSON files containing the split, budget, seed, tail of the training process, validation metrics, and test metrics. External baselines first output their own raw files and are then converted into unified JSON through the adaptation layers. The values in tables and figures come from these result files and logs.
