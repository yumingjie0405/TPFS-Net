# TPFS-Net

Official code for **TPFS-Net: Topology-Preserving Feature Modeling and Stage Prototype Guidance for Skeleton-Based Action Recognition**.

TPFS-Net extends a spatiotemporal Hyper-GCN backbone with two training modules:

- **Topology-Preserving Feature Modeling (TPFM)** adds multi-scale heat-kernel structural-role encodings and regularizes learned joint relations with low-frequency subspace preservation and physical-skeleton anchoring.
- **Stage Prototype Guidance (SPG)** pools features into ordered, soft temporal stages and learns class-conditional stage prototypes with a diversity objective.

The auxiliary losses are used during training. At inference, the global classifier and prototype classifier probabilities are averaged.

## Reported Results

The paper reports the following five-run mean top-1 accuracies. Four-stream results fuse separately trained joint, bone, joint-motion, and bone-motion streams.

| Protocol | Four-stream accuracy |
| --- | ---: |
| NTU RGB+D 60 X-Sub | 93.42 +/- 0.07% |
| NTU RGB+D 60 X-View | 97.48 +/- 0.05% |
| NTU RGB+D 120 X-Sub | 89.76 +/- 0.08% |
| NTU RGB+D 120 X-Set | 91.18 +/- 0.08% |
| Northwestern-UCLA | 97.31 +/- 0.09% |

The joint-stream model reports 92.22 +/- 0.09% on NTU60 X-Sub with 1.20 M parameters and 1.65 GFLOPs. These are paper-reported results, not a promise that a single run will reproduce the five-run mean. Dataset files and trained checkpoints are not included in this repository.

## Requirements

- Python 3.9
- PyTorch 1.13.1 (use a build compatible with your CUDA installation for GPU training)
- Dependencies listed in [`requirements.txt`](requirements.txt)

Install the Python dependencies in an environment with the appropriate PyTorch build:

```bash
pip install -r requirements.txt
```

## Data Preparation

Download the skeleton datasets from their official sources and follow their terms of use:

- NTU RGB+D 60/120: [NTU RGB+D Action Recognition](https://rose1.ntu.edu.sg/dataset/actionRecognition)
- Northwestern-UCLA: [Cross-View Action dataset](https://www.math.ucla.edu/~zjian/actionrec.html)

Place the NTU skeleton archives under `data/nturgbd_raw/`, following the directory names expected by the preprocessing scripts. Then preprocess each split:

```bash
cd data/ntu
python get_raw_skes_data.py
python get_raw_denoised_data.py
python seq_transformation.py

cd ../ntu120
python get_raw_skes_data.py
python get_raw_denoised_data.py
python seq_transformation.py
```

The generated arrays are intentionally ignored by Git. Set `data_path` in the selected YAML configuration to the resulting dataset file if your local layout differs.

## Training

Example: joint stream on NTU RGB+D 60 X-Sub using the TPFS-Net v3 configuration:

```bash
python main.py \
  --config config/ts_hgcn_v3p1/nturgbd-cross-subject/ts_hgcn_v3_joint.yaml \
  --work-dir work_dir/tpfs-net/ntu60/xsub/joint \
  --device 0
```

The configuration files under `config/ts_hgcn_v3/` and `config/ts_hgcn_v3p1/` define the corresponding model variants and benchmark protocols. The `config/ablation/` directory contains joint-stream ablation and sensitivity configurations. Check each YAML file's dataset path, model arguments, and training settings before launching an experiment.

## Evaluation

Evaluate a trained checkpoint with its saved configuration:

```bash
python main.py \
  --config work_dir/tpfs-net/ntu60/xsub/joint/config.yaml \
  --work-dir work_dir/tpfs-net/ntu60/xsub/joint \
  --phase test \
  --save-score True \
  --weights work_dir/tpfs-net/ntu60/xsub/joint/best.pt \
  --device 0
```

Replace `best.pt` with the checkpoint filename produced by your run. For four-stream evaluation, train each modality separately and use `ensemble.py` with the corresponding experiment directories.

## Repository Layout

- `model/`: Hyper-GCN backbone and TPFS-Net modules (TPFM and SPG)
- `config/`: benchmark, model-size, and ablation configurations
- `feeders/`, `graph/`, `data/`: input pipelines, skeleton graphs, and dataset preprocessing
- `tools/`: experiment utilities, robustness evaluation, and paper-figure scripts
- `tests/`: lightweight tests that do not require dataset files or checkpoints

Dataset contents, checkpoints, generated figures, logs, manuscript drafts, and local IDE settings are excluded from version control.

## Acknowledgements

The Hyper-GCN backbone builds on the work in [Adaptive Hyper-Graph Convolution Network for Skeleton-based Human Action Recognition with Virtual Connections](https://arxiv.org/abs/2411.14796). This repository also draws on publicly available skeleton-recognition implementations and preprocessing methods; please consult the relevant source papers and upstream repositories when using or adapting those components.

## Paper

Qi Bai, Mingjie Yu, Ruijing Xu, and Lin Xia, “TPFS-Net: Topology-Preserving Feature Modeling and Stage Prototype Guidance for Skeleton-Based Action Recognition.” Use the final publication metadata and DOI when citing the published version.
