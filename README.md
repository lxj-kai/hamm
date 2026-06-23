# HAMM Code and Model

This repository contains the code, evaluation scripts, training scripts, and released `mergeweight` parameters for the HAMM model.

HAMM uses a RETFound ViT-L backbone, task vectors from RETFound fine-tuned checkpoints, and learned model-merging weights for multi-task retinal disease recognition. The repository does not include RETFound datasets, the pretrained backbone, or fine-tuned checkpoints. These resources must be downloaded from their original sources and configured locally before running evaluation or training.

## Repository Layout

```text
HAMM/
├── LICENSE
├── README.md
├── requirements.txt
├── code/
│   ├── train_merge.py              # train HAMM mergeweight
│   ├── eval_from_ckpt.py           # evaluate a saved mergeweight checkpoint
│   ├── export_merged_backbone.py   # export a fully merged backbone checkpoint
│   ├── eval_from_merged_backbone.py # evaluate a fully merged backbone checkpoint
│   ├── csv_to_mergeweight_ckpt.py  # convert CSV mergeweight to PyTorch checkpoint
│   ├── task_vectors.py             # task-vector construction
│   ├── model_vit.py                # RETFound ViT-L backbone definition
│   ├── retfound_dataset.py         # ImageFolder dataset loader
│   └── third_party.py              # AugMix utilities
├── scripts/
│   ├── setup_env.sh
│   ├── run_eval.sh
│   ├── export_merged.sh
│   ├── run_eval_merged.sh
│   └── run_train.sh
├── model/
│   ├── mergeweight_final.pth       # released HAMM mergeweight, shape 294 x 8
│   └── Retina_final_mergeweight.csv
├── config/
│   ├── run_args.json               # local paths for data and checkpoints
│   └── percentiles_output.txt      # AugMix percentile thresholds
├── results/                        # local evaluation outputs
├── logs/                           # local runtime logs
└── runs/                           # local training outputs
```

## Environment

Recommended environment:

- Python 3.10
- CUDA-enabled GPU
- Linux, WSL, or a Linux server
- Approximately 22GB GPU memory for training

`torch` and `torchvision` are not pinned in `requirements.txt` because they should match the local CUDA version.

### Manual Setup

```bash
conda create -n hamm python=3.10 -y
conda activate hamm

pip install -U pip

# CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# CUDA 11.8 alternative:
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

pip install -r requirements.txt
```

Check the installation:

```bash
export PYTHONPATH=$PWD/code
python - <<'PY'
import torch, torchvision, timm, sklearn, scipy, numpy as np
from PIL import Image
from tqdm import tqdm
from task_vectors import TaskVector
from model_vit import RETFound_mae
from retfound_dataset import build_dataset
from third_party import aug

print("ok")
print(f"torch={torch.__version__}, torchvision={torchvision.__version__}")
print(f"timm={timm.__version__}, sklearn={sklearn.__version__}, scipy={scipy.__version__}")
print(f"numpy={np.__version__}, cuda={torch.cuda.is_available()}")
PY
```

### Setup Script

```bash
bash scripts/setup_env.sh
conda activate hamm
```

The setup script defaults to Python 3.10 and the CUDA 12.1 PyTorch wheels. To use another CUDA version:

```bash
ENV_NAME=myenv TORCH_INDEX=https://download.pytorch.org/whl/cu118 bash scripts/setup_env.sh
```

## External Resources

The following files are required but not distributed in this repository:

| Resource | Source | Purpose |
|---|---|---|
| Data splits for the 8 benchmark datasets | RETFound benchmark release | evaluation and training |
| 8 RETFound fine-tuned checkpoints | RETFound benchmark release | task-vector construction and classifier heads |
| RETFound pretrained ViT-L checkpoint | RETFound official instructions | base model for merging |

Relevant links:

- RETFound paper: [A foundation model for generalizable disease detection from retinal images](https://doi.org/10.1038/s41586-023-06555-x)
- RETFound repository: [rmaphoh/RETFound](https://github.com/rmaphoh/RETFound)
- RETFound benchmark resources: [BENCHMARK.md](https://github.com/rmaphoh/RETFound/blob/main/BENCHMARK.md)

The RETFound pretrained checkpoints are external resources and are not redistributed in this repository. Please obtain them by following the official RETFound instructions and the original license terms.

## Path Configuration

After downloading the external resources, organize them as follows:

```text
HAMM_resources/
├── data/
│   ├── APTOS2019/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── ...
├── c_meh/
│   ├── RETFound_mae_meh-APTOS2019/
│   │   └── checkpoint-nohead.pth
│   └── ...
├── new/
│   └── ViT-L-16/
│       ├── head_APTOS2019.pt
│       └── ...
└── checkpoint_pretrained_fc_norm.pth
```

Before evaluation or training, configure the external-resource paths using one of the following two options.

Option 1: edit `config/run_args.json` directly:

```json
{
  "data_path": "/path/to/HAMM_resources/data/",
  "pretrained_checkpoint_path": "/path/to/HAMM_resources/checkpoint_pretrained_fc_norm.pth",
  "checkpoints_path": "/path/to/HAMM_resources/c_meh",
  "head_path": "/path/to/HAMM_resources/new/ViT-L-16"
}
```

Option 2: keep the `/path/to/HAMM_resources` placeholders in `config/run_args.json` and set `ADA_ROOT` before running scripts or Python entry points:

```bash
export ADA_ROOT=/path/to/HAMM_resources
```

`ADA_ROOT` must point to the directory that contains `data/`, `c_meh/`, `new/`, and `checkpoint_pretrained_fc_norm.pth`. If neither `config/run_args.json` nor `ADA_ROOT` is configured, evaluation will fail because the external resources are not included in this repository.

The variable name `ADA_ROOT` is kept for compatibility with the original scripts.

## Evaluation

```bash
conda activate hamm
cd hamm
bash scripts/run_eval.sh 0 16 eval
```

Arguments:

```text
bash scripts/run_eval.sh <gpu_id> <batch_size> <tag>
```

The script writes logs to:

```text
logs/eval_eval_nohup.log
```

and metrics to:

```text
results/metrics_eval.csv
```

The evaluation script can also be run directly:

```bash
export PYTHONPATH=$PWD/code
python code/eval_from_ckpt.py \
  --config config/run_args.json \
  --ckpt model/mergeweight_final.pth \
  --gpu 0 \
  --batch_size 16 \
  --tag eval
```

## Exporting and Evaluating a Merged Backbone

The released `model/mergeweight_final.pth` stores only the HAMM merging weights. To materialize a merged RETFound backbone checkpoint locally:

```bash
bash scripts/export_merged.sh
```

By default, this writes:

```text
model/merged_backbone_final.pth
```

This file is large and is intentionally not tracked by git. If a pre-exported merged backbone checkpoint is provided separately, place it at `model/merged_backbone_final.pth` or pass its path explicitly.

A pre-exported merged backbone checkpoint is available from the GitHub release:

- Release: https://github.com/lxj-kai/hamm/releases/tag/v1.0-merged-backbone
- Direct file: https://github.com/lxj-kai/hamm/releases/download/v1.0-merged-backbone/merged_backbone_final.pth

Download it into `model/`:

```bash
mkdir -p model
wget -O model/merged_backbone_final.pth \
  https://github.com/lxj-kai/hamm/releases/download/v1.0-merged-backbone/merged_backbone_final.pth
```

or:

```bash
mkdir -p model
curl -L \
  -o model/merged_backbone_final.pth \
  https://github.com/lxj-kai/hamm/releases/download/v1.0-merged-backbone/merged_backbone_final.pth
```

Checkpoint information:

```text
file: model/merged_backbone_final.pth
size: 1,213,299,314 bytes
sha256: 7ebd384739be905068c50bebfff5c3ea5d85330bdd6ccac13eb1460dffc8099a
```

Evaluate a merged backbone checkpoint:

```bash
bash scripts/run_eval_merged.sh 0 16 merged_backbone model/merged_backbone_final.pth
```

or run the Python entry point directly:

```bash
export PYTHONPATH=$PWD/code
python code/eval_from_merged_backbone.py \
  --config config/run_args.json \
  --backbone model/merged_backbone_final.pth \
  --gpu 0 \
  --batch_size 16 \
  --tag merged_backbone
```

## Training

The training script optimizes only the HAMM `mergeweight`; it does not retrain the backbone or dataset-specific classifier heads.

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_train.sh train_run
```

Outputs are written to:

```text
runs/train_run/
```

The final mergeweight checkpoint is saved as:

```text
runs/train_run/ckpt/mergeweight_final.pth
```

## Released Model

We release one HAMM model. The table below reports the AUC values reorganized from the original per-dataset experimental records.

| Model | Avg. | APTOS2019 | OCTID | Glaucoma | IDRID | JSIEC | MESSIDOR2 | PAPILA | Retina | Checkpoints |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| HAMM (ours) | **0.8633** | 0.9530 | 0.9497 | 0.9375 | 0.7698 | 0.9718 | 0.7835 | 0.7677 | 0.7735 | [`mergeweight_final.pth`](model/mergeweight_final.pth), [`merged_backbone_final.pth`](https://github.com/lxj-kai/hamm/releases/download/v1.0-merged-backbone/merged_backbone_final.pth) |

Checkpoint files:

| File | Size | Purpose | How to use |
|---|---:|---|---|
| `model/mergeweight_final.pth` | 20 KB | learned HAMM merging weights, shape `294 x 8` | evaluate with `code/eval_from_ckpt.py`; requires local RETFound pretrained checkpoint, fine-tuned checkpoints, classifier heads, and datasets |
| [`merged_backbone_final.pth`](https://github.com/lxj-kai/hamm/releases/download/v1.0-merged-backbone/merged_backbone_final.pth) | 1,213,299,314 bytes | pre-exported merged RETFound backbone | download from the [GitHub release](https://github.com/lxj-kai/hamm/releases/tag/v1.0-merged-backbone) or export locally with `scripts/export_merged.sh`; still requires classifier heads and datasets for evaluation |

The small `mergeweight_final.pth` file is not a full ViT-L checkpoint. It stores only the learned HAMM merging weights. The pre-exported `merged_backbone_final.pth` is provided separately for users who prefer to evaluate without dynamically reconstructing the merged backbone.

## Evaluation Tasks

| Dataset | Modality | Task | Classes |
|---|---|---|---:|
| APTOS2019 | CFP | DR grading | 5 |
| OCTID | OCT | retinal/OCT disease classification | 5 |
| Glaucoma_fundus | CFP | glaucoma grading | 3 |
| IDRID | CFP | DR grading | 5 |
| JSIEC | CFP | multi-class ocular disease classification | 39 |
| MESSIDOR2 | CFP | DR grading | 5 |
| PAPILA | CFP | glaucoma screening | 3 |
| Retina | CFP | retinal abnormality classification | 4 |

## CSV to Checkpoint

```bash
python code/csv_to_mergeweight_ckpt.py \
  --csv model/Retina_final_mergeweight.csv \
  --out model/mergeweight_final.pth
```

## Notes

- The repository intentionally excludes RETFound datasets, pretrained checkpoints, and fine-tuned checkpoints.
- Large merged backbone checkpoints are distributed separately and ignored by git.
- Local metrics, logs, and training outputs are ignored by `.gitignore`.
- `scripts/*.sh` use bash, `nohup`, and Linux-style environment variables.
- Install `torch` and `torchvision` according to the local CUDA version.
- RETFound data and model checkpoints should be used according to their original license terms.
- The code in this repository is released under the MIT License. External RETFound resources are governed by their own license terms.
