#!/usr/bin/env bash
# Create a conda environment and install dependencies.
set -euo pipefail

SUB="$(cd "$(dirname "$0")/.." && pwd)"
ENV_NAME="${ENV_NAME:-hamm}"
PYTHON_VER="${PYTHON_VER:-3.10}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu121}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda was not found. Please install Miniconda or Anaconda first."
  echo "  https://docs.conda.io/en/latest/miniconda.html"
  exit 1
fi

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "conda env already exists: $ENV_NAME (skip creation and install packages)"
else
  echo "Creating conda env: $ENV_NAME (python=$PYTHON_VER)"
  conda create -n "$ENV_NAME" "python=${PYTHON_VER}" -y
fi

conda activate "$ENV_NAME"

echo "Installing PyTorch (index=$TORCH_INDEX) ..."
pip install -U pip
pip install torch torchvision --index-url "$TORCH_INDEX"

echo "Installing remaining dependencies ..."
pip install -r "$SUB/requirements.txt"

echo "Checking imports ..."
export PYTHONPATH="$SUB/code"
python - <<'PY'
import torch
import torchvision
import timm
import sklearn
import scipy
import numpy as np
from PIL import Image
from tqdm import tqdm
from task_vectors import TaskVector
from model_vit import RETFound_mae
from retfound_dataset import build_dataset
from third_party import aug
print(
    "ok\n"
    f"  torch={torch.__version__}  torchvision={torchvision.__version__}\n"
    f"  timm={timm.__version__}  sklearn={sklearn.__version__}  scipy={scipy.__version__}\n"
    f"  numpy={np.__version__}  cuda={torch.cuda.is_available()}"
)
PY

cat <<EOF

Environment ready: $ENV_NAME

Usage:
  conda activate $ENV_NAME
  export ADA_ROOT=/path/to/your/data_root   # contains data/, c_meh/, etc.
  cd $SUB
  bash scripts/run_eval.sh 0 16 eval

For CPU-only environments, reinstall PyTorch with:
  TORCH_INDEX=https://download.pytorch.org/whl/cpu bash scripts/setup_env.sh

EOF
