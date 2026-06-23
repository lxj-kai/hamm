#!/bin/bash
# Train merge weights. A dedicated GPU with about 22GB memory is recommended.
set -euo pipefail

SUB="$(cd "$(dirname "$0")/.." && pwd)"
ADA="${ADA_ROOT:-$(cd "$SUB/.." && pwd)}"
GPU="${CUDA_VISIBLE_DEVICES:-6}"
RUN_NAME="${1:-max_$(date +%Y%m%d_%H%M%S)}"
PYTHON="${PYTHON:-python3}"
SEED="${SEED:--1}"

export CUDA_VISIBLE_DEVICES="$GPU"
export ADA_ROOT="$ADA"
export PYTHONPATH="$SUB/code"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

RUN_DIR="$SUB/runs/$RUN_NAME"
mkdir -p "$RUN_DIR/logs"

echo "SUB=$SUB GPU=$GPU RUN=$RUN_NAME SEED=$SEED"
nohup env ADA_ROOT="$ADA" PYTHONPATH="$SUB/code" "$PYTHON" "$SUB/code/train_merge.py" \
  --percentile_threshold "30%" \
  --run_root "$SUB/runs" \
  --run_name "$RUN_NAME" \
  --num_workers 4 \
  --seed "$SEED" \
  > "$RUN_DIR/logs/nohup_output.log" 2>&1 &

echo "PID=$!"
echo "run_dir=$RUN_DIR"
echo "tail -f $RUN_DIR/logs/nohup_output.log"
