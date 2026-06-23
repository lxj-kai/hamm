#!/bin/bash
# Run evaluation from the repository root.
set -euo pipefail

SUB="$(cd "$(dirname "$0")/.." && pwd)"
ADA="${ADA_ROOT:-$(cd "$SUB/.." && pwd)}"
GPU="${1:-0}"
BS="${2:-16}"
TAG="${3:-eval}"
PYTHON="${PYTHON:-python3}"
LOG="$SUB/logs/eval_${TAG}_nohup.log"

mkdir -p "$SUB/logs" "$SUB/results"
export ADA_ROOT="$ADA"
export PYTHONPATH="$SUB/code"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "SUB=$SUB ADA_ROOT=$ADA GPU=$GPU BS=$BS"
nohup "$PYTHON" "$SUB/code/eval_from_ckpt.py" \
  --gpu "$GPU" \
  --batch_size "$BS" \
  --tag "$TAG" \
  --ckpt "$SUB/model/mergeweight_final.pth" \
  > "$LOG" 2>&1 &

echo "PID=$!"
echo "tail -f $LOG"
