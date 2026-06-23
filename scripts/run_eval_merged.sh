#!/bin/bash
# Evaluate a pre-exported merged_backbone_final.pth checkpoint.
set -euo pipefail

SUB="$(cd "$(dirname "$0")/.." && pwd)"
ADA="${ADA_ROOT:-$(cd "$SUB/.." && pwd)}"
GPU="${1:-0}"
BS="${2:-16}"
TAG="${3:-merged_backbone}"
BACKBONE="${4:-$SUB/model/merged_backbone_final.pth}"
PYTHON="${PYTHON:-python3}"
LOG="$SUB/logs/eval_${TAG}_nohup.log"

mkdir -p "$SUB/logs" "$SUB/results"
export ADA_ROOT="$ADA"
export PYTHONPATH="$SUB/code"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "SUB=$SUB ADA_ROOT=$ADA GPU=$GPU BS=$BS BACKBONE=$BACKBONE"
nohup "$PYTHON" "$SUB/code/eval_from_merged_backbone.py" \
  --gpu "$GPU" \
  --batch_size "$BS" \
  --tag "$TAG" \
  --backbone "$BACKBONE" \
  > "$LOG" 2>&1 &

echo "PID=$!"
echo "tail -f $LOG"
