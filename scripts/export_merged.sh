#!/bin/bash
# Export the merged RETFound backbone checkpoint.
set -euo pipefail

SUB="$(cd "$(dirname "$0")/.." && pwd)"
ADA="${ADA_ROOT:-$(cd "$SUB/.." && pwd)}"
CKPT="${1:-$SUB/model/mergeweight_final.pth}"
OUT="${2:-$SUB/model/merged_backbone_final.pth}"
PYTHON="${PYTHON:-python3}"

export ADA_ROOT="$ADA"
export PYTHONPATH="$SUB/code"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"

echo "SUB=$SUB ADA_ROOT=$ADA"
echo "ckpt=$CKPT"
echo "out=$OUT"
"$PYTHON" "$SUB/code/export_merged_backbone.py" \
  --ckpt "$CKPT" \
  --out "$OUT" \
  --device cpu
