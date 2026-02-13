#!/usr/bin/env zsh
set -euo pipefail

if [[ $# -lt 2 ]]; then
  cat <<'USAGE'
Usage:
  ./scripts/run_rsna_effb0_384_target058.zsh <RSNA_ROOT> <OUT_DIR> [DEVICE]

Example:
  ./scripts/run_rsna_effb0_384_target058.zsh /path/to/rsna results/rsna_target058_live/effb0_2d_img384_slicesplit_e8 mps
USAGE
  exit 2
fi

RSNA_ROOT="$1"
OUT_DIR="$2"
DEVICE="${3:-mps}"

ROOT_DIR="/Users/yusukefujinami/ToReBrain/ToReBrain-pipeline"
cd "$ROOT_DIR"

export TORCH_DEVICE="$DEVICE"

python train_rsna_cnn2d_classifier.py train \
  --rsna-root "$RSNA_ROOT" \
  --out-dir "$OUT_DIR" \
  --cache-dir "$(dirname "$OUT_DIR")/cache_train" \
  --limit-images 60000 \
  --val-frac 0.1 \
  --split-by slice \
  --seed 0 \
  --epochs 8 \
  --arch efficientnet_b0 \
  --pretrained \
  --input-normalize auto \
  --image-size 384 \
  --windows '40,80;80,200;600,2800' \
  --stack-slices 1 \
  --batch-size 8 \
  --num-workers 0 \
  --scheduler \
  --aug \
  --use-pos-weight \
  --use-sampler \
  --sampler-pos-factor 3.0 \
  --loss-any-weight 2.0 \
  --log-every-steps 200
