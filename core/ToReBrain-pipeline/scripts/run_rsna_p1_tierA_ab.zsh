#!/usr/bin/env zsh
set -euo pipefail

if [[ $# -lt 1 ]]; then
  cat <<'USAGE'
Usage:
  ./scripts/run_rsna_p1_tierA_ab.zsh <RSNA_ROOT> [OUT_BASE] [DEVICE]

Args:
  RSNA_ROOT : RSNA dataset root (contains stage_2_train.csv and stage_2_train/)
  OUT_BASE  : base output dir (default: results/rsna_p1)
  DEVICE    : TORCH_DEVICE (default: mps; use cpu/cuda if available)

Example:
  ./scripts/run_rsna_p1_tierA_ab.zsh ~/Datasets/rsna-ich results/rsna_p1 mps
USAGE
  exit 2
fi

RSNA_ROOT="$1"
OUT_BASE="${2:-results/rsna_p1}"
DEVICE="${3:-mps}"

ROOT_DIR="/Users/yusukefujinami/ToReBrain/ToReBrain-pipeline"
cd "$ROOT_DIR"

export TORCH_DEVICE="$DEVICE"

# Common knobs (tune if you have more time/compute)
LIMIT_IMAGES=20000
VAL_FRAC=0.1
SEED=0
EPOCHS=5
IMAGE_SIZE_2D=256
IMAGE_SIZE_25D=384
WINDOWS='40,80;80,200;600,2800'

# On-disk cache (decoded+windowed+resized tensors)
CACHE_DIR="$OUT_BASE/cache_slices"

mkdir -p "$OUT_BASE"

run_one() {
  local name="$1"; shift
  local out_dir="$OUT_BASE/$name"
  mkdir -p "$out_dir"
  echo "[run] $name -> $out_dir"
  python train_rsna_cnn2d_classifier.py train \
    --rsna-root "$RSNA_ROOT" \
    --out-dir "$out_dir" \
    --cache-dir "$CACHE_DIR" \
    --limit-images "$LIMIT_IMAGES" \
    --val-frac "$VAL_FRAC" \
    --split-by study \
    --seed "$SEED" \
    --epochs "$EPOCHS" \
    --windows "$WINDOWS" \
    --enforce-any-max \
    --input-normalize auto \
    "$@"
}

# A/B: pos_weight vs sampler (2D baseline)
run_one "p1_2d_resnet18_posw" \
  --arch resnet18 --pretrained \
  --image-size "$IMAGE_SIZE_2D" --stack-slices 1 \
  --use-pos-weight --no-use-sampler

run_one "p1_2d_resnet18_sampler" \
  --arch resnet18 --pretrained \
  --image-size "$IMAGE_SIZE_2D" --stack-slices 1 \
  --no-use-pos-weight --use-sampler --sampler-pos-factor 3.0

# Upgrade: 2.5D + stronger backbone (EffNet-B0)
run_one "p1_25d_effb0_posw" \
  --arch efficientnet_b0 --pretrained \
  --image-size "$IMAGE_SIZE_25D" --stack-slices 3 \
  --use-pos-weight --no-use-sampler

run_one "p1_25d_effb0_sampler" \
  --arch efficientnet_b0 --pretrained \
  --image-size "$IMAGE_SIZE_25D" --stack-slices 3 \
  --no-use-pos-weight --use-sampler --sampler-pos-factor 3.0

echo "[done] Results under: $OUT_BASE"
echo "Check each run's val_logloss_weighted in: <run>/log.jsonl (lower is better)"
