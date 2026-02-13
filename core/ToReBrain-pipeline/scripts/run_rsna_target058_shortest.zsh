#!/usr/bin/env zsh
set -euo pipefail

if [[ $# -lt 1 ]]; then
  cat <<'USAGE'
Usage:
  ./scripts/run_rsna_target058_shortest.zsh [RSNA_ROOT] [DEVICE]

Goal:
  Fastest path toward ~0.058 by training 2 strong 2D @384 models and ensembling.

Notes:
  - Uses slice split for speed (optimistic). After picking settings, do a final
    no-val run (val_frac=0) on more data/all data.
  - You can override SCALE via env vars below.

Env (optional):
  LIMIT_IMAGES   (default: 200000)
  EPOCHS         (default: 6)
  VAL_FRAC       (default: 0.02)
  LR             (default: 3e-4)

Example:
  export RSNA_ROOT="/Volumes/YSKFUJI's WDB 4TB/Datasets/rsna-intracranial-hemorrhage-detection"
  LIMIT_IMAGES=200000 EPOCHS=6 ./scripts/run_rsna_target058_shortest.zsh "$RSNA_ROOT" mps
USAGE
  exit 2
fi

RSNA_ROOT="${1:-${RSNA_ROOT:-}}"
if [[ -z "$RSNA_ROOT" ]]; then
  echo "ERROR: RSNA_ROOT is required. Provide as arg or set env RSNA_ROOT." >&2
  exit 2
fi
DEVICE="${2:-${TORCH_DEVICE:-mps}}"

ROOT_DIR="/Users/yusukefujinami/ToReBrain/ToReBrain-pipeline"
cd "$ROOT_DIR"

export TORCH_DEVICE="$DEVICE"

LIMIT_IMAGES="${LIMIT_IMAGES:-200000}"
EPOCHS="${EPOCHS:-6}"
VAL_FRAC="${VAL_FRAC:-0.02}"
LR="${LR:-3e-4}"

OUT_BASE="results/rsna_target058_shortest_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT_BASE"

COMMON=(
  --rsna-root "$RSNA_ROOT"
  --cache-dir "$OUT_BASE/cache_train"
  --limit-images "$LIMIT_IMAGES"
  --val-frac "$VAL_FRAC"
  --split-by slice
  --seed 0
  --epochs "$EPOCHS"
  --lr "$LR"
  --weight-decay 1e-4
  --windows '40,80;80,200;600,2800'
  --aug
  --scheduler
  --loss-any-weight 2.0
  --log-every-steps 200
  --image-size 384
  --stack-slices 1
  --num-workers 0
)

set -x

# A) EfficientNet-B0 (often very strong on RSNA)
python train_rsna_cnn2d_classifier.py train \
  "${COMMON[@]}" \
  --out-dir "$OUT_BASE/effb0_2d_img384_slicesplit_e${EPOCHS}" \
  --arch efficientnet_b0 \
  --pretrained \
  --preprocess gpt52 \
  --input-normalize auto \
  --batch-size 10 \
  --no-use-pos-weight \
  --no-use-sampler

# B) ConvNeXt-Tiny (complements effb0 well)
python train_rsna_cnn2d_classifier.py train \
  "${COMMON[@]}" \
  --out-dir "$OUT_BASE/convnext_tiny_2d_img384_slicesplit_e${EPOCHS}" \
  --arch convnext_tiny \
  --pretrained \
  --preprocess gpt52 \
  --input-normalize auto \
  --batch-size 6 \
  --no-use-pos-weight \
  --no-use-sampler

set +x

echo "[done] runs under: $OUT_BASE"

echo "[next] ensemble submission (full inference):"
cat <<'NEXT'
./scripts/make_rsna_submission_from_two_bests.zsh <RSNA_ROOT> <OUT_BASE>/effb0_2d_img384_slicesplit_e6 <OUT_BASE>/convnext_tiny_2d_img384_slicesplit_e6 submission_ensemble.csv mps

# smoke inference first:
MAX_TEST_IMAGES=200 ./scripts/make_rsna_submission_from_two_bests.zsh <RSNA_ROOT> <RUN_DIR_1> <RUN_DIR_2> submission_ensemble_smoke.csv mps
NEXT
