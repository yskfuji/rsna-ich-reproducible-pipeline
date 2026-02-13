#!/usr/bin/env zsh
set -euo pipefail

if [[ $# -lt 1 ]]; then
  cat <<'USAGE'
Usage:
  ./scripts/run_rsna_target058.zsh [RSNA_ROOT] [DEVICE]

Runs a small set of stronger P1 experiments aimed at ~0.058 Kaggle wlogloss.
(We still select by internal val_logloss_weighted; leaderboard may differ.)

Args:
  RSNA_ROOT : RSNA dataset root containing stage_2_train.csv and stage_2_train/
  DEVICE   : TORCH_DEVICE (default: mps)

Example:
  export RSNA_ROOT="/Volumes/YSKFUJI's WDB 4TB/Datasets/rsna-intracranial-hemorrhage-detection"
  ./scripts/run_rsna_target058.zsh "$RSNA_ROOT" mps
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

OUT_BASE="results/rsna_target058_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT_BASE"

COMMON=(
  --rsna-root "$RSNA_ROOT"
  --cache-dir "$OUT_BASE/cache_train"
  --val-frac 0.1
  --seed 0
  --epochs 8
  --windows '40,80;80,200;600,2800'
  --aug
  --scheduler
  --use-pos-weight
  --use-sampler
  --sampler-pos-factor 3.0
  --loss-any-weight 2.0
  --log-every-steps 200
)

# NOTE:
# - img384 is usually a big win for RSNA.
# - batch sizes may need tuning on MPS depending on memory.

set -x

# 1) EfficientNet-B0 @384 (slice split: faster + often strong; optimistic val)
python train_rsna_cnn2d_classifier.py train \
  "${COMMON[@]}" \
  --out-dir "$OUT_BASE/effb0_2d_img384_slicesplit_e8" \
  --limit-images 60000 \
  --split-by slice \
  --arch efficientnet_b0 \
  --pretrained \
  --input-normalize auto \
  --image-size 384 \
  --stack-slices 1 \
  --batch-size 8 \
  --num-workers 0

# 2) EfficientNet-B0 @384 (study split: more realistic)
python train_rsna_cnn2d_classifier.py train \
  "${COMMON[@]}" \
  --out-dir "$OUT_BASE/effb0_2d_img384_splitstudy_e8" \
  --limit-images 60000 \
  --split-by study \
  --arch efficientnet_b0 \
  --pretrained \
  --input-normalize auto \
  --image-size 384 \
  --stack-slices 1 \
  --batch-size 8 \
  --num-workers 0

# 3) ConvNeXt-Tiny @384 (slice split)
python train_rsna_cnn2d_classifier.py train \
  "${COMMON[@]}" \
  --out-dir "$OUT_BASE/convnext_tiny_2d_img384_slicesplit_e8" \
  --limit-images 60000 \
  --split-by slice \
  --arch convnext_tiny \
  --pretrained \
  --input-normalize auto \
  --image-size 384 \
  --stack-slices 1 \
  --batch-size 6 \
  --num-workers 0

# 4) ConvNeXt-Tiny @384 (study split)
python train_rsna_cnn2d_classifier.py train \
  "${COMMON[@]}" \
  --out-dir "$OUT_BASE/convnext_tiny_2d_img384_splitstudy_e8" \
  --limit-images 60000 \
  --split-by study \
  --arch convnext_tiny \
  --pretrained \
  --input-normalize auto \
  --image-size 384 \
  --stack-slices 1 \
  --batch-size 6 \
  --num-workers 0

set +x

echo "[done] runs under: $OUT_BASE"

echo "[next] pick best + make submission (optionally ensemble):"
cat <<'NEXT'
# single best
python tools/rsna_select_best_run.py --runs-dir <OUT_BASE> --fmt json

# submission from best (full inference)
./scripts/make_rsna_submission_from_best.zsh <RSNA_ROOT> <OUT_BASE> submission.csv mps

# for smoke inference first:
MAX_TEST_IMAGES=200 ./scripts/make_rsna_submission_from_best.zsh <RSNA_ROOT> <OUT_BASE> submission_smoke.csv mps
NEXT
