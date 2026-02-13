#!/usr/bin/env zsh
set -euo pipefail

if [[ $# -lt 2 ]]; then
  cat <<'USAGE'
Usage:
  ./scripts/make_rsna_submission_from_best.zsh <RSNA_ROOT> <RUNS_BASE_DIR> [OUT_CSV] [DEVICE]

Args:
  RSNA_ROOT       : RSNA dataset root containing stage_2_test/ and stage_2_sample_submission.csv
  RUNS_BASE_DIR   : directory containing multiple run subdirs (each has meta.json/log.jsonl)
  OUT_CSV         : output csv path (default: submission.csv)
  DEVICE          : TORCH_DEVICE (default: mps)

Example:
  ./scripts/make_rsna_submission_from_best.zsh ~/Datasets/rsna-ich results/rsna_p1 submission.csv mps
USAGE
  exit 2
fi

RSNA_ROOT="$1"
RUNS_BASE="$2"
OUT_CSV="${3:-submission.csv}"
DEVICE="${4:-mps}"

ROOT_DIR="/Users/yusukefujinami/ToReBrain/ToReBrain-pipeline"
cd "$ROOT_DIR"

export TORCH_DEVICE="$DEVICE"

echo "[select] picking best run under: $RUNS_BASE"
eval "$(python tools/rsna_select_best_run.py --runs-dir "$RUNS_BASE" --fmt shell)"

echo "[best] dir=$BEST_RUN_DIR wlogloss=$BEST_VAL_WLOGLOSS"
echo "[best] ckpt=$BEST_CKPT"
echo "[best] arch=$BEST_ARCH image_size=$BEST_IMAGE_SIZE stack=$BEST_STACK_SLICES"

TEST_CACHE_DIR="$RUNS_BASE/cache_test_slices"

extra_args=()
if [[ -n "${MAX_TEST_IMAGES:-}" ]]; then
  extra_args+=(--max-test-images "$MAX_TEST_IMAGES")
fi

python predict_rsna_ich_submission.py \
  --rsna-root "$RSNA_ROOT" \
  --ckpt "$BEST_CKPT" \
  --arch "$BEST_ARCH" \
  --image-size "$BEST_IMAGE_SIZE" \
  --windows "$BEST_WINDOWS" \
  --stack-slices "$BEST_STACK_SLICES" \
  --cache-dir "$TEST_CACHE_DIR" \
  --out-csv "$OUT_CSV" \
  --batch-size 32 \
  --input-normalize auto \
  "${extra_args[@]}" \
  --enforce-any-max

echo "[done] wrote: $OUT_CSV"
