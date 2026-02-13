#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   PY=/opt/anaconda3/envs/medseg_unet/bin/python \
#   bash tools/run_kfold5_train_ts222.sh

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

PY="${PY:-python}"
export PYTHONPATH="$REPO"

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_ROOT="results/diag/kfold5_ts222_${STAMP}"
mkdir -p "$OUT_ROOT"

# 1) Preprocess (raw -> processed ts222)
RAW_ROOT="${RAW_ROOT:-$REPO/../Datasets/ISLES-2022}"
DERIV_ROOT="${DERIV_ROOT:-$REPO/../Datasets/ISLES-2022/derivatives}"
CSV_MAIN="${CSV_MAIN:-$REPO/data/splits/my_dataset_train_val_test.csv}"
PROC_OUT="${PROC_OUT:-$REPO/data/processed/my_dataset_ts222}"

if [[ ! -f "$PROC_OUT/preprocess_meta.json" ]]; then
  echo "[1/3] preprocess -> $PROC_OUT" | tee "$OUT_ROOT/01_preprocess.log"
  "$PY" tools/preprocess_isles2022_from_csv.py \
    --csv-path "$CSV_MAIN" \
    --raw-root "$RAW_ROOT" \
    --derivatives-root "$DERIV_ROOT" \
    --out-root "$PROC_OUT" \
    --target-spacing "2.0,2.0,2.0" \
    --crop-margin "8,8,4" \
    --log-path "$OUT_ROOT/preprocess_log.jsonl" \
    2>&1 | tee -a "$OUT_ROOT/01_preprocess.log"
else
  echo "[1/3] preprocess skipped (exists): $PROC_OUT" | tee "$OUT_ROOT/01_preprocess.log"
fi

# 2) Generate fold configs
KFOLD_DIR="${KFOLD_DIR:-$REPO/data/splits/kfold5_my_dataset}"
BASE_CFG="${BASE_CFG:-$REPO/configs/generated/_recipe_20251227/medseg_3d_unet_recipe_dwi_adc_fp50_dicebce_patch96_e200.yaml}"
CFG_OUT_DIR="$REPO/configs/generated/_kfold5_ts222_${STAMP}"

echo "[2/3] generate configs -> $CFG_OUT_DIR" | tee "$OUT_ROOT/02_make_configs.log"
"$PY" tools/make_kfold_train_configs.py \
  --base-config "$BASE_CFG" \
  --kfold-dir "$KFOLD_DIR" \
  --out-dir "$CFG_OUT_DIR" \
  --data-root "$PROC_OUT" \
  --k 5 \
  2>&1 | tee -a "$OUT_ROOT/02_make_configs.log"

# 3) Train 5 folds sequentially (robust)
# NOTE: this can take a long time.
CONFIGS=("$CFG_OUT_DIR"/*.yaml)

echo "[3/3] train queue (${#CONFIGS[@]} configs)" | tee "$OUT_ROOT/03_train_queue.log"
"$PY" tools/run_train_queue.py \
  --python "$PY" \
  --repo "$REPO" \
  --configs "${CONFIGS[@]}" \
  2>&1 | tee -a "$OUT_ROOT/03_train_queue.log"

echo "[done] $OUT_ROOT"
