#!/usr/bin/env bash
set -euo pipefail

# Wait k-fold training -> saveprobs+sweep per fold -> prob-average ensemble -> sweep.
# Designed for unstable terminals: this script writes a single timestamped log directory.

REPO_DIR="${REPO_DIR:-/Users/yusukefujinami/ToReBrain/ToReBrain-pipeline}"
PY="${PY:-/opt/anaconda3/envs/medseg_unet/bin/python}"

CSV_PATH="${CSV_PATH:-$REPO_DIR/data/splits/my_dataset_train_val_test.csv}"
DATA_ROOT="${DATA_ROOT:-$REPO_DIR/data/processed/my_dataset_ts222}"

# Prefer explicit CONFIGS_DIR. If omitted, pick the latest generated kfold dir.
CONFIGS_DIR="${CONFIGS_DIR:-}"
if [[ -z "$CONFIGS_DIR" ]]; then
  CONFIGS_DIR=$(ls -1dt "$REPO_DIR"/configs/generated/_kfold5_ts222_* 2>/dev/null | head -n 1 || true)
fi
if [[ -z "$CONFIGS_DIR" ]]; then
  echo "ERROR: CONFIGS_DIR is empty. Set CONFIGS_DIR to the folder containing fold YAMLs." >&2
  exit 2
fi

SPLIT="${SPLIT:-val}"
PATCH_SIZE="${PATCH_SIZE:-96,96,96}"
OVERLAP="${OVERLAP:-0.5}"
THRESHOLDS="${THRESHOLDS:-0.12,0.14,0.16,0.18,0.20,0.22,0.24,0.26,0.28,0.30,0.32,0.34,0.36,0.38,0.40,0.42,0.44,0.46,0.48,0.50,0.80}"
TTA="${TTA:-none}"
RESAMPLE_MAX_ZOOM_MM="${RESAMPLE_MAX_ZOOM_MM:-0.0}"
SLICE_SPACING_SOURCE="${SLICE_SPACING_SOURCE:-effective}"
FOCUS_BUCKET="${FOCUS_BUCKET:-all}"
POLL_SECONDS="${POLL_SECONDS:-120}"
WAIT_COMPLETE="${WAIT_COMPLETE:-0}"  # 1 to wait for epoch>=train.epochs

STAMP=$(date +%Y%m%d_%H%M%S)
OUT_DIR="$REPO_DIR/results/diag/kfold_ensemble_driver_ts222_$STAMP"
mkdir -p "$OUT_DIR"

echo "[info] REPO_DIR=$REPO_DIR" | tee "$OUT_DIR/driver.log"
echo "[info] PY=$PY" | tee -a "$OUT_DIR/driver.log"
echo "[info] CONFIGS_DIR=$CONFIGS_DIR" | tee -a "$OUT_DIR/driver.log"
echo "[info] CSV_PATH=$CSV_PATH" | tee -a "$OUT_DIR/driver.log"
echo "[info] DATA_ROOT=$DATA_ROOT" | tee -a "$OUT_DIR/driver.log"
echo "[info] SPLIT=$SPLIT" | tee -a "$OUT_DIR/driver.log"

cd "$REPO_DIR"

set +e
"$PY" -m py_compile tools/run_kfold_wait_eval_ensemble.py >>"$OUT_DIR/driver.log" 2>&1
PYCOMPILE_RC=$?
set -e
if [[ $PYCOMPILE_RC -ne 0 ]]; then
  echo "ERROR: py_compile failed; see $OUT_DIR/driver.log" >&2
  exit $PYCOMPILE_RC
fi

WAIT_FLAG=()
if [[ "$WAIT_COMPLETE" == "1" ]]; then
  WAIT_FLAG=(--wait-complete)
fi

# NOTE: For fold-ensemble to work, the case list for SPLIT must be identical across folds.
# With standard k-fold, val differs by fold; consider using SPLIT=test with a fixed test list.

"$PY" tools/run_kfold_wait_eval_ensemble.py \
  --python "$PY" \
  --repo "$REPO_DIR" \
  --configs-dir "$CONFIGS_DIR" \
  --csv-path "$CSV_PATH" \
  --root "$DATA_ROOT" \
  --split "$SPLIT" \
  --patch-size "$PATCH_SIZE" \
  --overlap "$OVERLAP" \
  --thresholds "$THRESHOLDS" \
  --tta "$TTA" \
  --resample-max-zoom-mm "$RESAMPLE_MAX_ZOOM_MM" \
  --slice-spacing-source "$SLICE_SPACING_SOURCE" \
  --focus-bucket "$FOCUS_BUCKET" \
  --poll-seconds "$POLL_SECONDS" \
  "${WAIT_FLAG[@]}" \
  2>&1 | tee -a "$OUT_DIR/driver.log"

echo "[done] $OUT_DIR" | tee -a "$OUT_DIR/driver.log"
