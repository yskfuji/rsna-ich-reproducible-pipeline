#!/usr/bin/env bash
set -euo pipefail

# Make paths independent of the caller's current working directory.
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO_ROOT"

# Runs conditional cascade probmap generation (val/test) + evaluation.
# Usage:
#   bash tools/run_conditional_cascade_eval.sh [step]
# Steps:
#   all (default) | val | test | eval_val | eval_test | summarize
# Optional env overrides:
#   PY=... OUT=... CSV=... ROOT=... MODEL=... STAGE1_VAL=... STAGE1_TEST=...

PY=${PY:-/opt/anaconda3/envs/medseg_unet/bin/python}
CSV=${CSV:-data/splits/my_dataset_dwi_adc_flair_train_val_test.csv}
ROOT=${ROOT:-data/processed/my_dataset_dwi_adc_flair}
MODEL=${MODEL:-runs/3d_unet/medseg_3d_unet_e10_dwi_adc_flair_stage2_conditional_from_stage1probs/best.pt}
STAGE1_VAL=${STAGE1_VAL:-results/diag/cascade_stage1_20251225_150430/saveprobs_val/probs}
STAGE1_TEST=${STAGE1_TEST:-results/diag/cascade_stage1_20251225_150430/saveprobs_test/probs}
THRS=${THRS:-0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80}
PATCH_SIZE=${PATCH_SIZE:-56,56,24}
OVERLAP=${OVERLAP:-0.5}
NORM=${NORM:-nonzero_zscore}
TTA=${TTA:-none}
RESAMPLE_MAX_ZOOM_MM=${RESAMPLE_MAX_ZOOM_MM:-2.0}
FUSION=${FUSION:-max}
STAGE1_LOGIT_EPS=${STAGE1_LOGIT_EPS:-1e-4}
SLICE_SPACING_SOURCE=${SLICE_SPACING_SOURCE:-raw}
SLICE_SPACING_BINS_MM=${SLICE_SPACING_BINS_MM:-3.0}

STEP=${1:-all}

if [[ -z "${OUT:-}" ]]; then
  STAMP=$(date +%Y%m%d_%H%M%S)
  OUT=results/diag/cond_cascade_eval_${STAMP}
fi

mkdir -p "$OUT"

[[ -f "$MODEL" ]] || { echo "ERROR missing model: $MODEL"; exit 2; }
[[ -d "$STAGE1_VAL" ]] || { echo "ERROR missing stage1 val: $STAGE1_VAL"; exit 2; }
[[ -d "$STAGE1_TEST" ]] || { echo "ERROR missing stage1 test: $STAGE1_TEST"; exit 2; }

echo "[run] STEP=$STEP OUT=$OUT"

do_makeprob() {
  local split="$1"; local stage1_dir="$2"; local out_dir="$3"; local log="$4"
  PYTHONPATH=$PWD "$PY" tools/conditional_cascade_make_probmaps.py \
    --stage1-probs-dir "$stage1_dir" \
    --stage2-model "$MODEL" \
    --csv-path "$CSV" \
    --root "$ROOT" \
    --split "$split" \
    --normalize "$NORM" \
    --patch-size "$PATCH_SIZE" \
    --overlap "$OVERLAP" \
    --tta "$TTA" \
    --resample-max-zoom-mm "$RESAMPLE_MAX_ZOOM_MM" \
    --fusion "$FUSION" \
    --stage1-logit-eps "$STAGE1_LOGIT_EPS" \
    --skip-existing \
    --out-probs-dir "$out_dir" \
    >"$log" 2>&1
}

do_eval() {
  local split="$1"; local probs_dir="$2"; local out_dir="$3"; local log="$4"
  PYTHONPATH=$PWD "$PY" -m src.evaluation.evaluate_isles \
    --probs-dir "$probs_dir" \
    --csv-path "$CSV" \
    --root "$ROOT" \
    --split "$split" \
    --out-dir "$out_dir" \
    --patch-size "$PATCH_SIZE" \
    --overlap "$OVERLAP" \
    --thresholds "$THRS" \
    --min-size 0 \
    --cc-score none \
    --top-k 0 \
    --normalize "$NORM" \
    --tta "$TTA" \
    --slice-spacing-source "$SLICE_SPACING_SOURCE" \
    --slice-spacing-bins-mm "$SLICE_SPACING_BINS_MM" \
    --resample-max-zoom-mm "$RESAMPLE_MAX_ZOOM_MM" \
    --quiet \
    >"$log" 2>&1
}

summarize() {
  PYTHONPATH=$PWD "$PY" - <<PY
import json
from pathlib import Path

def best_row(p: Path):
    s=json.loads(p.read_text())
    row=max(s['per_threshold'], key=lambda d: float(d['mean_dice'] or 0.0))
    bs=row.get('by_slice_spacing',{})
    le=bs.get('le_3mm',{})
    gt=bs.get('gt_3mm',{})
    return {
        'best_thr': float(row['threshold']),
        'mean_dice': float(row['mean_dice'] or 0.0),
        'le_3mm': float(le.get('mean_dice') or 0.0) if le else None,
        'le_det': le.get('detection_rate_case') if le else None,
        'gt_3mm': float(gt.get('mean_dice') or 0.0) if gt else None,
        'gt_det': gt.get('detection_rate_case') if gt else None,
    }

out=Path("$OUT")
for split in ("val","test"):
    p=out/f"eval_{split}"/"summary.json"
    r=best_row(p)
    print(split, r)
PY
}

case "$STEP" in
  all)
    mkdir -p "$OUT/saveprobs_val/probs" "$OUT/saveprobs_test/probs"
    do_makeprob val "$STAGE1_VAL" "$OUT/saveprobs_val/probs" "$OUT/makeprob_val.log"
    do_makeprob test "$STAGE1_TEST" "$OUT/saveprobs_test/probs" "$OUT/makeprob_test.log"
    mkdir -p "$OUT/eval_val" "$OUT/eval_test"
    do_eval val "$OUT/saveprobs_val/probs" "$OUT/eval_val" "$OUT/eval_val.log"
    do_eval test "$OUT/saveprobs_test/probs" "$OUT/eval_test" "$OUT/eval_test.log"
    summarize
    ;;
  val)
    mkdir -p "$OUT/saveprobs_val/probs"
    do_makeprob val "$STAGE1_VAL" "$OUT/saveprobs_val/probs" "$OUT/makeprob_val.log"
    ;;
  test)
    mkdir -p "$OUT/saveprobs_test/probs"
    do_makeprob test "$STAGE1_TEST" "$OUT/saveprobs_test/probs" "$OUT/makeprob_test.log"
    ;;
  eval_val)
    mkdir -p "$OUT/eval_val"
    do_eval val "$OUT/saveprobs_val/probs" "$OUT/eval_val" "$OUT/eval_val.log"
    ;;
  eval_test)
    mkdir -p "$OUT/eval_test"
    do_eval test "$OUT/saveprobs_test/probs" "$OUT/eval_test" "$OUT/eval_test.log"
    ;;
  summarize)
    summarize
    ;;
  *)
    echo "ERROR: unknown step '$STEP'" >&2
    echo "Usage: bash tools/run_conditional_cascade_eval.sh [all|val|test|eval_val|eval_test|summarize]" >&2
    exit 2
    ;;
esac

echo "[done] $OUT"
