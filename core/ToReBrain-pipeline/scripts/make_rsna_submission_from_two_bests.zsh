#!/usr/bin/env zsh
set -euo pipefail

if [[ $# -lt 3 ]]; then
  cat <<'USAGE'
Usage:
  ./scripts/make_rsna_submission_from_two_bests.zsh <RSNA_ROOT> <RUNS_BASE_DIR_1> <RUNS_BASE_DIR_2> [OUT_CSV] [DEVICE]

Args:
  RSNA_ROOT        : RSNA dataset root containing stage_2_sample_submission.csv
                    (If using preprocessed DB, this can be <preprocessed_root>/rsna_meta and does NOT need stage_2_test/)
  RUNS_BASE_DIR_1  : dir containing one or more runs (or a single run dir)
  RUNS_BASE_DIR_2  : dir containing one or more runs (or a single run dir)
  OUT_CSV          : output csv path (default: submission_ensemble.csv)
  DEVICE           : TORCH_DEVICE (default: mps)

Notes:
  - Ensembles 2 models via `predict_rsna_ich_submission.py --model ARCH:CKPT ...`.
  - Requires both runs to have the same image_size/windows/stack_slices.
USAGE
  exit 2
fi

RSNA_ROOT="$1"
RUNS_BASE_1="$2"
RUNS_BASE_2="$3"
OUT_CSV="${4:-submission_ensemble.csv}"
DEVICE="${5:-mps}"

ROOT_DIR="/Users/yusukefujinami/ToReBrain/ToReBrain-pipeline"
cd "$ROOT_DIR"

export TORCH_DEVICE="$DEVICE"

# used by the embedded python snippet
export RSNA_ROOT OUT_CSV

# Optionally speed up smoke inference (leave unset for full inference)
MAX_TEST_IMAGES="${MAX_TEST_IMAGES:-}"

# Pick best from each base dir (JSON to avoid eval/quoting issues)
J1="$(python tools/rsna_select_best_run.py --runs-dir "$RUNS_BASE_1" --fmt json)"
J2="$(python tools/rsna_select_best_run.py --runs-dir "$RUNS_BASE_2" --fmt json)"

export J1 J2

python - <<'PY'
import json, os, subprocess, sys

def load(s: str):
    return json.loads(s)

j1 = load(os.environ['J1'])
j2 = load(os.environ['J2'])

keys = ['image_size','windows','preprocess','stack_slices']
for k in keys:
    if j1.get(k) != j2.get(k):
        raise SystemExit(f"Config mismatch for ensemble: {k} {j1.get(k)!r} != {j2.get(k)!r}")

cmd = [
    sys.executable,
    'predict_rsna_ich_submission.py',
    '--rsna-root', os.environ['RSNA_ROOT'],
    '--out-csv', os.environ['OUT_CSV'],
    '--image-size', str(j1['image_size']),
    '--windows', str(j1['windows']),
    '--preprocess', str(j1.get('preprocess','legacy')),
    '--stack-slices', str(j1['stack_slices']),
    '--batch-size', '32',
    '--input-normalize', 'none',
    '--enforce-any-max',
    '--model', f"{j1['arch']}:{j1['ckpt']}",
    '--model', f"{j2['arch']}:{j2['ckpt']}",
]

pre_root = os.environ.get('RSNA_PREPROCESSED_ROOT','').strip()
if pre_root:
    cmd += ['--preprocessed-root', pre_root]

max_n = os.environ.get('MAX_TEST_IMAGES','').strip()
if max_n:
    cmd += ['--max-test-images', max_n]

print('[best1]', j1['run_dir'], 'wlogloss', j1['best_val_wlogloss'], 'ckpt', j1['ckpt'])
print('[best2]', j2['run_dir'], 'wlogloss', j2['best_val_wlogloss'], 'ckpt', j2['ckpt'])
print('[run]', ' '.join(cmd))

subprocess.check_call(cmd)
PY