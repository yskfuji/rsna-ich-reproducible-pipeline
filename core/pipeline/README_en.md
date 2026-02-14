# RSNA Intracranial Hemorrhage (Kaggle 2019) — Auditable, Reproducible Experiment README (Portfolio)

Japanese version: [README.md](README.md)

This folder (pipeline/) is a minimal pipeline to run RSNA ICH training/evaluation/inference in an **auditable, reproducible** way.
To help a reviewer decide quickly, this README prioritizes **reproducibility / split design (leakage prevention) / comparison axes / ablation**.

---

## TL;DR (what you can verify here)

- Reproducibility (this bundle): supported by recorded artifacts (`meta.json` / `log.jsonl`, including `split_stats`) and fixed CLI commands (e.g., `results/rsna_convnext25d_ft_repro_val05_short_20260210_101836_run1` and `_run2`).
- Source commit (upstream, reference): `9bc2684fe0b564c792025b52694f9c4fe1a0d32d` (commit id of the upstream repository at the time this code snapshot was exported).
- Source repo URL: this distribution does not include git metadata, so the upstream URL/history cannot be traced from this bundle alone (no `.git`).
- Distribution integrity (optional): generate a file manifest (paths + sha256) via `python tools/make_manifest.py` (e.g., `--out MANIFEST.sha256`) to fingerprint this bundle.

- **Robustness ("was it just luck?")**: Study-level **GroupKFold(5)** with `split_by=study` yields `val_logloss_weighted = 0.05346 ± 0.00624` (`epochs=1` / `kept=6852`, fast CV).
- **Reproducibility**: two runs with identical seed/args reproduce `split_stats` and `val_*` (observed exact match: `abs_diff=0` in this environment; generally verify with `abs_diff <= 1e-6` depending on backend).
- **Leakage-safe evaluation**: use a study-level **group split** (`split_by=study`) so the same Study never spans train/val.
- **Practical value**: audit scripts mechanically verify that train/val group intersection is 0.
- **Leakage audit status (2026-02-14)**: across 10 seeds, Study/Series intersections are 0, and with **pre-split tensor-hash dedup (default ON)**, exact-duplicate hash intersection between train/val is also 0.
- **Primary metric**: Kaggle-style weighted multi-label logloss (`val_logloss_weighted`); AUC is auxiliary.
- **Uncertainty / Calibration (10 seeds, `split_by=study`, dedup enabled, retrained with dropout)**:
  - Error-detection AUROC(any): **0.9424 ± 0.0190**
  - ECE(any): **0.0231 ± 0.0032**
  - Brier(any): **0.0209 ± 0.0054**
  - AURC(weighted logloss): **0.00837 ± 0.00214**
  - coverage=0.8 accuracy(any) gain: **+2.53 ± 0.89 pp**
  - vs. baseline (same protocol): ECE **-0.0091**, Brier **-0.0040**, AURC **-0.00139**, AUROC **+0.0046**
- **Temperature-scaling boundary (important)**: the uncertainty numbers in this README come from `--fit-temperature` on the same holdout val split (fit set = eval set), so ECE/Brier can be optimistic. For strict reporting, use a calib/eval split (or a separate holdout for eval).
- **Audit hook (subset identity)**: in addition to `split_stats` consistency, record sorted `image_id` SHA256 as `subset_fingerprint_sha256` in **`meta.json` (implemented)**. `tools/eval_rsna_uncertainty.py` also outputs `adopted_subset_fingerprint_sha256` / `val_subset_fingerprint_sha256` in its JSON.
  - `subset_fingerprint_sha256` definition: sort adopted `image_id` ascending, join with `\n`, take SHA256 of the UTF-8 string (hex)
  - `adopted_subset_*` is the full adopted subset after `limit_images` etc; `val_subset_*` is the `image_id` subset that ended up in validation

- Quick reproduce (holdout, 1 epoch):

```bash
# Quick reproduce (holdout, 1 epoch)
TORCH_DEVICE=mps python train_rsna_cnn2d_classifier.py train --rsna-root ... --preprocessed-root ... --out-dir results/repro_demo --limit-images 8000 --val-frac 0.05 --split-by study --seed 0 --epochs 1 --num-workers 0 --no-aug --no-scheduler --arch convnext_tiny --pretrained --init-from "results/rsna_convnext25d_final2_full_mps_nosamp_nopw_20260202_222608/best.pt"

# If --init-from is unavailable:
# - omit it (ImageNet-pretrained only), or
# - set it to results/**/best*.pt
```

  → outputs: `meta.json`, `log.jsonl`, checkpoints under `results/repro_demo/` (see §2.2 for the full command).
  For concrete paths, see §2.1 Data example (rsna_root / preprocessed_root).

---

## 1. Metric Summary (this report)

### 1.1 Robustness check via GroupKFold(5) CV (study-level)

To address the concern “validation might be too small and lucky”, we run **GroupKFold(5)** with `split_by=study` (StudyInstanceUID).

Settings (fast CV):
- `limit_images=8000` (after excluding missing keys in the preprocessed DB: `kept=6852`)
- `epochs=1` (direction-check fast CV)
- `init_from=best.pt` (fine-tune)
  - `best.pt` was saved on validation in another run (see §2.2). It was not selected based on this CV’s validation.

Rough denominators (val-side; mean±std across folds):
- `n_val_studies = 1178.0 ± 1.0`
- `n_val_images = 1370.4 ± 9.6`
- `pos_rate_any = 0.1496 ± 0.0030`

CV results (val):
- `val_logloss_weighted`: **0.05346 ± 0.00624** (min 0.04732 / max 0.06347)
- `val_auc_mean`: **0.98815 ± 0.00311** (min 0.98441 / max 0.99188)

Aggregation (reproducible definition):
- Extract `val_logloss_weighted` / `val_auc_mean` from the **last line** of each fold’s `log.jsonl`, then compute mean±std (and min/max) across folds.
- Use `tools/summarize_cv.py` to aggregate automatically (see §2.3).

Run directory:
- `results/rsna_convnext25d_ft_gkfold5_e1_lim8k_20260210_150447`

---

### 1.2 Bridging fast CV to longer training (learning curve on a representative fold)

Fast CV (`epochs=1`) still leaves the question “what happens if we train longer?”
As a representative example, we run fold0 with `epochs=5` and report the learning curve.

Run:
- `results/rsna_convnext25d_ft_gkfold5_fold0_epochs5_20260210_160037`

| epoch | val_logloss_weighted | val_auc_mean | val_loss_plain |
|---:|---:|---:|---:|
| 1 | 0.04732 | 0.98608 | 0.04313 |
| 2 | 0.04639 | 0.98695 | 0.04235 |
| 3 | 0.04629 | 0.98777 | 0.04230 |
| 4 | 0.04672 | 0.98843 | 0.04264 |
| 5 | 0.04772 | 0.98839 | 0.04368 |

Note:
- In this fold, the best `val_logloss_weighted` occurs at **epoch=3 (0.04629)**; later epochs worsen, so an early-stop style decision is reasonable.

---

### 1.3 Single holdout (val_frac=0.05) and reproducibility (run1 vs run2)

Runs:
- `results/rsna_convnext25d_ft_repro_val05_short_20260210_101836_run1`
- `results/rsna_convnext25d_ft_repro_val05_short_20260210_111323_run2`

Key metrics (val):
- `val_logloss_weighted = 0.0555525`
- `val_auc_mean = 0.9915841`

Reproducibility outcome:
- With identical seed/args, `split_stats` and `val_*` match **exactly (abs_diff=0)**.
  - Exact equality may depend on the backend/kernels; generally, verifying `abs_diff <= 1e-6` is safer.
  - On CPU/CUDA, enabling `--num-workers > 0` or AMP may break exact equality (≠ “not reproducible”).

---

### 1.4 Seed sweep (0/1/2): mean±std

Under the same settings (`val_frac=0.05`, `split_by=study`, `limit_images=8000`, `epochs=1`), we ran three seeds.

| seed | val_logloss_weighted | val_auc_mean |
|---:|---:|---:|
| 0 | 0.0555525 | 0.9915841 |
| 1 | 0.0461215 | 0.9943803 |
| 2 | 0.0600081 | 0.9808989 |
| mean ± std | 0.0538940 ± 0.0070903 | 0.9889545 ± 0.0071150 |

Notes:
- This is **validation**, not Kaggle Private LB.
- AUC measures ranking/separation; logloss measures probability calibration/fit; they may not move together.

---

## 2. How to Reproduce (one command)

### 2.1 Prerequisites

- Assumes macOS + `mps` (designed to also run on CPU/CUDA)
- Examples use a **preprocessed SQLite** (fast; does not depend on having DICOMs)

Required for determinism (recommended):
- `--num-workers 0`, `--no-aug`
- fixed `--seed`, identical inputs (CSV/SQLite)
- Expected reproducibility check: meta.json.split_stats matches AND abs_diff <= 1e-6 for val_* (exact match may depend on backend/kernels).

Pass criteria (reproducibility, recommended):
1. meta.json.split_stats matches
2. abs_diff <= 1e-6 (recommended) for all of: `val_loss`, `val_loss_plain`, `val_auc_mean`, `val_logloss_weighted`
3. audit_rsna_split.py reports n_group_intersection == 0 AND n_imageid_intersection == 0
4. `subset_fingerprint_sha256` matches (optional but recommended)

Tested environment (this report):
- Python 3.12.4
- PyTorch 2.6.0 / torchvision 0.21.0
- timm: not used (`convnext_tiny` is `torchvision.models.convnext_tiny`)
- device: `mps`
- matmul precision: in these runs `torch.get_float32_matmul_precision() == "highest"` (default)
  - This can vary by PyTorch version/backend; set `torch.set_float32_matmul_precision("highest")` explicitly if needed.

Reproducibility note on `--limit-images`:
- When `--limit-images` is used, the subset selection is implemented to be stable given the same `--seed` and identical inputs (CSV + preprocessed DB).
- You can audit subset+split identity via `meta.json.split_stats` (it records `limit_images`, `seed`, and `split_stats`).

Preprocessing fit boundary (explicit, to avoid leakage suspicion):
- `--windows` (HU window) and `--stack-slices` are **fixed transforms**; they do not fit statistics from train/val.
- In this README’s settings, `--input-normalize none`, so no dataset mean/std is fitted.
- The preprocessed SQLite stores transformed images and metadata (Study/Series UID etc.); it does not depend on labels or splits.

Data example (used in these runs):
- `rsna_root`: `pipeline/Datasets/rsna_preprocessed_gpt52_img384_w3_f32/rsna_meta`
- `preprocessed_root`: `pipeline/Datasets/rsna_preprocessed_gpt52_img384_w3_f32`

---

### 2.2 Command (fine-tune short / val_frac=0.05)

Entry point:
- `train_rsna_cnn2d_classifier.py` (thin wrapper in this folder)
  - implementation: `src/training/train_rsna_cnn2d_classifier.py` (Typer app)

```bash
cd pipeline

TORCH_DEVICE=mps python train_rsna_cnn2d_classifier.py train \
  --rsna-root "Datasets/rsna_preprocessed_gpt52_img384_w3_f32/rsna_meta" \
  --preprocessed-root "Datasets/rsna_preprocessed_gpt52_img384_w3_f32" \
  --out-dir "results/<YOUR_RUN_DIR>" \
  --limit-images 8000 \
  --val-frac 0.05 \
  --split-by study \
  --seed 0 \
  --epochs 1 \
  --lr 5e-06 \
  --weight-decay 0.0 \
  --image-size 384 \
  --windows '40,80;80,200;600,2800' \
  --preprocess gpt52 \
  --stack-slices 3 \
  --batch-size 6 \
  --num-workers 0 \
  --arch convnext_tiny \
  --pretrained \
  --first-conv-init mean \
  --input-normalize none \
  --no-aug \
  --no-scheduler \
  --loss-any-weight 1.0 \
  --no-use-pos-weight \
  --no-use-sampler \
  --optimize-plain-loss \
  --init-from "results/rsna_convnext25d_final2_full_mps_nosamp_nopw_20260202_222608/best.pt" \
  --log-every-steps 200
```

Where `init_from` comes from:
- `results/rsna_convnext25d_final2_full_mps_nosamp_nopw_20260202_222608/best.pt`
  - A checkpoint saved as **minimum `val_loss`** on holdout validation (`split_by=study`, `val_frac=0.02`) in another run.
  - That run is independent from this README’s evaluation split (`val_frac=0.05`), and its selection did not use this report’s evaluation results.
  - Used as a stable fine-tuning initialization; we do not swap it based on the CV/holdout results in this README.
  - If you want a checkpoint saved by minimum weighted logloss, use `best_wlogloss.pt` in the same folder.

Artifacts:
- `meta.json`: experiment config (includes `split_stats`)
- `log.jsonl`: per-epoch aggregated metrics (last line contains final val metrics)
- `best.pt` / `best_auc.pt` / `best_wlogloss.pt` / `last.pt` / `last_state.pt`

Audit points:
- `meta.json` (args incl. at least `rsna_root`, `preprocessed_root`, `seed`, plus `split_stats`)
- the last line of `log.jsonl` (final `val_*`)

---

### 2.3 Command (GroupKFold(5) fast CV / epochs=1)

To reproduce fast CV, run each fold via `--cv-folds 5` and `--cv-fold-index <0..4>`.

Notes:
- Treat `python train_rsna_cnn2d_classifier.py train --help` as the source of truth for CLI names.
- When `--cv-folds >= 2`, the split is determined by fold index and `--val-frac` does not decide the split (it must still satisfy `0 <= val_frac < 1`).

```bash
cd pipeline

BASE="results/<YOUR_CV_DIR>"
for FOLD in 0 1 2 3 4; do
  OUT="$BASE/fold${FOLD}"
  mkdir -p "$OUT"
  TORCH_DEVICE=mps python train_rsna_cnn2d_classifier.py train \
    --rsna-root "Datasets/rsna_preprocessed_gpt52_img384_w3_f32/rsna_meta" \
    --preprocessed-root "Datasets/rsna_preprocessed_gpt52_img384_w3_f32" \
    --out-dir "$OUT" \
    --limit-images 8000 \
    --split-by study \
    --cv-folds 5 \
    --cv-fold-index "$FOLD" \
    --seed 0 \
    --epochs 1 \
    --lr 5e-06 \
    --weight-decay 0.0 \
    --image-size 384 \
    --windows '40,80;80,200;600,2800' \
    --preprocess gpt52 \
    --stack-slices 3 \
    --batch-size 6 \
    --num-workers 0 \
    --arch convnext_tiny \
    --pretrained \
    --first-conv-init mean \
    --input-normalize none \
    --no-aug \
    --no-scheduler \
    --loss-any-weight 1.0 \
    --no-use-pos-weight \
    --no-use-sampler \
    --optimize-plain-loss \
    --init-from "results/rsna_convnext25d_final2_full_mps_nosamp_nopw_20260202_222608/best.pt" \
    --log-every-steps 200
done

# Summarize CV (mean/std/min/max)
python tools/summarize_cv.py --cv-root "$BASE" --format md
```

---

## 3. Evidence of Reproducibility (run1 vs run2)

### 3.1 What is being compared

Here, `auc_max_abs_diff=0.0` means:
- the maximum absolute difference across per-class validation AUC values between run1 and run2 is 0.

### 3.2 Comparison command (copy/paste)

```bash
cd pipeline
python -c 'import json; from pathlib import Path
r1=Path("results/rsna_convnext25d_ft_repro_val05_short_20260210_101836_run1"); r2=Path("results/rsna_convnext25d_ft_repro_val05_short_20260210_111323_run2")
m1=json.loads(r1.joinpath("meta.json").read_text()); m2=json.loads(r2.joinpath("meta.json").read_text())

def last(p):
    lines=p.joinpath("log.jsonl").read_text().strip().splitlines();
    return json.loads(lines[-1])

l1=last(r1); l2=last(r2)
print("split_stats_match", m1.get("split_stats")==m2.get("split_stats"))
for k in ["val_loss","val_loss_plain","val_auc_mean","val_logloss_weighted"]:
    a=float(l1.get(k)); b=float(l2.get(k));
    print(k, "abs_diff", abs(a-b))
auc1=l1.get("val_auc",{}); auc2=l2.get("val_auc",{})
maxd=0.0
for c in sorted(set(auc1)|set(auc2)):
    if c in auc1 and c in auc2:
        maxd=max(maxd, abs(float(auc1[c])-float(auc2[c])))
print("auc_max_abs_diff", maxd)
'
```

Expected output:
- `split_stats_match True`
- `abs_diff 0.0` for each `val_*`
- `auc_max_abs_diff 0.0`

---

## 4. Why the split design matters (leakage prevention is the “main point”)

RSNA ICH contains many slices per patient/study/series.
If you random-split at the slice level, the same Study can leak into both train and val, inflating validation metrics.

Split options in this pipeline:
- `split_by=study` (recommended): group split by `StudyInstanceUID`
- `split_by=series`: group split by `SeriesInstanceUID`
- `split_by=slice`: speed-first approximation (not suitable for claims; leakage can occur)

### 4.1 Leakage audit (evidence code)

Result (study split): train/val group intersection is exactly 0 (`n_group_intersection=0`).

Pass criteria: n_group_intersection == 0 and n_imageid_intersection == 0.

Audit script to verify “group intersection is 0”:

```bash
cd pipeline
python tools/audit_rsna_split.py \
  --rsna-root "Datasets/rsna_preprocessed_gpt52_img384_w3_f32/rsna_meta" \
  --preprocessed-root "Datasets/rsna_preprocessed_gpt52_img384_w3_f32" \
  --limit-images 8000 \
  --val-frac 0.05 \
  --split-by study \
  --seed 0 \
  --out-json /tmp/audit_study_report.json
```

Observed highlights under the above settings:
- `n_group_intersection=0`
- `n_imageid_intersection=0`
- `n_train_groups=5596`, `n_val_groups=294`

For reference, `split_by=slice` can have no image-id overlap but still leak studies/series across train/val.

Leakage indicator (slice split): n_study_intersection > 0 (non-zero means study leakage into val).

```bash
cd pipeline
python tools/audit_rsna_slice_leakage.py \
  --rsna-root "Datasets/rsna_preprocessed_gpt52_img384_w3_f32/rsna_meta" \
  --preprocessed-root "Datasets/rsna_preprocessed_gpt52_img384_w3_f32" \
  --limit-images 8000 \
  --val-frac 0.05 \
  --seed 0 \
  --out-json /tmp/audit_slice_leakage.json
```

Observed highlights:
- `n_imageid_intersection=0` (expected for slice split)
- `n_study_intersection=89` (≈26% of val-side studies also appear in train)
- `frac_val_studies_in_train=0.261765`

Definitions:
- `n_study_intersection = |{StudyUID in val} ∩ {StudyUID in train}|`
- `frac_val_studies_in_train = n_study_intersection / n_val_studies`

### 4.2 Additional audit (exact-duplicate prevention)

`split_by=study` prevents same-study leakage, but exact content duplicates can still exist under different IDs in preprocessed data.
To address this, this pipeline applies **tensor-hash dedup before split (default ON)** in preprocessed mode (`--dedup-before-split`).

Verification (10 seeds):
- `tools/audit_rsna_dedup_effect.py` reports zero train/val hash overlap after dedup for all seeds (`all_zero_after_dedup=true`).
- `tools/hypothesis_audit_rsna_leakage.py` also reconfirms no Study/Series/Image intersection.

Conclusion (within this README’s audit scope):
- It is reasonable to treat this setup as **free of known evaluation-distorting leakage** (Study/Series crossing and exact-duplicate crossing removed).
- Near-duplicates that are visually similar but not byte-identical require separate similarity audits (e.g., pHash) if needed.

---

## 5. Metric definition and what “good” means

### 5.1 Primary metric (aligned with Kaggle evaluation)

- `val_logloss_weighted`: RSNA ICH weighted multi-label logloss ("any" is heavier)
  - Minimizing this is closest to the Kaggle goal.

Source (reference, Kaggle official definition):
- Kaggle competition “RSNA Intracranial Hemorrhage Detection” Evaluation: https://www.kaggle.com/c/rsna-intracranial-hemorrhage-detection/overview/evaluation
  - 6-class multi-label logloss, weighted average with `any=2` and others `=1`.

Implementation notes (this repository):
- We compute `val_logloss_weighted` using Kaggle weights (`any=2`, others=1).
- The implementation uses the mapping between `RSNA_CLASSES` and `RSNA_LOGLOSS_CLASS_WEIGHTS`, and computes the weighted average in `_weighted_multilabel_logloss()`.
- Class order in `RSNA_CLASSES` is `classes = [epidural, intraparenchymal, intraventricular, subarachnoid, subdural, any]`.
  - Class names follow Kaggle’s `stage_2_train.csv` columns (`ID_<class>`).
  - Kaggle interprets submitted CSV columns as `<image_id>_<class>`.
  - This repo’s submission generation also maps by `<class>` name.
- Note: class order is an implementation detail as long as labels/predictions are aligned consistently; the key is the per-class logloss + Kaggle weights (`any=2`, others=1).
- `--loss-any-weight` affects **training loss only** (in this README’s commands it is `1.0`, i.e., not upweighting `any` during training loss).
- Probabilities are clipped for numerical stability with $\varepsilon$ (default $\varepsilon=10^{-7}$): `p = clip(sigmoid(logit), eps, 1-eps)`.
- Kaggle’s internal evaluation may differ in numeric details (e.g., clipping/aggregation), but all results here use the same implementation, so comparisons across runs/ablations are internally consistent.

$$
\ell_{i,c} = -\bigl(y_{i,c}\log p_{i,c} + (1-y_{i,c})\log(1-p_{i,c})\bigr),\quad
\mathrm{ll}_c = \frac{1}{N}\sum_{i=1}^{N}\ell_{i,c},\quad
\mathrm{wlogloss} = \frac{\sum_c w_c\,\mathrm{ll}_c}{\sum_c w_c}
$$

### 5.2 Auxiliary metric (sanity check)

- `val_auc_mean`: mean of per-class AUC (ranking/separation; does not directly reflect calibration)

### 5.3 Why `val_loss_plain` differs from `val_logloss_weighted`

- When `plain < weighted`, miscalibration/errors in heavy-weight classes (especially `any`) may remain.
- A natural next step is probability calibration (e.g., temperature scaling) to improve logloss.

---

## 6. Comparison axes (so a reviewer can judge)

At minimum, present these three together:

1) **Reproducibility**
- Same seed/args reproduce metrics (demonstrated in this README)

2) **Leakage-safe evaluation**
- Rationale for `split_by=study` plus audits showing group intersection is 0

3) **Fair comparisons**
- Compare under matched conditions (split/val_frac/seed/limit_images), using `val_logloss_weighted` as primary.

Recommended:
- Seed sweep (0/1/2) and report mean±std (done in this README)

---

## 7. Ablation (minimal but decisive)

| ID | split_by | val_frac | preprocess | stack | aug | optimize_plain_loss | init_from | val_logloss_weighted | val_auc_mean | Notes |
|---:|:--|--:|:--|--:|:--|:--|:--|--:|--:|:--|
| A | slice | 0.05 | gpt52 | 3 | off | on | best.pt | 0.05170 | 0.99479 | slice split (warning example: study leakage can inflate metrics) |
| B | study | 0.02 | gpt52 | 3 | off | on | best.pt | TBD | TBD | small val can be noisy (TBD intentionally left blank; fill after running small-val variance checks to avoid cherry-picking) |
| C | study | 0.05 | gpt52 | 3 | off | on | best.pt | 0.05389 ± 0.00709 | 0.98895 ± 0.00712 | mean±std over seeds 0/1/2 |

Important:
- If A (slice) looks better than C (study), that is typically **leakage**, not a real improvement (see §4.1).

---

## 8. Existing scripts (reference)

- `scripts/run_rsna_target058_shortest.zsh`: shortest recipe to run a strong 2D baseline
- `scripts/make_rsna_submission_from_best.zsh`: generate a submission from the best checkpoint

Note (uncertainty / calibration output):
- `predict_rsna_ich_submission.py` also supports MC-Dropout uncertainty export.
  - Example: `--mc-dropout-stage-p 0.2 --mc-dropout-head-p 0.2 --mc-samples 30 --out-uncertainty-csv submission_uncertainty.csv`
  - `submission_uncertainty.csv` writes `ProbStd` (probability standard deviation) per `ID` (separate from submission.csv).

Calibration (temperature scaling) + ECE + coverage–risk curve (one PNG) + coverage=80% improvement (README-friendly):
- `tools/eval_rsna_uncertainty.py` evaluates the following on a holdout split:
  - **Evaluation boundary**: standard `--fit-temperature` usage fits temperature on the same holdout val split used for ECE/Brier/AURC reporting (calibration metrics may look optimistic).
  - **Strict protocol recommendation**: split val into calib/eval, fit temperature on calib, and report final metrics on eval (or use a separate holdout for eval).
  - temperature scaling temperature $T$ (estimated with `--fit-temperature`)
  - `ECE(any)` (binary, `any` class only, 15 bins)
  - `Brier(any)` and weighted Brier (overall probabilistic quality; less bin-dependent than ECE)
  - uncertainty error-detection `AUROC` (label=1 if `any` is misclassified at threshold 0.5; predictor=uncertainty `ProbStd`)
  - `NLL` (here equivalent to Bernoulli weighted logloss)
  - a coverage–risk curve PNG (risk = weighted logloss)
  - a reliability diagram PNG (any)
  - improvement in `accuracy(any)` at coverage=0.8 (percentage points)

```bash
cd pipeline

TORCH_DEVICE=mps python tools/eval_rsna_uncertainty.py \
  --rsna-root "Datasets/rsna_preprocessed_gpt52_img384_w3_f32/rsna_meta" \
  --preprocessed-root "Datasets/rsna_preprocessed_gpt52_img384_w3_f32" \
  --ckpt "results/rsna_convnext25d_final2_full_mps_nosamp_nopw_20260202_222608/best.pt" \
  --arch convnext_tiny \
  --image-size 384 \
  --windows '40,80;80,200;600,2800' \
  --preprocess gpt52 \
  --stack-slices 3 \
  --limit-images 8000 \
  --val-frac 0.05 \
  --split-by study \
  --seed 0 \
  --mc-samples 30 \
  --dropout-stage-p 0.2 \
  --dropout-head-p 0.2 \
  --fit-temperature \
  --out-reliability-png results/uncertainty/reliability_any.png \
  --out-curve-png results/uncertainty/coverage_risk.png
```

Key JSON fields:
- `ece_any`: ECE for `any`
- `brier_any`: Brier score for `any`
- `auroc_uncertainty_detect_error_any`: uncertainty AUROC for detecting errors (`any`)
- `nll_weighted_logloss`: NLL (weighted logloss)
- `accuracy_any_improve_pp`: improvement at coverage=0.8 (pp)
- `curve_png`: saved coverage–risk plot path
- `reliability_png`: saved reliability diagram path

---

## 9. Limitations and next steps (honest = stronger)

Limitations:
- Single holdout validation can be lucky/unlucky (especially with small `val_frac`).
- Validation does not necessarily match Kaggle Private LB.

Next steps (rough priority for a portfolio):
1. Seed sweep mean±std (done)
2. GroupKFold(5) mean±std (done)
3. Calibration (e.g., temperature scaling) to improve logloss
4. Case-level error analysis (typical false negatives)
5. (Implemented) Record a deterministic subset fingerprint (sorted image_ids sha256) in meta.json to allow third-party subset identity checks.

This bundle prioritizes auditable reproducibility and leakage-safe evaluation; reported numbers are meaningful for comparison under identical protocols (not as a direct proxy for Kaggle leaderboard scores).
