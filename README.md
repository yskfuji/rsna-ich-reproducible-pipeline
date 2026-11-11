# rsna-ich-reproducible-pipeline

**Language:** English | [Japanese](README_ja.md)

[![ci](https://github.com/yskfuji/rsna-ich-reproducible-pipeline/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/yskfuji/rsna-ich-reproducible-pipeline/actions/workflows/ci.yml)

Reproducible intracranial hemorrhage classification pipeline for the RSNA ICH challenge, with audit-ready documentation, calibration analysis, and leakage-safe group-split evaluation.

## Review in 5 minutes

| What to check | Where |
|---|---|
| DICOM loading, HU windowing, de-duplication | [core/pipeline/src/datasets/rsna_ich_dataset.py](core/pipeline/src/datasets/rsna_ich_dataset.py) |
| Training loop, weighted loss, study-level GroupKFold | [core/pipeline/src/training/train_rsna_cnn2d_classifier.py](core/pipeline/src/training/train_rsna_cnn2d_classifier.py) |
| 3D U-Net encoder classifier | [core/pipeline/src/models/unet3d_encoder_classifier.py](core/pipeline/src/models/unet3d_encoder_classifier.py) |
| Submission inference | [core/pipeline/src/inference/predict_rsna_ich_submission.py](core/pipeline/src/inference/predict_rsna_ich_submission.py) |
| Train/validation leakage audit | [core/pipeline/tools/audit_rsna_split.py](core/pipeline/tools/audit_rsna_split.py) |
| MC Dropout, ECE, temperature scaling, AURC | [core/pipeline/tools/eval_rsna_uncertainty.py](core/pipeline/tools/eval_rsna_uncertainty.py) |
| Per-fold and per-seed numbers behind the table below | [core/pipeline/reports/rsna_cv_summary.json](core/pipeline/reports/rsna_cv_summary.json) |
| Evaluation conditions, limitations and references | [docs/evaluation_notes.md](docs/evaluation_notes.md) |

**Quick links**
- English entry: [rsna_ich/README_en.md](rsna_ich/README_en.md)
- Japanese entry: [rsna_ich/README.md](rsna_ich/README.md)
- Detailed documentation: [core/pipeline/README_en.md](core/pipeline/README_en.md)
- API demo notes: [docs/api_demo.md](docs/api_demo.md)
- Reproducibility checklist: [docs/reproducibility_checklist.md](docs/reproducibility_checklist.md)
- Suggested review order: [EN snapshot guide](docs/releases/v0.6.0-rsna.md) | [JA snapshot guide](docs/releases/v0.6.0-rsna_ja.md)
- GitHub About settings: [EN](docs/github_about.md) | [JA](docs/github_about_ja.md)
- Citation: [CITATION.cff](CITATION.cff)
- Release note source: [EN](docs/releases/v0.6.0-rsna.md) | [JA](docs/releases/v0.6.0-rsna_ja.md)
- Roadmap: [ROADMAP.md](ROADMAP.md)
- Repository integrity audit (2026-11): [docs/audit/2026-11-repo-integrity.md](docs/audit/2026-11-repo-integrity.md)

## What this repository provides

- A reproducible training and evaluation workflow for RSNA intracranial hemorrhage classification
- Study-level group splits with leakage-audit hooks
- Calibration and uncertainty analysis for the `any` label
- Portfolio-ready documentation for external review
- A no-data smoke test that checks the public bundle in under a minute

## Who this is for

- Hiring managers reviewing medical AI implementation quality
- ML engineers looking for an auditable medical-imaging baseline
- Researchers who need a reproducible CT classification project structure

## 3-minute overview

![RSNA architecture](docs/assets/architecture.svg)

![RSNA repository map](docs/assets/repo_map.svg)

![RSNA metrics snapshot](docs/assets/results_snapshot.svg)

### Representative results

Mean ± sample standard deviation (ddof=1). The per-fold and per-seed values are in [core/pipeline/reports/rsna_cv_summary.json](core/pipeline/reports/rsna_cv_summary.json), and CI recomputes every number in this table from that file.

| Metric | Value | n | Why it matters |
|---|---:|---:|---|
| Weighted multi-label logloss ↓ | 0.05346 ± 0.00624 | 5 folds | Kaggle competition metric |
| Mean AUC ↑ | 0.98815 ± 0.00311 | 5 folds | Ranking quality across the 6 labels |
| Error-detection AUROC (`any`) ↑ | 0.9424 ± 0.0200 | 10 seeds | How well uncertainty flags wrong predictions |
| ECE (`any`) ↓ | 0.0231 ± 0.0034 | 10 seeds | Probability calibration |

> **Read these numbers together with their evaluation conditions.**
> - Each fold or seed fine-tunes for **1 epoch** on a subsample drawn with `limit_images=8000` (6,852 images remained in the CV run after dropping images missing from the preprocessed store).
> - Every run starts from the same **initial checkpoint**, which was trained on 98% of the studies in the same preprocessed training pool (632,808 of 645,793 slices). Most validation images were therefore probably seen before fine-tuning. Read the values as fine-tuning stability, not as performance on unseen studies.
> - Temperature scaling is fitted on the **same holdout** that is used to report ECE, so calibration metrics are optimistic.
> - Splits group by study, not by patient.
> - This is a research and portfolio implementation, **not a medical device**.
>
> Details and references: [docs/evaluation_notes.md](docs/evaluation_notes.md). Medical images are not included.

## Quickstart

### 1. Verify the repository without medical data

```bash
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.6.0 torchvision==0.21.0
pip install -r core/pipeline/requirements.txt
python scripts/smoke_test.py --use_dummy_data
python scripts/check_imports.py
python scripts/check_metric_claims.py
```

### 2. Generate a bundle manifest

```bash
cd core/pipeline
python tools/make_manifest.py
```

### 3. Run full training / evaluation with your own data

- Full guide in English: [core/pipeline/README_en.md](core/pipeline/README_en.md)
- Full guide in Japanese: [core/pipeline/README.md](core/pipeline/README.md)

### 4. Inspect the API demo surface

```bash
cd core/pipeline
python serve_rsna_ich_api.py
```

If port `8000` is already in use, run `API_PORT=8011 python serve_rsna_ich_api.py` instead.

Then open `/docs` locally to inspect the FastAPI schema.

## What is included vs excluded

Included:
- source code
- configs
- audit and evaluation documentation
- static summary figures and release-note sources

Not included:
- RSNA images and labels (download them from Kaggle under the competition's non-commercial terms)
- `Datasets/`
- `runs/`
- `results/`
- `logs/`
- trained weights (`*.pt`)

## Current portfolio snapshot

Active development continues in this repository. The current portfolio snapshot is:

✅ `v0.6.0-rsna`

## How to cite

See [CITATION.cff](CITATION.cff).

## Commit message convention

To keep ongoing changes reviewable, future commits follow Conventional Commits (`type: summary`):

- `fix: leakage check in group split`
- `feat: add calibration evaluation`
- `refactor: manifest validation logic`
- `docs: evaluation protocol clarification`
