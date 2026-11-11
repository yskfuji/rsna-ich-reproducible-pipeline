# RSNA Intracranial Hemorrhage (Kaggle 2019)

**Language:** English | [Japanese](README.md)

This folder is the **public entry point** for the RSNA ICH classification pipeline. It is written for hiring review, external technical inspection, and fast project onboarding.

## What a reviewer can verify quickly

- **What it does**: trains, evaluates, and audits RSNA ICH models
- **API demo**: exposes public-facing `health`, `model metadata`, and `ich inference` schemas via FastAPI
- **Who it is for**: hiring managers, ML engineers, and researchers who care about reproducibility
- **Fastest first run**: `python ../scripts/smoke_test.py --use_dummy_data`
- **Detailed docs**:
  - Japanese: `../core/pipeline/README.md`
  - English: `../core/pipeline/README_en.md`

## Representative metrics

Mean ± sample standard deviation (ddof=1). The underlying values are in [`../core/pipeline/reports/rsna_cv_summary.json`](../core/pipeline/reports/rsna_cv_summary.json).

- weighted multi-label logloss: **0.05346 ± 0.00624** (5 folds)
- mean AUC: **0.98815 ± 0.00311** (5 folds)
- error-detection AUROC (`any`): **0.9424 ± 0.0200** (10 seeds)
- ECE (`any`): **0.0231 ± 0.0034** (10 seeds)

Evaluation conditions:
- Each fold or seed fine-tunes for 1 epoch on an 8,000-image subsample.
- The initial checkpoint was trained on 98% of the studies in the same training pool. Read the values as fine-tuning stability, not as performance on unseen studies.
- Temperature scaling is fitted on the same holdout used to report ECE, so calibration metrics are optimistic.
- This is a research and portfolio implementation, not a medical device.

See the [evaluation notes](../docs/evaluation_notes.md) for details.

## Why this repository is useful

- study-level `split_by=study` splits and audit scripts
- pre-split tensor-hash de-duplication to reduce leakage risk
- reproducibility artifacts such as `meta.json`, `log.jsonl`, and `split_stats`
- subset identity checks via `subset_fingerprint_sha256`

## Quick links

- Audit quick guide: `./AUDIT_GUIDE.md`
- API demo note: `../docs/api_demo.md`
- Citation: `../CITATION.cff`
- Release-note source: `../docs/releases/v0.6.0-rsna.md`
- Roadmap: `../ROADMAP.md`

## MLflow tracking schema

When `--mlflow` is enabled, this repository follows the same public tracking schema used across the three portfolio repositories.

- Common run tags: `repo_name`, `task_type`, `model_family`, `tracking_schema=public_portfolio_v1`
- Common artifact groups:
  - `run_metadata/`: `meta.json` and, when available, a config snapshot or task-specific JSON
  - `training_trace/`: `log.jsonl`
  - `checkpoints/`: `last.pt`, `best.pt`, and any task-specific best-checkpoint variants
- Goal: make run review easier across segmentation and classification repos without claiming a production MLOps platform

## Current portfolio snapshot

The current portfolio snapshot corresponds to:

✅ `v0.6.0-rsna`

Active development continues in the repository.

This public package intentionally excludes `Datasets/`, `runs/`, and `results/`.
