# RSNA Intracranial Hemorrhage (Kaggle 2019)

**Language:** English | [Japanese](README.md)

This folder is the **public entry point** for the RSNA ICH classification pipeline. Pictures come before prose.

## Look first

![Pipeline](../docs/assets/architecture.svg)

![Six labels](../docs/assets/labels_taxonomy.svg)

![Reader paths](../docs/assets/reader_path.svg)

![Metrics](../docs/assets/results_snapshot.svg)

![Constraints](../docs/assets/eval_protocol.svg)

Picture-first root README: [../README.md](../README.md)

## Representative metrics

Mean ± sample standard deviation (ddof=1). The underlying values are in [`../core/pipeline/reports/rsna_cv_summary.json`](../core/pipeline/reports/rsna_cv_summary.json).

- weighted multi-label logloss: **0.05346 ± 0.00624** (5 folds)
- mean AUC: **0.98815 ± 0.00311** (5 folds)
- error-detection AUROC (`any`): **0.9424 ± 0.0200** (10 seeds)
- ECE (`any`): **0.0231 ± 0.0034** (10 seeds)

Evaluation conditions:
- Each fold or seed fine-tunes for 1 epoch on a subsample drawn with `limit_images=8000` (6,852 images remained in the CV run after dropping images missing from the preprocessed store).
- The initial checkpoint was trained on 98% of the studies in the same preprocessed training pool (632,808 of 645,793 slices). Read the values as fine-tuning stability, not as performance on unseen studies.
- Temperature scaling is fitted on the same holdout used to report ECE, so calibration metrics are optimistic.
- This is a research and portfolio implementation, not a medical device.

See the [evaluation notes](../docs/evaluation_notes.md) for details.

## Why this repository is useful

- study-level `split_by=study` splits and audit scripts
- pre-split tensor-hash de-duplication to reduce leakage risk
- reproducibility artifacts such as `meta.json`, `log.jsonl`, and `split_stats`
- subset identity checks via `subset_fingerprint_sha256`

![Study-level split](../docs/assets/split_study.svg)

## Quick links

| Need | Open |
|---|---|
| Fastest first run | `python ../scripts/smoke_test.py --use_dummy_data` |
| Detailed JA docs | [../core/pipeline/README.md](../core/pipeline/README.md) |
| Detailed EN docs | [../core/pipeline/README_en.md](../core/pipeline/README_en.md) |
| Audit quick guide | [./AUDIT_GUIDE.md](./AUDIT_GUIDE.md) |
| API demo | [../docs/api_demo.md](../docs/api_demo.md) |
| Citation | [../CITATION.cff](../CITATION.cff) |
| Release notes | [../docs/releases/v0.6.0-rsna.md](../docs/releases/v0.6.0-rsna.md) |
| Roadmap | [../ROADMAP.md](../ROADMAP.md) |

![Included vs excluded](../docs/assets/included_excluded.svg)

## MLflow tracking schema

When `--mlflow` is enabled, this public repository uses a shared tracking schema across the three portfolio repositories.

- Shared run tags: `repo_name`, `task_type`, `model_family`, `tracking_schema=public_portfolio_v1`
- Shared artifact groups:
  - `run_metadata/`: `meta.json` plus optional config snapshots and task-specific JSON
  - `training_trace/`: `log.jsonl`
  - `checkpoints/`: `last.pt`, `best.pt`, and task-specific best checkpoints
- Goal: make segmentation and classification runs browsable the same way, without claiming a production MLOps platform

## Current portfolio snapshot

The current portfolio snapshot corresponds to:

✅ `v0.6.0-rsna`

Active development continues. This public folder does not ship `Datasets/`, `runs/`, or `results/`. Use the detailed README for commands that need your own data.
