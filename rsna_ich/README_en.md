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

- weighted multi-label logloss: **0.05346 ± 0.00624**
- mean AUC: **0.98815 ± 0.00311**
- error-detection AUROC (`any`): **0.9424 ± 0.0190**
- ECE (`any`): **0.0231 ± 0.0032**

## Why this repository is useful

- study-level `split_by=study` splits and audit scripts
- pre-split tensor-hash de-duplication to reduce leakage risk
- reproducibility artifacts such as `meta.json`, `log.jsonl`, and `split_stats`
- subset identity checks via `subset_fingerprint_sha256`

## Quick links

- Audit quick guide: `./AUDIT_GUIDE.md`
- API demo note: `../docs/api_demo.md`
- Citation: `../CITATION.cff`
- Release-note source: `../docs/releases/v0.5.0-rsna.md`
- Roadmap: `../ROADMAP.md`

## Current portfolio snapshot

The current portfolio snapshot corresponds to:

✅ `v0.5.0-rsna`

Active development continues in the repository.

This public package intentionally excludes `Datasets/`, `runs/`, and `results/`.
