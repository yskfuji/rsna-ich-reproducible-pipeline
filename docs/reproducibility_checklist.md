# Reproducibility Checklist

Use this page as a fast external-review checklist for the public RSNA ICH repository.

## 1. Package integrity

- Confirm the stable snapshot tag referenced in the README and release note.
- Confirm the repository excludes protected medical data and private run artifacts.
- Optionally generate a fresh manifest with `python scripts/smoke_test.py --use_dummy_data` or `python tools/make_manifest.py` from `core/pipeline`.

## 2. Documentation consistency

- Read the landing page: `README.md` or `README_ja.md`.
- Read the task-facing guide: `rsna_ich/README_en.md` or `rsna_ich/README.md`.
- Read the release note source under `docs/releases/v0.5.0-rsna.md`.
- Confirm the reported logloss, AUC, uncertainty, and calibration claims are consistent across those files.

## 3. Code-path sanity

- Verify that train, predict, and API-demo entrypoints exist:
  - `core/pipeline/train_rsna_cnn2d_classifier.py`
  - `core/pipeline/predict_rsna_ich_submission.py`
  - `core/pipeline/serve_rsna_ich_api.py`
- Verify that leakage-audit and uncertainty-evaluation scripts are present in the public bundle.

## 4. Smoke-test validation

- Run `python scripts/smoke_test.py --use_dummy_data`.
- Confirm the command completes without requiring medical data.
- Confirm the generated summary points at the expected public files and entrypoints.

## 5. Evaluation-readiness checks

- Confirm the README states that `split_by=study` is used for leakage-safe validation.
- Confirm uncertainty and calibration metrics are surfaced, not just classification scores.
- Confirm the FastAPI demo is documented as a schema-level public demo, not a production deployment claim.

## 6. Reviewer pass criteria

- A reviewer can identify the main metric, leakage controls, and first commands to run in under 3 minutes.
- A reviewer can trace train -> evaluate -> audit -> API demo without needing hidden scripts.
- A reviewer can validate repository wiring without access to protected data.

## 7. Known limits

- This checklist validates public reproducibility scaffolding, not Kaggle leaderboard reproduction.
- Full metric reproduction still requires separately prepared RSNA data and the corresponding training environment.