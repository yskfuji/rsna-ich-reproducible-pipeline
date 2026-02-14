# RSNA Intracranial Hemorrhage (Kaggle 2019)

This folder is the RSNA public-entry point for auditors and reviewers.

- Main code: `../core/pipeline/`
- Experiment READMEs (JP/EN):
  - `../core/pipeline/README.md`
  - `../core/pipeline/README_en.md`
- Audit quick guide:
  - `./AUDIT_GUIDE.md`

## Key checkpoints

- `split_by=study` group split and audit scripts
- pre-split tensor-hash de-dup (leakage mitigation)
- reproducibility artifacts: `meta.json`, `log.jsonl`, `split_stats`
- subset identity audit: `subset_fingerprint_sha256`

This public package intentionally excludes `Datasets/`, `runs/`, and `results/`.
