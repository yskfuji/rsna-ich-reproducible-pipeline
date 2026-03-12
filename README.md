# rsna-ich-reproducible-pipeline

**Language:** English | [Japanese](README_ja.md)

Reproducible intracranial hemorrhage classification pipeline for the RSNA ICH challenge, with audit-ready documentation, calibration analysis, and leakage-safe group-split evaluation.

**Quick links**
- English entry: [rsna_ich/README_en.md](rsna_ich/README_en.md)
- Japanese entry: [rsna_ich/README.md](rsna_ich/README.md)
- Detailed documentation: [core/pipeline/README_en.md](core/pipeline/README_en.md)
- API demo entry: [core/pipeline/serve_rsna_ich_api.py](core/pipeline/serve_rsna_ich_api.py)
- Reproducibility checklist: [docs/reproducibility_checklist.md](docs/reproducibility_checklist.md)
- GitHub About settings: [EN](docs/github_about.md) | [JA](docs/github_about_ja.md)
- Citation: [CITATION.cff](CITATION.cff)
- Release note source: [EN](docs/releases/v0.6.0-rsna.md) | [JA](docs/releases/v0.6.0-rsna_ja.md)
- Roadmap: [ROADMAP.md](ROADMAP.md)

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

| Metric | Value | Why it matters |
|---|---:|---|
| Weighted multi-label logloss | 0.05346 ± 0.00624 | Primary Kaggle-style optimization target |
| Mean AUC | 0.98815 ± 0.00311 | Ranking quality across classes |
| Error-detection AUROC (`any`) | 0.9424 ± 0.0190 | Uncertainty usefulness for error triage |
| ECE (`any`) | 0.0231 ± 0.0032 | Probability calibration quality |

> Notes: values are from the bundled reproducibility reports. Protected medical data is intentionally not included.

## Quickstart

### 1. Verify the repository without medical data

```bash
python scripts/smoke_test.py --use_dummy_data
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
- `Datasets/`
- `runs/`
- `results/`
- `logs/`

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
