# rsna-ich-reproducible-pipeline

**Language:** English | [Japanese](README_ja.md)

Reproducible intracranial hemorrhage classification pipeline for the RSNA ICH challenge, with audit-ready documentation, calibration analysis, and leakage-safe group-split evaluation.

**Quick links**
- English entry: [rsna_ich/README_en.md](rsna_ich/README_en.md)
- Japanese entry: [rsna_ich/README.md](rsna_ich/README.md)
- Detailed documentation: [core/pipeline/README_en.md](core/pipeline/README_en.md)
- Citation: [CITATION.cff](CITATION.cff)
- Release note source: [docs/releases/v1.0-interview.md](docs/releases/v1.0-interview.md)
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

## Stable portfolio version

Active development continues in this repository. The stable snapshot used for portfolio and interview review is:

✅ `rsna-ich-v1.0-interview`

## How to cite

See [CITATION.cff](CITATION.cff).

## Commit message convention

To keep ongoing changes reviewable, future commits follow Conventional Commits (`type: summary`):

- `fix: leakage check in group split`
- `feat: add calibration evaluation`
- `refactor: manifest validation logic`
- `docs: evaluation protocol clarification`
