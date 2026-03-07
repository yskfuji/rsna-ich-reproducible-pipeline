# rsna-ich-reproducible-pipeline

Reproducible intracranial hemorrhage classification pipeline for the RSNA ICH challenge, with audit-ready documentation, calibration analysis, and leakage-safe group-split evaluation.

**Quick links**
- English entry: [rsna_ich/README_en.md](rsna_ich/README_en.md)
- 日本語入口: [rsna_ich/README.md](rsna_ich/README.md)
- Detailed experiment docs: [core/pipeline/README_en.md](core/pipeline/README_en.md)
- Citation: [CITATION.cff](CITATION.cff)
- Release note source: [docs/releases/v1.0-interview.md](docs/releases/v1.0-interview.md)
- Roadmap: [ROADMAP.md](ROADMAP.md)

## What this repository provides

- Reproducible training / evaluation workflow for RSNA intracranial hemorrhage classification
- Study-level group split and leakage audit hooks
- Calibration and uncertainty evaluation for the `any` label
- Portfolio-friendly documentation for external review
- A no-data smoke test that verifies repository wiring in under a minute

## Who this is for

- Hiring managers reviewing medical AI implementation quality
- ML engineers who want an auditable medical-imaging baseline
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

- English full guide: [core/pipeline/README_en.md](core/pipeline/README_en.md)
- 日本語詳細: [core/pipeline/README.md](core/pipeline/README.md)

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

Active development continues on this repository. The stable review snapshot used for portfolio / interview review is:

✅ `rsna-ich-v1.0-interview`

## Japanese summary

このリポジトリは、**RSNA ICH 分類**を第三者が理解しやすい形で再現できるよう整理した公開版です。

- 何ができるか: 学習 / 評価 / 推論 / リーク監査 / 校正評価
- 強み: `split_by=study`、監査しやすいログ、uncertainty / calibration の整理
- 最短確認: `python scripts/smoke_test.py --use_dummy_data`
- 日本語入口: [rsna_ich/README.md](rsna_ich/README.md)
- 実験詳細: [core/pipeline/README.md](core/pipeline/README.md)

## How to cite

See [CITATION.cff](CITATION.cff).

## Commit message convention

To keep ongoing changes reviewable, future commits follow Conventional Commits (`type: summary`):

- `fix: leakage check in group split`
- `feat: add calibration evaluation`
- `refactor: manifest validation logic`
- `docs: evaluation protocol clarification`
