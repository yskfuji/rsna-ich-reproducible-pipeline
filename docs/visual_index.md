# Picture-only index

**Language:** English | [Japanese](visual_index_ja.md)

Almost no prose. Scroll the figures. For what the numbers mean, open [evaluation_notes.md](evaluation_notes.md).

## Pipeline

<p align="center"><img src="assets/architecture.png" alt="architecture" width="100%" /></p>

## Labels

<p align="center"><img src="assets/labels_taxonomy.png" alt="labels" width="100%" /></p>

## Split

<p align="center"><img src="assets/split_study.png" alt="split" width="100%" /></p>

## Metrics

<p align="center"><img src="assets/results_snapshot.png" alt="results" width="100%" /></p>

<p align="center"><img src="assets/cv_folds.png" alt="cv folds" width="100%" /></p>

<p align="center"><img src="assets/uncertainty_seeds.png" alt="uncertainty" width="100%" /></p>

## How to read the numbers

<p align="center"><img src="assets/eval_protocol.png" alt="protocol" width="100%" /></p>

<p align="center"><img src="assets/evidence_chain.png" alt="evidence" width="100%" /></p>

## Repository

<p align="center"><img src="assets/repo_map.png" alt="repo map" width="100%" /></p>

<p align="center"><img src="assets/reader_path.png" alt="reader path" width="100%" /></p>

<p align="center"><img src="assets/included_excluded.png" alt="included" width="100%" /></p>

## Regenerate

```bash
python docs/assets/generate_figures.py
```

Metric cards are filled from `core/pipeline/reports/rsna_cv_summary.json`. No patient pixels.
