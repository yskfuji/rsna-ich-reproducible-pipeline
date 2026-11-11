# Picture-only index

**Language:** English | [Japanese](visual_index_ja.md)

Almost no prose. Scroll the figures. For what the numbers mean, open [evaluation_notes.md](evaluation_notes.md).

## Pipeline

![architecture](assets/architecture.svg)

## Labels

![labels](assets/labels_taxonomy.svg)

## Split

![split](assets/split_study.svg)

## Metrics

![results](assets/results_snapshot.svg)

![cv folds](assets/cv_folds.svg)

![uncertainty](assets/uncertainty_seeds.svg)

## How to read the numbers

![protocol](assets/eval_protocol.svg)

![evidence](assets/evidence_chain.svg)

## Repository

![repo map](assets/repo_map.svg)

![reader path](assets/reader_path.svg)

![included](assets/included_excluded.svg)

## Regenerate

```bash
python docs/assets/generate_figures.py
```

Metric cards are filled from `core/pipeline/reports/rsna_cv_summary.json`. No patient pixels.
