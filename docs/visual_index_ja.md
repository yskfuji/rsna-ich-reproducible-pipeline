# 図索引

**言語:** 日本語 | [英語版](visual_index.md)

本稿は文章を最小限にとどめ、図の閲覧順を示す。数値の解釈は [evaluation_notes_ja.md](evaluation_notes_ja.md)、用語は [glossary_ja.md](glossary_ja.md) を参照する。

## 全体の流れ

<p align="center"><img src="assets/architecture.png" alt="architecture" width="100%" /></p>

## 予測ラベル

<p align="center"><img src="assets/labels_taxonomy.png" alt="labels" width="100%" /></p>

## 分割

<p align="center"><img src="assets/split_study.png" alt="split" width="100%" /></p>

## 指標

<p align="center"><img src="assets/results_snapshot.png" alt="results" width="100%" /></p>

<p align="center"><img src="assets/cv_folds.png" alt="cv folds" width="100%" /></p>

<p align="center"><img src="assets/uncertainty_seeds.png" alt="uncertainty" width="100%" /></p>

## 数値の解釈

<p align="center"><img src="assets/eval_protocol.png" alt="protocol" width="100%" /></p>

<p align="center"><img src="assets/evidence_chain.png" alt="evidence" width="100%" /></p>

## リポジトリ

<p align="center"><img src="assets/repo_map.png" alt="repo map" width="100%" /></p>

<p align="center"><img src="assets/reader_path.png" alt="reader path" width="100%" /></p>

<p align="center"><img src="assets/included_excluded.png" alt="included" width="100%" /></p>

## 図の再生成

```bash
python docs/assets/generate_figures.py
```

指標カードは `core/pipeline/reports/rsna_cv_summary.json` から埋め込む。患者画像は用いない。
