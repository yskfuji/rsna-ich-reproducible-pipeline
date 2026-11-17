# 図索引

**言語:** 日本語 | [英語版](visual_index.md)

本稿は文章を最小限にとどめ、図の閲覧順を示す。数値の解釈は [evaluation_notes_ja.md](evaluation_notes_ja.md)、用語は [glossary_ja.md](glossary_ja.md) を参照する。

## 処理構成

<p align="center"><img src="assets/architecture_ja.png" alt="処理構成" width="100%" /></p>

## 動作確認の流れ

<p align="center"><img src="assets/smoke_flow_ja.png" alt="動作確認の流れ" width="100%" /></p>

## 予測ラベル

<p align="center"><img src="assets/labels_taxonomy_ja.png" alt="予測ラベル" width="100%" /></p>

## 分割

<p align="center"><img src="assets/split_study_ja.png" alt="分割" width="100%" /></p>

## 指標

<p align="center"><img src="assets/results_snapshot_ja.png" alt="代表指標" width="100%" /></p>

<p align="center"><img src="assets/cv_folds_ja.png" alt="fold別指標" width="100%" /></p>

<p align="center"><img src="assets/uncertainty_seeds_ja.png" alt="不確実性" width="100%" /></p>

## 数値の解釈

<p align="center"><img src="assets/eval_constraints_ja.png" alt="評価上の制約" width="100%" /></p>

<p align="center"><img src="assets/evidence_chain_ja.png" alt="根拠連鎖" width="100%" /></p>

## リポジトリ

<p align="center"><img src="assets/repo_map_ja.png" alt="リポジトリ案内" width="100%" /></p>

<p align="center"><img src="assets/reader_path_ja.png" alt="閲覧経路" width="100%" /></p>

<p align="center"><img src="assets/included_excluded_ja.png" alt="公開範囲" width="100%" /></p>

## 図の再生成

```bash
python docs/assets/generate_figures.py
```

指標カードは `core/pipeline/reports/rsna_cv_summary.json` から埋め込む。患者画像は用いない。日本語文書では `assets/*_ja.png` を用いる。
