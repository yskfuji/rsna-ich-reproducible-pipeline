# 図だけ見る索引

**言語:** 日本語 | [英語版](visual_index.md)

このページは文章をほぼ書きません。上から順に見てください。数値の意味は [evaluation_notes_ja.md](evaluation_notes_ja.md)、用語は [glossary_ja.md](glossary_ja.md) です。

## 全体の流れ

![architecture](assets/architecture.svg)

## 予測ラベル

![labels](assets/labels_taxonomy.svg)

## 分割

![split](assets/split_study.svg)

## 指標

![results](assets/results_snapshot.svg)

![cv folds](assets/cv_folds.svg)

![uncertainty](assets/uncertainty_seeds.svg)

## 数値の読み方

![protocol](assets/eval_protocol.svg)

![evidence](assets/evidence_chain.svg)

## リポジトリ

![repo map](assets/repo_map.svg)

![reader path](assets/reader_path.svg)

![included](assets/included_excluded.svg)

## 図の再生成

```bash
python docs/assets/generate_figures.py
```

指標カードは `core/pipeline/reports/rsna_cv_summary.json` から埋めます。患者画像は使いません。
