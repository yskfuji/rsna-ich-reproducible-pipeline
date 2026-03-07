# rsna-ich-reproducible-pipeline

**Language:** 日本語 | [English](README.md)

RSNA ICH challenge 向けの、**再現可能な頭蓋内出血分類パイプライン**です。監査しやすいドキュメント、calibration 解析、リークを避ける group-split 評価を含みます。

**クイックリンク**
- 英語入口: [rsna_ich/README_en.md](rsna_ich/README_en.md)
- 日本語入口: [rsna_ich/README.md](rsna_ich/README.md)
- 実験詳細: [core/pipeline/README.md](core/pipeline/README.md)
- Citation: [CITATION.cff](CITATION.cff)
- リリースノート原稿: [docs/releases/v1.0-interview.md](docs/releases/v1.0-interview.md)
- ロードマップ: [ROADMAP.md](ROADMAP.md)

## このリポジトリで分かること

- RSNA 頭蓋内出血分類の学習 / 評価ワークフロー
- `split_by=study` による group split と leakage audit
- `any` ラベルに対する calibration / uncertainty 評価
- 外部レビュー向けに整理したポートフォリオ導線
- 実データ不要で配線確認できる no-data smoke test

## 想定読者

- 医療AI実装を確認したい採用担当
- 監査しやすい医用画像ベースラインを見たい ML エンジニア
- 再現性重視の CT 分類プロジェクト構成を探している研究者

## 3分で分かる概要

![RSNA architecture](docs/assets/architecture.svg)

![RSNA repository map](docs/assets/repo_map.svg)

![RSNA metrics snapshot](docs/assets/results_snapshot.svg)

### 代表指標

| 指標 | 値 | 意味 |
|---|---:|---|
| Weighted multi-label logloss | 0.05346 ± 0.00624 | Kaggle 互換の主指標 |
| Mean AUC | 0.98815 ± 0.00311 | クラス横断の順位付け性能 |
| Error-detection AUROC (`any`) | 0.9424 ± 0.0190 | uncertainty による誤り検出性能 |
| ECE (`any`) | 0.0231 ± 0.0032 | 確率の校正品質 |

> 数値は同梱レポート由来です。医療データ本体は公開物に含めていません。

## Quickstart

### 1. 実データなしで配線確認

```bash
python scripts/smoke_test.py --use_dummy_data
```

### 2. 配布物マニフェストを生成

```bash
cd core/pipeline
python tools/make_manifest.py
```

### 3. 実データで学習 / 評価

- 日本語詳細: [core/pipeline/README.md](core/pipeline/README.md)
- English full guide: [core/pipeline/README_en.md](core/pipeline/README_en.md)

## 含まれるもの / 含まれないもの

含まれるもの:
- ソースコード
- 設定ファイル
- 監査 / 評価ドキュメント
- 静的図表と release note 原稿

含まれないもの:
- `Datasets/`
- `runs/`
- `results/`
- `logs/`

## Stable portfolio version

開発は継続中ですが、ポートフォリオ / 面接レビュー用の固定版は次のタグです。

✅ `rsna-ich-v1.0-interview`

## How to cite

[CITATION.cff](CITATION.cff) を参照してください。

## Commit message convention

今後の変更は Conventional Commits（`type: summary`）で揃えます。

- `fix: leakage check in group split`
- `feat: add calibration evaluation`
- `refactor: manifest validation logic`
- `docs: evaluation protocol clarification`
