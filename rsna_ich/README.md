# RSNA 頭蓋内出血の分類（Kaggle 2019）

**言語:** 日本語 | [英語版](README_en.md)

このフォルダは、**RSNA ICH 分類パイプライン公開版の概要ページ**です。字面より先に図を見てください。

## まず絵

![流れ](../docs/assets/architecture.svg)

![6ラベル](../docs/assets/labels_taxonomy.svg)

![読み方](../docs/assets/reader_path.svg)

![指標](../docs/assets/results_snapshot.svg)

![制約](../docs/assets/eval_protocol.svg)

ルートの絵中心README: [../README_ja.md](../README_ja.md)

## 成果の要点

表記は「平均 ± 標本標準偏差（ddof=1）」です。元の値は [`../core/pipeline/reports/rsna_cv_summary.json`](../core/pipeline/reports/rsna_cv_summary.json) にあります。

- 加重マルチラベル対数損失: **0.05346 ± 0.00624**（5 fold）
- 平均AUC: **0.98815 ± 0.00311**（5 fold）
- 誤り検出AUROC（`any`）: **0.9424 ± 0.0200**（10シード）
- ECE（`any`）: **0.0231 ± 0.0034**（10シード）

評価条件は次のとおりです。

- 各foldと各シードは、`limit_images=8000` で抽出したデータ（交差検証では、前処理済みデータにない画像を除いた6,852枚）で1エポックだけ追加学習しています。
- 初期重みは、同じ前処理済みの学習データの98%の検査（645,793スライスのうち632,808スライス）で学習済みです。数値は追加学習の安定性を示すもので、未知の検査に対する性能ではありません。
- 温度スケーリングはECEの評価と同じ検証データで推定しているため、校正の指標は楽観的に出ます。
- 研究とポートフォリオのための実装で、医療機器ではありません。

詳しくは [評価ノート](../docs/evaluation_notes_ja.md) を参照してください。

## この公開物の強み

- `split_by=study` のグループ分割と監査スクリプト
- 分割前テンソルのハッシュによる重複除去でリークを抑制
- `meta.json` / `log.jsonl` / `split_stats` による再現性監査
- `subset_fingerprint_sha256` による対象集合の同一性チェック

![分割の考え方](../docs/assets/split_study.svg)

## すぐ使うリンク

| 用途 | 場所 |
|---|---|
| 最短の動作確認 | `python ../scripts/smoke_test.py --use_dummy_data` |
| 詳細（日本語） | [../core/pipeline/README.md](../core/pipeline/README.md) |
| 詳細（英語） | [../core/pipeline/README_en.md](../core/pipeline/README_en.md) |
| 監査ショートガイド | [./AUDIT_GUIDE.md](./AUDIT_GUIDE.md) |
| APIデモ | [../docs/api_demo_ja.md](../docs/api_demo_ja.md) |
| 引用 | [../CITATION.cff](../CITATION.cff) |
| リリースノート | [../docs/releases/v0.6.0-rsna_ja.md](../docs/releases/v0.6.0-rsna_ja.md) |
| ロードマップ | [../ROADMAP.md](../ROADMAP.md) |

![公開範囲](../docs/assets/included_excluded.svg)

## MLflow 追跡スキーマ

`--mlflow` を有効にした場合、この公開リポジトリでは 3 本のポートフォリオ用リポジトリで共通の追跡スキーマを使います。

- 共通の run tag: `repo_name`, `task_type`, `model_family`, `tracking_schema=public_portfolio_v1`
- 共通の artifact グループ:
  - `run_metadata/`: `meta.json` と、存在する場合は設定スナップショットやタスク固有の JSON
  - `training_trace/`: `log.jsonl`
  - `checkpoints/`: `last.pt`, `best.pt`, およびタスク固有の best 系チェックポイント
- 目的: セグメンテーション系と分類系の run を同じ見方で追えるようにしつつ、本格的な本番用 MLOps 基盤を主張しないこと

## 現行のポートフォリオ用スナップショット

現行のポートフォリオ用スナップショットは、次のタグに対応します。

✅ `v0.6.0-rsna`

リポジトリは継続的に開発中です。

この公開フォルダには `Datasets/`, `runs/`, `results/` を同梱していません。実データでの再現コマンドは上記の詳細 README を参照してください。
