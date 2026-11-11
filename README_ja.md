# rsna-ich-reproducible-pipeline

**言語:** 日本語 | [英語版](README.md)

[![ci](https://github.com/yskfuji/rsna-ich-reproducible-pipeline/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/yskfuji/rsna-ich-reproducible-pipeline/actions/workflows/ci.yml)

RSNA ICH challenge 向けの、**再現可能な頭蓋内出血の分類パイプライン**です。監査しやすいドキュメント、校正解析、リークを避けるグループ分割評価を含みます。

## 5分で確認する場所

| 確認したいこと | 場所 |
|---|---|
| DICOMの読み込み、HUウィンドウ処理、重複除去 | [core/pipeline/src/datasets/rsna_ich_dataset.py](core/pipeline/src/datasets/rsna_ich_dataset.py) |
| 学習ループ、加重損失、検査単位のGroupKFold | [core/pipeline/src/training/train_rsna_cnn2d_classifier.py](core/pipeline/src/training/train_rsna_cnn2d_classifier.py) |
| 3D U-Netエンコーダを使った分類器 | [core/pipeline/src/models/unet3d_encoder_classifier.py](core/pipeline/src/models/unet3d_encoder_classifier.py) |
| 提出用の推論 | [core/pipeline/src/inference/predict_rsna_ich_submission.py](core/pipeline/src/inference/predict_rsna_ich_submission.py) |
| 学習側と検証側のリーク監査 | [core/pipeline/tools/audit_rsna_split.py](core/pipeline/tools/audit_rsna_split.py) |
| MC Dropout、ECE、温度スケーリング、AURC | [core/pipeline/tools/eval_rsna_uncertainty.py](core/pipeline/tools/eval_rsna_uncertainty.py) |
| 下の表の元になったfold別・シード別の値 | [core/pipeline/reports/rsna_cv_summary.json](core/pipeline/reports/rsna_cv_summary.json) |
| 評価条件、制約、参考文献 | [docs/evaluation_notes_ja.md](docs/evaluation_notes_ja.md) |
| 用語の対応表 | [docs/glossary_ja.md](docs/glossary_ja.md) |

**クイックリンク**
- 英語版: [rsna_ich/README_en.md](rsna_ich/README_en.md)
- 日本語版: [rsna_ich/README.md](rsna_ich/README.md)
- 実験詳細: [core/pipeline/README.md](core/pipeline/README.md)
- API デモ概要: [docs/api_demo_ja.md](docs/api_demo_ja.md)
- 再現性チェックリスト: [docs/reproducibility_checklist_ja.md](docs/reproducibility_checklist_ja.md)
- レビューの見始め方: [英語版スナップショット案内](docs/releases/v0.6.0-rsna.md) | [日本語版スナップショット案内](docs/releases/v0.6.0-rsna_ja.md)
- GitHub About 欄の説明文: [英語版](docs/github_about.md) | [日本語版](docs/github_about_ja.md)
- 引用情報: [CITATION.cff](CITATION.cff)
- リリースノート原稿: [英語版](docs/releases/v0.6.0-rsna.md) | [日本語版](docs/releases/v0.6.0-rsna_ja.md)
- ロードマップ: [ROADMAP.md](ROADMAP.md)
- リポジトリ整合性監査（2026年11月）: [docs/audit/2026-11-repo-integrity.md](docs/audit/2026-11-repo-integrity.md)

## このリポジトリでできること

- RSNA 頭蓋内出血の分類モデルの学習 / 評価ワークフロー
- `split_by=study` による study 単位のグループ分割とリーク監査
- `any` ラベルに対する校正評価と不確実性評価
- FastAPI による公開用 API スキーマ確認デモ
- 外部レビュー向けに整理したポートフォリオ用ドキュメント
- 実データなしで公開物が正しく動くかを確かめる簡易動作確認

## 想定している読者

- 医療AI実装を確認したい採用担当
- 監査しやすい医用画像ベースラインを見たい ML エンジニア
- 再現性重視の CT 分類プロジェクト構成を探している研究者

## 3分で分かる概要

![RSNA パイプライン構成図](docs/assets/architecture.svg)

![RSNA リポジトリ構成図](docs/assets/repo_map.svg)

![RSNA 指標サマリー](docs/assets/results_snapshot.svg)

### 代表指標

表記は「平均 ± 標本標準偏差（ddof=1）」です。fold別・シード別の値は [core/pipeline/reports/rsna_cv_summary.json](core/pipeline/reports/rsna_cv_summary.json) にあり、この表の数値はすべてCIがこのファイルから計算し直して照合しています。

| 指標 | 値 | n | 意味 |
|---|---:|---:|---|
| 加重マルチラベル対数損失（低いほど良い） | 0.05346 ± 0.00624 | 5 fold | Kaggleコンペティションの評価指標 |
| 平均AUC（高いほど良い） | 0.98815 ± 0.00311 | 5 fold | 6ラベルを通した順位付けの性能 |
| 誤り検出AUROC（`any`、高いほど良い） | 0.9424 ± 0.0200 | 10シード | 不確実性で誤った予測をどれだけ見分けられるか |
| ECE（`any`、低いほど良い） | 0.0231 ± 0.0034 | 10シード | 予測確率の校正の良さ |

> **数値は、次の評価条件とあわせて読んでください。**
> - 各foldと各シードは、`limit_images=8000` で抽出したデータ（交差検証では、前処理済みデータにない画像を除いた6,852枚）で**1エポック**だけ追加学習しています。
> - すべての実行は同じ**初期重み**から始めています。この初期重みは、同じ前処理済みの学習データの98%の検査（645,793スライスのうち632,808スライス）で学習済みです。そのため、検証画像の多くは追加学習の前にすでに学習されていたと考えられます。数値は追加学習の安定性を示すもので、未知の検査に対する性能ではありません。
> - 温度スケーリングは、ECEの評価と**同じ検証データ**で推定しています。そのため、校正の指標は楽観的に出ます。
> - 分割は検査単位で、患者単位ではありません。
> - 研究とポートフォリオのための実装で、**医療機器ではありません**。
>
> 詳しい条件と参考文献は [docs/evaluation_notes_ja.md](docs/evaluation_notes_ja.md) にあります。医用画像は公開物に含めていません。

## 最短の確認方法

### 1. 実データなしで動作確認

```bash
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.14.1 torchvision==0.29.1
pip install -r core/pipeline/requirements.txt
python scripts/smoke_test.py --use_dummy_data
python scripts/check_imports.py
python scripts/check_metric_claims.py
```

### 2. 配布物マニフェストを生成

```bash
cd core/pipeline
python tools/make_manifest.py
```

### 3. 実データで学習 / 評価

- 日本語詳細: [core/pipeline/README.md](core/pipeline/README.md)
- 英語版の詳細ガイド: [core/pipeline/README_en.md](core/pipeline/README_en.md)

### 4. API デモを確認

```bash
cd core/pipeline
python serve_rsna_ich_api.py
```

ポート `8000` が使用中なら `API_PORT=8011 python serve_rsna_ich_api.py` のように変更できます。

詳細: [docs/api_demo_ja.md](docs/api_demo_ja.md)

## 含まれるものと含まれないもの

このリポジトリに含まれるものは次のとおりです。

- ソースコード
- 設定ファイル
- 監査 / 評価ドキュメント
- 静的図表とリリースノート原稿


次のものは含まれません。

- RSNAの画像とラベル（Kaggleから、コンペティションの非営利利用の条件に従って入手してください）
- `Datasets/`
- `runs/`
- `results/`
- `logs/`
- 学習済みの重み（`*.pt`）

## 現行のポートフォリオ用スナップショット

開発は継続中ですが、現行のポートフォリオ用固定スナップショットは次のタグです。

✅ `v0.6.0-rsna`

## 引用

[CITATION.cff](CITATION.cff) を参照してください。

## コミットメッセージの規約

今後の変更は Conventional Commits（`type: summary`）で揃えます。

- `fix: leakage check in group split`
- `feat: add calibration evaluation`
- `refactor: manifest validation logic`
- `docs: evaluation protocol clarification`
