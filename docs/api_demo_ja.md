# RSNA API デモ

この公開リポジトリには、RSNA 用の軽量な FastAPI デモを追加しています。

## 目的

- RSNA タスクを型付き HTTP スキーマとしてどう公開するかを示す
- 公開ポートフォリオ内で、MLflow と API 連携を見据えた設計意図を示す
- 機微な医療データや本番チェックポイントを同梱せずに API 面を確認できるようにする

## 起動ファイル

- [core/pipeline/serve_rsna_ich_api.py](../core/pipeline/serve_rsna_ich_api.py)

## エンドポイント

- `GET /health`
- `GET /v1/models/{alias_or_version}`
- `POST /v1/infer/ich_classification`

## このデモで確認できること

- RSNA のタスク名やモデル種別の情報を型付きで返す
- `ich_classification` の構造化レスポンスを返す
- `any` の確率を各 subtype の最大値以上になるよう正規化する

## このデモでまだ扱っていないこと

- 機微な医療データの読み込み
- 本番デプロイ相当の運用保証
- 実際の MLflow model URI や checkpoint alias の解決

## ローカル起動

```bash
cd core/pipeline
python serve_rsna_ich_api.py
```

起動後に `http://127.0.0.1:8000/docs` を開くとスキーマを確認できます。

ポート `8000` が使用中なら、たとえば次のように別ポートで起動できます。

```bash
cd core/pipeline
API_PORT=8011 python serve_rsna_ich_api.py
```

## 次の拡張候補

次の公開向け拡張候補は、デモ用の応答を実際のチェックポイントまたは MLflow のモデル参照に置き換えることです。