# RSNA API デモ

この公開リポジトリには、RSNA 用の軽量な FastAPI デモ導線を追加しています。

## 目的

- RSNA task を型付き HTTP schema としてどう公開するかを示す
- 公開ポートフォリオ内で MLflow / API 方向の設計意図を見せる
- 保護対象の医療データや本番 checkpoint を同梱せずに API 面を確認できるようにする

## 起動ファイル

- [core/pipeline/serve_rsna_ich_api.py](../core/pipeline/serve_rsna_ich_api.py)

## エンドポイント

- `GET /health`
- `GET /v1/models/{alias_or_version}`
- `POST /v1/infer/ich_classification`

## このデモでできること

- RSNA task / model family の metadata を型付きで返す
- `ich_classification` の構造化レスポンスを返す
- `any` 確率を subtype 最大値以上になるよう正規化する

## このデモでまだやっていないこと

- 保護対象の医療データ読込
- 本番デプロイ相当の運用保証
- 実 MLflow model URI や checkpoint alias の解決

## ローカル起動

```bash
cd core/pipeline
python serve_rsna_ich_api.py
```

起動後に `http://127.0.0.1:8000/docs` を開くと schema を確認できます。

ポート `8000` が使用中なら、たとえば次のように別ポートで起動できます。

```bash
cd core/pipeline
API_PORT=8011 python serve_rsna_ich_api.py
```

## 次の拡張候補

次の公開向け拡張は、demo response を実 checkpoint または MLflow 参照に置き換えることです。