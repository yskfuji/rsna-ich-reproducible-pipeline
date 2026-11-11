# RSNA API デモ

本公開リポジトリは、RSNA課題向けの軽量なFastAPIデモを含む。

## 目的

- RSNAタスクを型付きHTTPスキーマとして公開する設計を示す。
- 公開ポートフォリオ内で、MLflowおよびAPI連携を見据えた設計意図を示す。
- 機微な医療データや本番チェックポイントを同梱せずに、API面を確認できるようにする。

## 起動ファイル

- [core/pipeline/serve_rsna_ich_api.py](../core/pipeline/serve_rsna_ich_api.py)

## エンドポイント

- `GET /health`
- `GET /v1/models/{alias_or_version}`
- `POST /v1/infer/ich_classification`

## 本デモで確認できること

- RSNAのタスク名およびモデル種別の情報を型付きで返す。
- `ich_classification` の構造化レスポンスを返す。
- `any` の確率を各subtypeの最大値以上となるよう正規化する。

## 本デモで扱わないこと

- 機微な医療データの読込み
- 本番デプロイ相当の運用保証
- 実際のMLflow model URIやcheckpoint aliasの解決

## ローカル起動

```bash
cd core/pipeline
python serve_rsna_ich_api.py
```

起動後に `http://127.0.0.1:8000/docs` を開くとスキーマを確認できる。

ポート `8000` が使用中の場合は、次のように別ポートで起動する。

```bash
cd core/pipeline
API_PORT=8011 python serve_rsna_ich_api.py
```

## 今後の拡張候補

公開向けの次段拡張として、デモ応答を実際のチェックポイントまたはMLflowのモデル参照へ置き換えることが挙げられる。
