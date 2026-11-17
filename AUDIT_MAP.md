# RSNA 監査マップ

本書は、本リポジトリの公開物を検証する読者のために、確認すべき箇所とその順序を示す。

## 1. 読む順番

1. `./rsna_ich/AUDIT_GUIDE.md`
2. `./core/pipeline/README.md`（又は `README_en.md`）
3. `./core/pipeline/tools/` の監査スクリプト

## 2. 主な監査ポイント

- `split_by=study` のグループ交差 0
- 重複除去後の完全重複交差 0
- run間の `split_stats` / `val_*` の再現性
- `subset_fingerprint_sha256` の一致確認
- API デモが公開スキーマとして自己完結していること（`core/pipeline/serve_rsna_ich_api.py`）

## 3. API デモの見る場所

1. `./docs/api_demo_ja.md`
2. `./core/pipeline/serve_rsna_ich_api.py`
3. `./core/pipeline/src/inference/serve_rsna_ich_api.py`

## 4. 除外物

- `Datasets/`
- `runs/`
- `results/`
- `logs/`

## 5. 合格判定

- 評価と公開物の訂正記録: `./docs/audit/2026-11-evaluation-corrections.md`
