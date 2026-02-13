# RSNA Intracranial Hemorrhage (Kaggle 2019)

このフォルダは RSNA ICH 公開物の入口です。

- 主要コード: `../core/ToReBrain-pipeline/`
- 実験README（JP/EN）:
  - `../core/ToReBrain-pipeline/README.md`
  - `../core/ToReBrain-pipeline/README_en.md`
- 監査用ショートガイド:
  - `./AUDIT_GUIDE.md`

## ポイント

- `split_by=study` の group split + 監査スクリプト
- split前のテンソルhash de-dup（リーク対策）
- 再現性成果物: `meta.json` / `log.jsonl` / `split_stats`
- subset 同一性の監査: `subset_fingerprint_sha256`

この公開フォルダでは `Datasets/`, `runs/`, `results/` は同梱しません。
データ準備・実行コマンドは上記 README を参照してください。
