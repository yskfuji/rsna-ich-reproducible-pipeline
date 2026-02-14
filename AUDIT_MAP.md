# RSNA 監査マップ

この公開物は RSNA 監査向けに最小導線で整理しています。

## 1. 読む順番

1. `./rsna_ich/AUDIT_GUIDE.md`
2. `./core/pipeline/README.md`（または `README_en.md`）
3. `./core/pipeline/tools/` の監査スクリプト

## 2. 主な監査ポイント

- `split_by=study` の group 交差 0
- de-dup 後の exact duplicate 交差 0
- run間の `split_stats` / `val_*` の再現性
- `subset_fingerprint_sha256` の一致確認

## 3. 除外物

- `Datasets/`
- `runs/`
- `results/`
- `logs/`