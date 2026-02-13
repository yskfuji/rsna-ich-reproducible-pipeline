# RSNA ICH 監査ガイド（公開物内完結）

このガイドは、監査者が `github_public/` 配下だけで再現性・分割設計・リーク対策を確認するための入口です。

## 1) まず読むファイル

1. `../core/ToReBrain-pipeline/README.md`（日本語）
2. `../core/ToReBrain-pipeline/README_en.md`（英語）

## 2) 監査で見る主要成果物

- 学習run配下: `meta.json`, `log.jsonl`
- 分割監査: `tools/audit_rsna_split.py`, `tools/audit_rsna_slice_leakage.py`
- de-dup監査: `tools/audit_rsna_dedup_effect.py`
- subset同一性: `subset_fingerprint_sha256`（`meta.json`）

## 3) 最小監査観点

- `split_by=study` で group 交差が 0
- de-dup 後に exact duplicate 交差が 0
- 再現runで `split_stats` と `val_*` が許容差内で一致
- 必要に応じて `subset_fingerprint_sha256` が一致

## 4) 補足

この公開物には `Datasets/`, `runs/`, `results/` は同梱していません。データは監査者側で準備してください。
