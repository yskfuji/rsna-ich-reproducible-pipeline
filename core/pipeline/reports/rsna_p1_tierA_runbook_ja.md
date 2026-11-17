# RSNA ICH（P1 / Tier A）実験ランブック

目的: Kaggle RSNA ICH の加重マルチラベル対数損失 ≤ 0.060（Tier A）を最短で狙う。

## 前提

- RSNAデータの配置は `<RSNA_ROOT>/stage_2_train.csv` および `<RSNA_ROOT>/stage_2_train/*.dcm` とする。
- 実行は `core/pipeline/` から行う（スクリプトは自身の位置から `core/pipeline/` に移動する）。
- デバイスは `TORCH_DEVICE`（例: `mps`）で選択する。

## 実行（推奨順）

最小の A/B 比較で、不均衡対策（`pos_weight` と sampler）および 2.5D と強バックボーンの効果を確認する。

```zsh
cd core/pipeline
chmod +x scripts/run_rsna_p1_tierA_ab.zsh

# 例: MPS
./scripts/run_rsna_p1_tierA_ab.zsh <RSNA_ROOT> results/rsna_p1 mps
```

## 確認すべき成果物

各実行ディレクトリに次が出力される。

- `meta.json`: 実験条件（分割、アーキテクチャ、`stack_slices`、窓処理など）
- `log.jsonl`: 各エポックの指標
- `best_wlogloss.pt`: 主指標（`val_logloss_weighted`）が最小のチェックポイント

最重要の指標は次である。

- `val_logloss_weighted`（Kaggle の加重対数損失。小さいほど良い）

併記する指標は次である。

- `val_auc_mean`（大きいほど良いが、Tier A 判定は対数損失を優先する）

## 次の作業

- `val_logloss_weighted` が最小の実行結果を採用する。
- その実行結果の `best_wlogloss.pt` を用い、`src/inference/predict_rsna_ich_submission.py` で提出用ファイルを作成する。
  - `enforce_any_max` は有効を推奨する（`any` の論理整合）。
  - ただし `enforce_any_max` は `val_logloss_weighted` を悪化させうる。
  - 必要に応じて `log.jsonl` の `val_logloss_weighted_raw`（後処理前）も併記して診断する。

最短手順は、最良の実行結果を自動選択して提出用ファイルを生成することである。

```zsh
cd core/pipeline
chmod +x scripts/make_rsna_submission_from_best.zsh
./scripts/make_rsna_submission_from_best.zsh <RSNA_ROOT> results/rsna_p1 submission.csv mps
```
