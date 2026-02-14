# RSNA ICH（P1 / Tier A）実験ランブック

目的: Kaggle RSNA ICH の **weighted multi-label log loss ≤ 0.060（Tier A）** を最短で狙う。

## 前提
- RSNAデータ配置: `<RSNA_ROOT>/stage_2_train.csv` と `<RSNA_ROOT>/stage_2_train/*.dcm`
- 実行は `pipeline/` から（スクリプトが `cd` します）
- デバイスは `TORCH_DEVICE`（例: `mps`）で選択

## 実行（推奨順）
最小のA/Bで「不均衡対策（pos_weight vs sampler）」と「2.5D + 強バックボーン」の効きを確認します。

```zsh
cd /Users/yusukefujinami/ToReBrain/pipeline
chmod +x scripts/run_rsna_p1_tierA_ab.zsh

# 例: MPS
./scripts/run_rsna_p1_tierA_ab.zsh <RSNA_ROOT> results/rsna_p1 mps
```

## 何を見るか（重要）
各 run ディレクトリに以下が出ます。

- `meta.json`: 実験条件（split/arch/stack_slices/windows など）
- `log.jsonl`: 各epochの指標
- `best_wlogloss.pt`: **主KPI（val_logloss_weighted）が最小**のチェックポイント

最重要KPI:
- `val_logloss_weighted`（=Kaggle weighted logloss。**小さいほど良い**）

併記:
- `val_auc_mean`（大きいほど良いが、Tier A判定は logloss 優先）

## 次アクション
- `val_logloss_weighted` が一番小さい run を採用
- その run の `best_wlogloss.pt` を使って `src/inference/predict_rsna_ich_submission.py` で submission を作る
  - `enforce_any_max` はON推奨（anyの論理整合）

最短（best run を自動選択して submission 生成）:

```zsh
cd /Users/yusukefujinami/ToReBrain/pipeline
chmod +x scripts/make_rsna_submission_from_best.zsh
./scripts/make_rsna_submission_from_best.zsh <RSNA_ROOT> results/rsna_p1 submission.csv mps
```
