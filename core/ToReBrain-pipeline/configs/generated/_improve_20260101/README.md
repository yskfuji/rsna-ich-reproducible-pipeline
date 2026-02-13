# Improve sweep (2026-01-01)

Fold0 / ts222 / patch96 の「改善スクリーニング」用config（まずは100epochで当たりを探す）。

## Run

```bash
cd /Users/yusukefujinami/ToReBrain/ToReBrain-pipeline

/opt/anaconda3/envs/medseg_unet/bin/python tools/run_train_queue.py \
  --python /opt/anaconda3/envs/medseg_unet/bin/python \
  --repo /Users/yusukefujinami/ToReBrain/ToReBrain-pipeline \
  --configs \
    configs/generated/_improve_20260101/medseg_3d_unet_improve1_fg080_ccinv_a1_dicebce_pw1_e100_kfold5_f0_ts222.yaml \
    configs/generated/_improve_20260101/medseg_3d_unet_improve2_fg080_ccinv_a1_dicebce_pw2_e100_kfold5_f0_ts222.yaml \
    configs/generated/_improve_20260101/medseg_3d_unet_improve3_fg080_ccinv_a1_tversky_a03_b07_e100_kfold5_f0_ts222.yaml
```

## What changes

- `data.foreground_prob=0.8`
- `data.fg_component_sampling=inverse_size`（小さい病変CCを優先）
- Loss比較：`dice_bce(pos_weight=1/2)` vs `tversky(alpha=0.3,beta=0.7)`
