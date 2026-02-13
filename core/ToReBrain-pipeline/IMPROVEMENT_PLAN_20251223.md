# ISLES-2022 (ToReBrain) 改善しない原因（優先度順）と改善プラン

## いま観測できている事実（e20: tversky_ohem_fgccinv_a1）
- best threshold が `thr=0.8` に張り付き（none/flipとも）。
  - `tta=none`: global=0.395 / le_3mm=0.327 / det_le=0.789 / gt_3mm=0.613
  - `tta=flip`: global=0.391 / le_3mm=0.316 / det_le=0.737 / gt_3mm=0.629
- `tta=flip` は le_3mm を悪化（ピーク潰れ由来のFNが増えている可能性）。
- `thr=0.8` 時点では FP は大きくはない（mean_fp_vox ~170）一方、voxel recall が低い（~0.39）＝「取りこぼし寄り」。

## 最優先の根本原因（要修正）
### 1) 評価の resample が“軸順ミスマッチ”で誤っている可能性（高）
- `IslesVolumeDataset` が返す volume/mask は **(C, Z, Y, X)/(Z, Y, X)**。
- しかし `evaluate_isles.py` の `_resample_to_max_zoom_mm()` は元々、
  - volume を (C, X, Y, Z)
  - zooms を (X, Y, Z)
  と仮定しており、**実データ軸と不一致**。
- この不一致は「厚スライスだけ改善した/薄スライスでは効かない」のような評価上の挙動を“見かけ上”作ります。

対応:
- `src/evaluation/evaluate_isles.py` の resample を **(C, Z, Y, X) と zooms(X,Y,Z) を正しく対応**させる（修正済み）。
- 以降、`--resample-max-zoom-mm 2.0` の結果は **再評価が必須**。

> まずここを直さないと、以後の「resampleが効いた/効かない」議論が崩れます。

## 改善しない原因候補（確度が高い順）
### 2) 監督信号の“薄スライス小病変”が学習で支配されていない（高）
- le_3mm の病変は小さく、1パッチ内の陽性率が極端に低くなりやすい。
- 現状の e20 は `foreground_prob=0.33` + `fg_component_sampling=inverse_size` だが、
  - `patches_per_volume=1` で「1 volume あたり 1パッチ」
  - `force_fg_prob=0.0` で **FG強制は一切なし**
  → “小病変を確実に踏む”確率は十分ではない可能性がある。

### 3) all-positive test（真の陰性なし）により FP/閾値/損失のトレードが不安定（高）
- testが全例陽性だと、モデルは「多少FPを出してでも検出」へ寄りやすいが、
  実際には best thr が 0.8 に寄っており、逆に“強いピーク以外を捨てる”方向に調整されている。
- これは「背景を学ぶ（空パッチ/負例）」が足りない、または“確信度校正”が崩れているサイン。

### 4) 閾値候補の設計ミスマッチ（中）
- val で `val_threshold_candidates` が 0.3〜0.7 までで、test best が 0.8。
- 「モデル選択（best.pt）と、本当に良い閾値」がズレると改善が見えづらい。

### 5) モデル容量/受容野/解像度が足りない（中）
- 3D U-Net base_ch=48 / patch 48×48×24 は軽め。
- le_3mm の微小病変は、
  - コントラストが薄い
  - 形状が小さい
  - 周辺文脈が重要
 で、単一スケールの訓練だと天井が早い。

## 「確実性が高い」改善プラン（段階的・検証可能）

### Phase 0（必須）: 評価の整合性を固める
1. resample修正後に、同じモデルで `tta=none/flip` を再評価
   - 目的: これまでの改善/悪化が“評価バグ由来”でないか確定
   - 出力: `results/diag/test_eval_*/*/summary.json` の global/le_3mm/gt_3mm と FP 指標
2. 今後は **resample有り/無し**を必ずセットで報告

### Phase 1（最優先で効きやすい）: 学習サンプリングを「FGもBGも必ず混ぜる」へ
狙い: 小病変の踏み外し（FN）と、全体確信度の崩れ（thr=0.8張り付き）を同時に改善。

推奨変更（新YAMLを切る）:
- `data.patches_per_volume: 2`
- `data.patches_force_one_bg: true`  （2枚目をBG強制）
- `data.ensure_empty_bg_prob: 1.0`（BG強制時は必ず空パッチ探索）
- `data.bg_min_dist: 12`（目安: min(patch)/4〜1/2。まずは12）
- `data.bg_min_dist_relax: true`
- `data.foreground_prob: 0.33` は維持（急に上げない）
- `data.force_fg_prob: 0.2`（“FG強制”を少量だけ入れて、小病変を踏み外しにくくする）
  - 0.0→0.2程度から。FPが増えるなら戻す。

損失（まずは現状維持でOK）:
- `tversky_ohem_bce` は維持しつつ、次のどちらかを比較
  - A: `beta` を少し上げる（例: 0.70〜0.75）＝FNをより重く
  - B: `neg_fraction` を少し上げる（例: 0.05）＝背景の校正強化

評価:
- `tta=none` 固定（flipは今は入れない）
- le_3mm: `mean_dice` + `det_le`（検出率）を第一指標
- 併せて `mean_fp_vox` と `mean_pred_vox` で崩壊検知

### Phase 2（効けば伸びる）: Deep supervision + 受容野の増加
狙い: 微小病変の特徴抽出を早い層から安定させる。
- `train.deep_supervision: true`
- `train.deep_supervision_aux2_weight: 0.5`
- `train.deep_supervision_aux3_weight: 0.25`
- 可能なら patch を少し大きく
  - 例: `patch_size: [64,64,32]`, `val_stride: [32,32,16]`（メモリが許せば）

### Phase 3（最後の切り札）: 解像度/データ側の再設計
狙い: le_3mm の小病変が「入力解像度・前処理で潰れている」ケースを潰す。
- 前処理でのリサンプル/補間/クリップが小病変のコントラストを落としていないか点検
- 可能なら、
  - z方向だけでなく in-plane 解像度も統一（1mm等）
  - ラベル補間が nearest になっているか

## 次にやるべき“最短ルート”
1. resample修正後の再評価（今ある best.pt でOK）
2. Phase1 の pv2bg + force_fg 少量版を e20（同条件）で学習→test評価
3. それでも le_3mm が伸びない場合のみ Phase2 に進む
