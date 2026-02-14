# RSNA Intracranial Hemorrhage (Kaggle 2019) — 再現性つき実験README（ポートフォリオ向け）

英語版: [README_en.md](README_en.md)

このフォルダ（pipeline/）は RSNA ICH の学習・評価・推論を「**再現できる形で**」回すための最小パイプラインです。
読者が短時間で判断できるよう、**再現性 / 分割設計（リーク対策）/ 比較軸 / Ablation** を先頭にまとめています。

---

## TL;DR（このREADMEで分かること）

- 再現性（この配布物内）: 記録済みの成果物（`meta.json` / `log.jsonl`、`split_stats` を含む）と固定CLIコマンドにより検証可能（例: `results/rsna_convnext25d_ft_repro_val05_short_20260210_101836_run1` と `_run2`）。
- 参照コミット（上流・参照用）: `9bc2684fe0b564c792025b52694f9c4fe1a0d32d`（この配布物のコードスナップショットをエクスポートした時点の上流リポジトリのコミットID）。
- 上流リポジトリURL: 配布物は git メタデータ非同梱のため、上流URL/履歴は配布物単体では追跡不能です（`.git` なし）。
- 配布物の完全性（任意）: `python tools/make_manifest.py` でファイル一覧（path + sha256）のマニフェストを生成し（例: `--out MANIFEST.sha256`）、この配布物を指紋化できます。

- **頑健性（偶然に依存していないかの確認）**: `split_by=study` の **GroupKFold(5)** で `val_logloss_weighted = 0.05346 ± 0.00624` を確認（`epochs=1` / `kept=6852` の fast CV）
- **再現性**: 同一 seed・同一引数で 2回実行し、`split_stats` と `val_*` が再現可能（この環境では **完全一致（abs_diff=0）** を観測。一般には backend により `abs_diff <= 1e-6` 程度で検証推奨）
- **評価設計（リーク対策）**: `split_by=study` の **Group split** を採用（同一 Study が train/val に跨がない）
- **実務上の価値**: リークしやすい問題設定に対し、**group split の監査スクリプト**で「train/val の group 交差が 0」を機械的に検証できる
- **リーク監査の現状（2026-02-14）**: 10 seed 監査で Study/Series 交差は 0、さらに **split前のテンソルhash de-dup（既定ON）後は train/val の exact duplicate 交差も 0** を確認
- **指標の意味**: Kaggle本番の重み付き multi-label logloss（`val_logloss_weighted`）を主軸に、AUCは補助として扱う
- **Uncertainty / Calibration（10 seeds, `split_by=study`, de-dup有効, retrained with dropout）**:
  - Error-detection AUROC(any): **0.9424 ± 0.0190**
  - ECE(any): **0.0231 ± 0.0032**
  - Brier(any): **0.0209 ± 0.0054**
  - AURC(weighted logloss): **0.00837 ± 0.00214**
  - coverage=0.8 accuracy(any) gain: **+2.53 ± 0.89 pp**
  - （同条件 baseline 比）ECE **-0.0091**, Brier **-0.0040**, AURC **-0.00139**, AUROC **+0.0046**
- **温度スケーリングの評価境界（重要）**: 本READMEの uncertainty 数値は `--fit-temperature` を同一 holdout val で実行した値（fit集合=eval集合）であり、ECE/Brier は楽観側に寄る可能性があります。厳密評価では calib/eval 分割（または別holdout）で再計測してください。
- **監査フック（subset 同一性）**: `split_stats` 一致に加え、採用 `image_id` のソート済み一覧 SHA256（`subset_fingerprint_sha256`）を **`meta.json` に記録（実装済み）**。また `tools/eval_rsna_uncertainty.py` の出力JSONにも `adopted_subset_fingerprint_sha256` / `val_subset_fingerprint_sha256` を出します。
  - `subset_fingerprint_sha256` 定義: 採用 `image_id` を昇順ソートし、`\n` 区切りで連結した UTF-8 文字列の SHA256(hex)
  - `adopted_subset_*` は limit_images 等で採用された全体 subset、`val_subset_*` はその subset のうち validation に入った `image_id` subset
- すぐ再現（holdout, 1 epoch）:

```bash
# すぐ再現（holdout, 1 epoch）
TORCH_DEVICE=mps python train_rsna_cnn2d_classifier.py train --rsna-root ... --preprocessed-root ... --out-dir results/repro_demo --limit-images 8000 --val-frac 0.05 --split-by study --seed 0 --epochs 1 --num-workers 0 --no-aug --no-scheduler --arch convnext_tiny --pretrained --init-from "results/rsna_convnext25d_final2_full_mps_nosamp_nopw_20260202_222608/best.pt"

# --init-from が使えない場合:
# - 省略（ImageNet-pretrained から学習開始）、または
# - results/**/best*.pt のいずれかを指定
```

  → 出力: `meta.json`, `log.jsonl`, checkpoint は `results/repro_demo/` 配下（完全なコマンドは §2.2）。
  ※実パス例は §2.1「データ例」を参照（rsna_root / preprocessed_root）。

---

## 1. 指標サマリ（今回の結果）

### 1.1 GroupKFold(5) CV（study単位）での頑健性チェック

「validation が小さくて偶然良かったのでは？」という懸念を解消するため、`split_by=study`（StudyInstanceUID）で **GroupKFold(5)** を回しました。

条件（短縮CV）:
- `limit_images=8000`（preprocessed DB 欠損除外後: `kept=6852`）
- `epochs=1`（方向性確認の fast CV）
- `init_from=best.pt`（fine-tune）
  - 使用した `best.pt` は別 run の validation に対して保存された checkpoint です（詳細は「2.2 実行コマンド」参照。今回の CV の val を見て選んでいません）

母数（val 側の目安; fold 間の mean±std）:
- `n_val_studies = 1178.0 ± 1.0`
- `n_val_images = 1370.4 ± 9.6`
- `pos_rate_any = 0.1496 ± 0.0030`（any 陽性率）

CV結果（val）:
- `val_logloss_weighted`: **0.05346 ± 0.00624**（min 0.04732 / max 0.06347）
- `val_auc_mean`: **0.98815 ± 0.00311**（min 0.98441 / max 0.99188）

集計方法（再現可能な定義）:
- 各 fold の `log.jsonl` **最終行**から `val_logloss_weighted` / `val_auc_mean` を抽出し、fold 間で mean±std（および min/max）を計算
- まとめ出力は `tools/summarize_cv.py` で自動集計できます（下記 2.3 参照）

補足（ばらつきの解釈）:
- fold 間の変動は、症例構成・撮像条件・アーチファクト等の **ドメイン差**が残る限り自然です（それでも平均が 0.053 台にまとまっているのがポイント）。

実行結果ディレクトリ:
- results/rsna_convnext25d_ft_gkfold5_e1_lim8k_20260210_150447

---

### 1.2 短縮CV → 本学習へのブリッジ（代表foldでの学習曲線）

短縮CV（epochs=1）だけだと「フルに回したらどうなる？」が残るので、代表として fold0 を `epochs=5` で回し、指標の推移を出しました。

run:
- results/rsna_convnext25d_ft_gkfold5_fold0_epochs5_20260210_160037

| epoch | val_logloss_weighted | val_auc_mean | val_loss_plain |
|---:|---:|---:|---:|
| 1 | 0.04732 | 0.98608 | 0.04313 |
| 2 | 0.04639 | 0.98695 | 0.04235 |
| 3 | 0.04629 | 0.98777 | 0.04230 |
| 4 | 0.04672 | 0.98843 | 0.04264 |
| 5 | 0.04772 | 0.98839 | 0.04368 |

メモ:
- この fold では `val_logloss_weighted` の最良は **epoch=3（0.04629）** で、それ以降は悪化しているため、早めの打ち切り（early stop）的な判断が妥当です。

---

### 1.3 単発 holdout（val_frac=0.05）と再現性（run1 vs run2）

対象 run:
- results/rsna_convnext25d_ft_repro_val05_short_20260210_101836_run1
- results/rsna_convnext25d_ft_repro_val05_short_20260210_111323_run2

主要指標（val）:
- `val_logloss_weighted = 0.0555525`
- `val_auc_mean = 0.9915841`

再現性の確認結果:
- 同一 seed・同一引数で、`split_stats` と `val_*` が **完全一致（abs_diff=0）**
  - ただし PyTorch / MPS / 演算カーネルの組み合わせ次第で厳密一致が崩れる可能性はあるため、汎用的には `abs_diff <= 1e-6` 程度で確認するのが安全です。
  - 例: CPU/CUDA で `--num-workers > 0` や AMP を有効化すると、厳密一致しない（≠不再現）ケースがあります。

---

### 1.4 seed 3本（0/1/2）の統計（mean±std）

同一条件（`val_frac=0.05`, `split_by=study`, `limit_images=8000`, `epochs=1`）で seed を変えて 3本 回した結果です。

| seed | val_logloss_weighted | val_auc_mean |
|---:|---:|---:|
| 0 | 0.0555525 | 0.9915841 |
| 1 | 0.0461215 | 0.9943803 |
| 2 | 0.0600081 | 0.9808989 |
| mean ± std | 0.0538940 ± 0.0070903 | 0.9889545 ± 0.0071150 |

注意:
- これは **KaggleのPrivate LBではなく validation** です（LB と 1:1 対応しません）
- AUC は主に「順位付け（分離）」の指標、logloss は「確率の当たり具合（校正も含む）」を見る指標なので、両者が同じ方向に動かないことがあります（主張の主軸は logloss）。

---

## 2. 再現手順（1コマンド）

### 2.1 前提

- macOS / `mps` を想定（CPU/CUDAでも動く設計）
- 例では **preprocessed SQLite** を使います（DICOM の有無に依らず速い）

決定性（再現性）を高めるための前提（推奨）:
- `--num-workers 0`, `--no-aug`
- `--seed` を固定し、入力（CSV/SQLite）を同一にする
- 期待する再現性チェック: meta.json.split_stats が一致 AND val_* の abs_diff <= 1e-6（厳密一致は backend/kernels に依存し得る）

合格条件（再現性・推奨）:
1. meta.json.split_stats が一致
2. 以下すべてで abs_diff <= 1e-6（推奨）: `val_loss`, `val_loss_plain`, `val_auc_mean`, `val_logloss_weighted`
3. audit_rsna_split.py が n_group_intersection == 0 かつ n_imageid_intersection == 0 を報告
4. `subset_fingerprint_sha256` が一致（任意だが推奨）

検証済み環境（本レポート）:
- Python 3.12.4
- PyTorch 2.6.0 / torchvision 0.21.0
- timm: 未使用（`convnext_tiny` は `torchvision.models.convnext_tiny` を使用）
- device: `mps`
- matmul precision: 本 run では `torch.get_float32_matmul_precision() == "highest"`（既定値）
  - 注: これは PyTorch バージョンやバックエンドにより挙動/有効性が変わり得ます（必要なら `torch.set_float32_matmul_precision("highest")` を明示してください）

再現性の注記:
- `--limit-images` を用いる場合、同一 `--seed` / 同一入力（CSV と preprocessed DB）で subset が安定するよう実装しています。再現性チェックは run1/run2 の `split_stats` 一致で監査できます（`meta.json` に `limit_images` / `seed` / `split_stats` が記録されます）。

前処理の fit 境界（リーク疑いを避けるための明文化）:
- `--windows`（HU window）や `--stack-slices` は **固定の変換**で、train/val を見て統計量を fit しません
- この README の設定では `--input-normalize none` のため、データセット全体の mean/std を学習して使う処理はありません
- preprocessed SQLite は「画像の変換結果」と「Study/Series UID などのメタ情報」を保持するだけで、ラベルや split に依存しません（split はその後の段階で group 単位に行います）

データ例（今回の run で使用）:
- `rsna_root`: `pipeline/Datasets/rsna_preprocessed_gpt52_img384_w3_f32/rsna_meta`
- `preprocessed_root`: `pipeline/Datasets/rsna_preprocessed_gpt52_img384_w3_f32`

### 2.2 実行コマンド（fine-tune short / val_frac=0.05）

実行入口（ファイル）:
- `train_rsna_cnn2d_classifier.py`（このフォルダ直下の薄い wrapper）
  - 実装本体は `src/training/train_rsna_cnn2d_classifier.py`（Typer app）

```bash
cd pipeline

TORCH_DEVICE=mps python train_rsna_cnn2d_classifier.py train \
  --rsna-root "Datasets/rsna_preprocessed_gpt52_img384_w3_f32/rsna_meta" \
  --preprocessed-root "Datasets/rsna_preprocessed_gpt52_img384_w3_f32" \
  --out-dir "results/<YOUR_RUN_DIR>" \
  --limit-images 8000 \
  --val-frac 0.05 \
  --split-by study \
  --seed 0 \
  --epochs 1 \
  --lr 5e-06 \
  --weight-decay 0.0 \
  --image-size 384 \
  --windows '40,80;80,200;600,2800' \
  --preprocess gpt52 \
  --stack-slices 3 \
  --batch-size 6 \
  --num-workers 0 \
  --arch convnext_tiny \
  --pretrained \
  --first-conv-init mean \
  --input-normalize none \
  --no-aug \
  --no-scheduler \
  --loss-any-weight 1.0 \
  --no-use-pos-weight \
  --no-use-sampler \
  --optimize-plain-loss \
  --init-from "results/rsna_convnext25d_final2_full_mps_nosamp_nopw_20260202_222608/best.pt" \
  --log-every-steps 200
```

`init_from` の由来:
- `results/rsna_convnext25d_final2_full_mps_nosamp_nopw_20260202_222608/best.pt`
  - 別 run の holdout validation（`split_by=study`, `val_frac=0.02`）で **`val_loss` 最小**として保存された checkpoint
  - 上記 run は今回の評価 split（`val_frac=0.05`）とは独立で、`init_from` の選択に今回の評価結果は使っていません
  - ハイパラ探索のための run ではなく、fine-tune を安定に始めるための初期値として参照（この README では参照 run を固定しており、CV の結果を見て恣意的に差し替えていません）
  - 今回の CV / holdout の val を見て `init_from` を選び直す（val最適化）ことはしていません
  - 重み付き logloss 最小の checkpoint を使いたい場合は、同フォルダの `best_wlogloss.pt` を選べます

生成物:
- `meta.json`: 実験設定（split_stats含む）
- `log.jsonl`: epochごとの集計指標（最終行に val 指標）
- `best.pt` / `best_auc.pt` / `best_wlogloss.pt` / `last.pt` / `last_state.pt`

監査ポイント: `meta.json`（少なくとも `rsna_root`, `preprocessed_root`, `seed`, `split_stats` を含む）と、`log.jsonl` の最終行（最終 `val_*`）。

### 2.3 実行コマンド（GroupKFold(5) fast CV / epochs=1）

GroupKFold(5) の再現（fast CV）は、`--cv-folds 5` と `--cv-fold-index <0..4>` を使って fold ごとに実行します。

注: CLI 引数名は `python train_rsna_cnn2d_classifier.py train --help` の表示を正としてください（将来変更される可能性があるため）。

注: この README のコマンドは、本リポジトリの同一コード状態で動作確認済みです。第三者が再現する場合は、README とコード一式を同じ状態に揃える（例: 同一コミットを checkout）ことを推奨します。
（上流リポジトリが git 管理下であれば `git rev-parse HEAD` で commit SHA を取得でき、TL;DR のように README に固定できます）

注: `--cv-folds >= 2` のとき split は fold index により決まり、`--val-frac` は分割の決定には使われません（値は `0 <= val_frac < 1` の範囲である必要があります）。

```bash
cd pipeline

BASE="results/<YOUR_CV_DIR>"
for FOLD in 0 1 2 3 4; do
  OUT="$BASE/fold${FOLD}"
  mkdir -p "$OUT"
  TORCH_DEVICE=mps python train_rsna_cnn2d_classifier.py train \
    --rsna-root "Datasets/rsna_preprocessed_gpt52_img384_w3_f32/rsna_meta" \
    --preprocessed-root "Datasets/rsna_preprocessed_gpt52_img384_w3_f32" \
    --out-dir "$OUT" \
    --limit-images 8000 \
    --split-by study \
    --cv-folds 5 \
    --cv-fold-index "$FOLD" \
    --seed 0 \
    --epochs 1 \
    --lr 5e-06 \
    --weight-decay 0.0 \
    --image-size 384 \
    --windows '40,80;80,200;600,2800' \
    --preprocess gpt52 \
    --stack-slices 3 \
    --batch-size 6 \
    --num-workers 0 \
    --arch convnext_tiny \
    --pretrained \
    --first-conv-init mean \
    --input-normalize none \
    --no-aug \
    --no-scheduler \
    --loss-any-weight 1.0 \
    --no-use-pos-weight \
    --no-use-sampler \
    --optimize-plain-loss \
    --init-from "results/rsna_convnext25d_final2_full_mps_nosamp_nopw_20260202_222608/best.pt" \
    --log-every-steps 200
done

# CV結果の集計（mean/std/min/max）
# 前提: --cv-root 配下に fold0/..fold4/ が存在し、各 fold に log.jsonl（最終行=最後に書かれた epoch の集計値）があること（meta.json は任意）。
# ※本ツールはデフォルトで log.jsonl 最終行のみを集計します（best checkpoint 指標の集計は対象外）。
python tools/summarize_cv.py --cv-root "$BASE" --format md
```

---

## 3. 再現性の証拠（run1 vs run2）

### 3.1 何と何を比較しているか

ここでの `auc_max_abs_diff=0.0` は、
**「run1 と run2 の val_auc（各クラス）の差分の最大値」** が 0 という意味です（算出方法の差ではありません）。

### 3.2 比較コマンド（そのまま貼って使える）

```bash
cd pipeline
python -c 'import json; from pathlib import Path
r1=Path("results/rsna_convnext25d_ft_repro_val05_short_20260210_101836_run1"); r2=Path("results/rsna_convnext25d_ft_repro_val05_short_20260210_111323_run2")
m1=json.loads(r1.joinpath("meta.json").read_text()); m2=json.loads(r2.joinpath("meta.json").read_text())

def last(p):
    lines=p.joinpath("log.jsonl").read_text().strip().splitlines();
    return json.loads(lines[-1])

l1=last(r1); l2=last(r2)
print("split_stats_match", m1.get("split_stats")==m2.get("split_stats"))
for k in ["val_loss","val_loss_plain","val_auc_mean","val_logloss_weighted"]:
    a=float(l1.get(k)); b=float(l2.get(k));
    print(k, "abs_diff", abs(a-b))
auc1=l1.get("val_auc",{}); auc2=l2.get("val_auc",{})
maxd=0.0
for c in sorted(set(auc1)|set(auc2)):
    if c in auc1 and c in auc2:
        maxd=max(maxd, abs(float(auc1[c])-float(auc2[c])))
print("auc_max_abs_diff", maxd)
'
```

期待出力:
- `split_stats_match True`
- `abs_diff 0.0`（各 `val_*`）
- `auc_max_abs_diff 0.0`

---

## 4. 評価設計の正当性（リーク対策が“主役”）

RSNA ICH は同一患者・同一検査（Study/Series）の画像が多数含まれます。
**slice 単位でランダム split すると、同一 Study が train/val に跨いでリーク**し、val が過大評価されます。

このパイプラインの方針:
- `split_by=study`（推奨）: `StudyInstanceUID` 単位で Group split
- `split_by=series`: `SeriesInstanceUID` 単位で Group split
- `split_by=slice`: 速度優先の近似（リークが入り得るので主張用には不向き）

### 4.1 リーク検証（証拠コード）

結果（study split）: train/val の group 交差は 0（`n_group_intersection=0`）。

合格条件: n_group_intersection == 0 かつ n_imageid_intersection == 0。

以下は **group の交差が 0 か**を検証する監査スクリプトです。

```bash
cd pipeline
python tools/audit_rsna_split.py \
  --rsna-root "Datasets/rsna_preprocessed_gpt52_img384_w3_f32/rsna_meta" \
  --preprocessed-root "Datasets/rsna_preprocessed_gpt52_img384_w3_f32" \
  --limit-images 8000 \
  --val-frac 0.05 \
  --split-by study \
  --seed 0 \
  --out-json /tmp/audit_study_report.json
```

上の条件（`limit_images=8000, val_frac=0.05, seed=0`）での実測（要点）:
- `n_group_intersection=0`（train/val の Study が交差していない）
- `n_imageid_intersection=0`
- `n_train_groups=5596`, `n_val_groups=294`（Study数）

参考までに、`split_by=slice` は **画像IDの重複は無くても、Study/Series が train/val に跨いで混ざる**（リーク症状）ので、次の監査でそれを数値化できます。

リーク指標（slice split）: n_study_intersection > 0（0以外なら val に Study がリーク）。

```bash
cd pipeline
python tools/audit_rsna_slice_leakage.py \
  --rsna-root "Datasets/rsna_preprocessed_gpt52_img384_w3_f32/rsna_meta" \
  --preprocessed-root "Datasets/rsna_preprocessed_gpt52_img384_w3_f32" \
  --limit-images 8000 \
  --val-frac 0.05 \
  --seed 0 \
  --out-json /tmp/audit_slice_leakage.json
```

同条件での実測（要点）:
- `n_imageid_intersection=0`（slice分割なので当然）
- `n_study_intersection=89`（**val側340 Studyのうち約26%がtrainにも存在**）
- `frac_val_studies_in_train=0.261765`
- 分母の注意: `n_val_studies=340` は「`split_by=slice` の val に出現した Study 数」、`n_val_groups=294` は「`split_by=study` の val group 数」です（別 split 手順なので一致しません）

定義（監査指標の意味）:
- `n_study_intersection = |{StudyUID in val} ∩ {StudyUID in train}|`
- `frac_val_studies_in_train = n_study_intersection / n_val_studies`

※`n_val_groups` / `n_val_studies` は `limit_images` のサンプリング・欠損除外（preprocessed DB のキー一致）・split手法に依存して変動します。上記は各監査スクリプト出力の実測値です。

補足（なぜ 340 vs 294 になり得るか）:
- `split_by=slice` は「画像（slice）を先に train/val に分ける」ため、結果として val 側に入る Study 数が `split_by=study`（Study単位で先に group を分ける）と一致しません。

出力 JSON の見どころ:
- `n_group_intersection == 0`（train/val の Study が交差していない）
- `split_stats`（グループ総数・valグループ数・陽性グループ数など）

### 4.2 追加監査（exact duplicate 対策）

`split_by=study` だけでは「同一Studyリーク」は防げますが、前処理済みテンソルの同一内容が別IDで混在する可能性は別問題です。
そのため本パイプラインでは、preprocessed モード時に **split前にテンソルhashで de-dup（既定ON）** を適用しています（`--dedup-before-split`）。

検証（10 seed）:
- `tools/audit_rsna_dedup_effect.py` で、de-dup後の train/val 間 hash 交差が全seedで 0（`all_zero_after_dedup=true`）
- `tools/hypothesis_audit_rsna_leakage.py` でも Study/Series/Image の交差なしを再確認

結論（このREADMEの監査スコープ内）:
- **「評価を歪める既知のリーク（Study/Series跨ぎ・exact duplicate跨ぎ）は除去済み」** と判断してよい状態です。
- ただし、見た目が非常に近いが byte 一致ではない near-duplicate まで完全否定するには、追加の類似度監査（pHash等）を別途運用します。

---

## 5. 指標の定義と「何を良いとするか」

### 5.1 主指標（Kaggle本番に寄せる）

- `val_logloss_weighted`: RSNA ICH の重み付き multi-label logloss（"any" が重い）
  - ここを下げることが本番目的に近い

根拠（参照; Kaggle公式の評価定義）:
- 参照: [Kaggle competition “RSNA Intracranial Hemorrhage Detection” の Evaluation セクション](https://www.kaggle.com/c/rsna-intracranial-hemorrhage-detection/overview/evaluation)
  - 評価は 6クラス multi-label logloss の加重平均で、`any` の重みが2、他が1です。

定義（このリポジトリの実装の要点）:
- `val_logloss_weighted` は **Kaggleの重み**（`any=2`、他=1）で計算します（指標としての定義）
- 重みは `any=2` / `others=1` を前提に計算します。実装では `RSNA_CLASSES` と重み定義（`RSNA_LOGLOSS_CLASS_WEIGHTS`）の対応に基づき、`_weighted_multilabel_logloss()` で重み付き平均を取る実装になっています（定数と関数名はコード参照のフックとして明示）。
- クラス順は `classes = [epidural, intraparenchymal, intraventricular, subarachnoid, subdural, any]` です（実装の `RSNA_CLASSES`）。
  - ※クラス名は Kaggle の `stage_2_train.csv` の列名（`ID_<class>`）に準拠。
  - ※Kaggleの評価は提出CSVの列名（`<image_id>_<class>`）で解釈されます。
  - ※本repoの submission 生成も `<class>` 名で対応付けて出力する設計です（詳細は `scripts/make_rsna_submission_from_best.zsh` を参照）。
- 注: クラス順は、ラベルと予測が一貫して対応していれば実装上の都合です。重要なのは「クラス別 logloss を計算し、Kaggleの重み（`any=2`, 他=1）で加重平均」することです。
- `--loss-any-weight` は **学習時の損失**にだけ効くパラメータです（この README のコマンドでは `--loss-any-weight 1.0` で学習損失の any を重くしない設定）
- 本実装は **クラス別 logloss を計算し、重み付き平均**します（Kaggleの評価説明と同形）。
  - 実装: `src/training/train_rsna_cnn2d_classifier.py` の `_weighted_multilabel_logloss()`
  - 実行入口: `train_rsna_cnn2d_classifier.py`（wrapper。中で上記モジュールを呼びます）
- $p$ は sigmoid 後の確率で、数値安定化のため $\varepsilon$ で clip します（本repoの既定は $\varepsilon=10^{-7}$）。
- 注: logloss の計算前に `p = clip(sigmoid(logit), eps, 1-eps)` として確率を clip します。
- 比較実験（相対比較）は本repo内の同一定義で一貫して行います。
- Kaggle の内部評価は数値の細部（clipping/aggregation 等）が異なる可能性がありますが、本配布物内の結果はすべて同一実装で計算しているため、run/ablation 間の比較は内部的に一貫しています。

$$
\ell_{i,c} = -\bigl(y_{i,c}\log p_{i,c} + (1-y_{i,c})\log(1-p_{i,c})\bigr),\quad
\mathrm{ll}_c = \frac{1}{N}\sum_{i=1}^{N}\ell_{i,c},\quad
\mathrm{wlogloss} = \frac{\sum_c w_c\,\mathrm{ll}_c}{\sum_c w_c}
$$

補足: 重み $w_c$ がサンプル $i$ に依存せず、欠損が無い前提では、**「サンプル内でクラス方向に重み付き平均→サンプル平均」**の書き方とも同値です。

### 5.2 補助指標（健康診断）

- `val_auc_mean`: クラス別 AUC の平均（主に「順位付け/分離」の良さを見る指標で、確率の校正は直接は反映しません）

### 5.3 `val_loss_plain` と `val_logloss_weighted` の差

- `plain < weighted` のとき、重みの大きいクラス（特に `any`）周りで確率の外れが残っている可能性があります。
- 次の改善の説明としては「**確率校正（温度スケーリング等）で logloss を下げる余地**」が自然です。

---

## 6. 比較軸（採用者が“判断できる”ための軸）

最低限この 3つをセットで提示します:

1) **再現性**
- 同一 seed・同一引数で指標が一致する（この README で実施済み）

2) **評価設計（リーク対策）**
- `split_by=study` の根拠と、交差 0 の監査ログ

3) **比較対象**
- `val_logloss_weighted` を主軸に、条件（split/val_frac/seed/limit_images）を揃えて比較

推奨（次に取り組むと有効）:
- seed を 3本（0/1/2）回して **平均±標準偏差** を出す（この README で実施済み）

---

## 7. Ablation（最小構成の3本）

| ID | split_by | val_frac | preprocess | stack | aug | optimize_plain_loss | init_from | val_logloss_weighted | val_auc_mean | 備考 |
|---:|:--|--:|:--|--:|:--|:--|:--|--:|--:|:--|
| A | slice | 0.05 | gpt52 | 3 | off | on | best.pt | 0.05170 | 0.99479 | slice split（警告例：Studyリークにより指標が過大評価されうる） |
| B | study | 0.02 | gpt52 | 3 | off | on | best.pt | TBD | TBD | val が小さくブレやすい（TBD は意図的に未記載。小valの分散確認後に追記し、チェリーピックを避ける） |
| C | study | 0.05 | gpt52 | 3 | off | on | best.pt | 0.05389 ± 0.00709 | 0.98895 ± 0.00712 | seed 0/1/2 の mean±std |

重要: **A（slice）が C（study）より良く見えるのは「正しい改善」ではなく、同一 Study が train/val に混ざるリーク症状の典型**です（数値の根拠は「4.1 リーク検証」参照）。

※A/B は「比較軸を作るための最小セット」です（TBD を埋めると説得力が跳ねます）。

補足:
- B（`val_frac=0.02`）は「val が小さいとブレやすい」ことを示す枠で、数値は追って追加予定です（未記載は意図的）。

---

## 8. 既存スクリプト（参考）

- scripts/run_rsna_target058_shortest.zsh
  - 2D強モデルを短時間で回す（速度優先のレシピ）
- scripts/make_rsna_submission_from_best.zsh
  - best checkpoint から submission を作る

補足（uncertainty / calibration の出力）:
- `predict_rsna_ich_submission.py` は MC-Dropout による uncertainty 出力にも対応しています。
  - 例: `--mc-dropout-stage-p 0.2 --mc-dropout-head-p 0.2 --mc-samples 30 --out-uncertainty-csv submission_uncertainty.csv`
  - `submission_uncertainty.csv` は `ID` ごとの `ProbStd`（予測確率の標準偏差）を出力します（submission.csvとは別ファイル）。

校正（temperature scaling）+ ECE + coverage–risk 曲線（1枚）+ coverage=80% での改善量（README用）:
- `tools/eval_rsna_uncertainty.py` は、holdout split 上で以下をまとめて出力します。
  - **評価境界**: `--fit-temperature` の標準実行は「同一 holdout val で temperature を fit し、同じ val で ECE/Brier/AURC を算出」です（校正指標は楽観寄りになり得る）
  - **厳密評価の推奨**: val を calib/eval に分割し、calib で温度推定・eval で最終指標算出（または別holdoutをeval専用に利用）
  - 温度スケーリングの温度 $T$（`--fit-temperature` で推定）
  - `ECE(any)`（二値・`any` のみ、bin=15）
  - `Brier(any)` と weighted Brier（確率予測の総合品質; ECEよりビン依存が小さい）
  - uncertainty で誤り検出する `AUROC`（`any` の 0.5閾値分類の誤り=1 をラベルにし、uncertainty=ProbStd を predictor にした AUROC）
  - `NLL`（ここでは Bernoulli の weighted logloss と同義）
  - coverage–risk 曲線（risk=weighted logloss）の PNG
  - Reliability diagram（any, PNG）
  - coverage=0.8 のときの `accuracy(any)` の改善（percentage points）

```bash
cd pipeline

TORCH_DEVICE=mps python tools/eval_rsna_uncertainty.py \
  --rsna-root "Datasets/rsna_preprocessed_gpt52_img384_w3_f32/rsna_meta" \
  --preprocessed-root "Datasets/rsna_preprocessed_gpt52_img384_w3_f32" \
  --ckpt "results/rsna_convnext25d_final2_full_mps_nosamp_nopw_20260202_222608/best.pt" \
  --arch convnext_tiny \
  --image-size 384 \
  --windows '40,80;80,200;600,2800' \
  --preprocess gpt52 \
  --stack-slices 3 \
  --limit-images 8000 \
  --val-frac 0.05 \
  --split-by study \
  --seed 0 \
  --mc-samples 30 \
  --dropout-stage-p 0.2 \
  --dropout-head-p 0.2 \
  --fit-temperature \
  --out-reliability-png results/uncertainty/reliability_any.png \
  --out-curve-png results/uncertainty/coverage_risk.png
```

出力（JSONの例; 数値は環境/ckptに依存）:
- `ece_any`: ECE（`any`）
- `brier_any`: Brier（`any`）
- `auroc_uncertainty_detect_error_any`: uncertaintyで誤り検出のAUROC（`any`）
- `nll_weighted_logloss`: NLL（weighted logloss）
- `accuracy_any_improve_pp`: coverage=0.8 のときの `accuracy(any)` 改善（pp）
- `curve_png`: coverage–risk 曲線の保存先
- `reliability_png`: reliability diagram（any）の保存先

---

## 9. 制限と次の一手（透明性のために明記）

制限:
- 単発 val は偶然に強い/弱いが起こり得る（特に `val_frac` が小さい場合）
- Kaggle Private LB と一致するとは限らない

次の一手（ポートフォリオで有効な順）:
1. seed 3本の平均±標準偏差（済）
2. GroupKFold(5) CV の平均±標準偏差（済）
3. 温度スケーリング等の校正（logloss 改善の筋が良い）
4. ケース別エラー解析（false negative の典型パターン）
5. （実装済み）`meta.json` に subset 指紋（採用 `image_id` のソート済み一覧の sha256）を記録し、第三者が subset 同一性を機械検証できるようにする

この配布物は「監査可能な再現性」と「リークしない評価設計」を優先しており、同一プロトコル下での比較に意味がある数値を報告しています（KaggleのLeaderboardスコアの直接的な代理ではありません）。
