# 公開リポジトリ整合性監査（2026-11-11）

この文書は、GitHub上の `main` がRSNA ICHとは別のプロジェクト（ISLES 2022 2D+3D blend）で上書きされていた問題と、その修復の記録です。

## 1. 修復前の状態

2026-11-11 17:30（日本時間）に `git ls-remote` で取得した参照です。

| 参照 | コミット | 内容 |
|---|---|---|
| `refs/heads/main` | `7b8699e3e845d02dedbc60e94aa35f343fb04a68` | ISLES 2D+3D blend「docs: prepare v0.0.0.1 release」 |
| `refs/heads/dev` | `902c8072e63280710cc261a2f7bf652c19c5e576` | RSNA ICH（`main` より20コミット古い） |
| `refs/tags/v0.0.0.0` | `d1febddf639ec5b2e26dc6f379bccdd25f3b7462` | ISLESのタグ（混入） |
| `refs/tags/v0.0.0.1` | `7b8699e3e845d02dedbc60e94aa35f343fb04a68` | ISLESのタグ（混入） |

- 正しいRSNAの `main` は、作者の手元のクローンに `297b5e30ec60837492ca900f4bd234c42701573e` として残っていました。`v0.6.0-rsna` と `rsna-ich-v1.0-interview` は、この履歴に含まれています。
- `7b8699e` は、[isles2022-2d3d-blend-reproducible-pipeline](https://github.com/yskfuji/isles2022-2d3d-blend-reproducible-pipeline) の履歴にも含まれています。上書きによってISLES側の成果は失われていません。
- 修復前の全参照は、作者の手元に `git bundle` で退避しました（SHA-256 `342bd2c3f94b2159…`、`git bundle verify` で検証済み）。

## 2. 見つかった問題と原因

| ID | 問題 | 原因 | 対処 |
|---|---|---|---|
| F-1 | `main` にISLESの内容が表示される | 別リポジトリの履歴が `main` に強制pushされた | ISLES版を `archive/isles-misplaced-20260615` に退避し、`main` をRSNAの正しい履歴に戻す |
| F-2 | `src/datasets/rsna_ich_dataset.py` が公開されていない | `.gitignore` の `Datasets/` は位置を固定していないため、大文字小文字を区別しない環境（`core.ignorecase=true`）では `src/datasets/` にも一致した | `/Datasets/` と `/datasets/` に固定し、`!/src/datasets/` を明示する |
| F-3 | `reports/latest_eval_report.*` がISLESのDice評価になっている | ISLESの評価レポートが混入した | 削除し、指標の出典を `reports/rsna_cv_summary.json` に一本化する |
| F-4 | 見出しの指標4値に、機械可読な出典がない | foldごとの値が公開物に含まれていない | 出典JSONを追加し、READMEとの一致をCIで検査する |
| F-5 | 依存定義とCIがない。smoke testが欠落を見逃す | smoke testが作業ツリーのファイル存在だけを確認していた | `requirements.txt`、CI、Gitの管理下にあるファイルだけで行うimport検査を追加する |

## 3. 合格基準の定義に対する監査

計画時に定義した合格基準のうち、次の2点は定義そのものに誤りがあったため修正しました。

- **AC-5（秘匿情報）**：計画では「DICOMタグ名が0件」としていました。しかし、読み込み処理は `SeriesInstanceUID`、`InstanceNumber`、`RescaleSlope` などの技術的なタグを正当に参照します。この定義では、正しいコードが不合格になります。そこで判定対象を、患者や施設を識別できるタグ（`PatientName`、`PatientID`、`PatientBirthDate`、`OtherPatientIDs`、`AccessionNumber`、`InstitutionName`、`ReferringPhysicianName`）に絞りました。
- **AC-3（混入の除去）**：計画では「`rg -il 'isles|dice'` で当たるのはISLESへの案内文だけ」としていました。しかし、RSNAのコードには、セグメンテーション用の共通部品（`metrics_segmentation.py`、`unet_3d.py` など）が正当に含まれています。そこで判定を「評価レポートと見出しの指標にISLESの値が含まれないこと」に修正しました。

修復後の判定結果は、付録の合否表（[acceptance_2026-11.md](acceptance_2026-11.md)）に記録します。
