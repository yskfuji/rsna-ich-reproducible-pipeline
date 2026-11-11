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

- タグの行は、注釈付きタグが指すコミットを示しています。
- 正しいRSNAの `main` は、作者の手元のクローンに `297b5e30ec60837492ca900f4bd234c42701573e` として残っていました。`v0.6.0-rsna` と `rsna-ich-v1.0-interview` は、この履歴に含まれています。
- `7b8699e` は、[isles2022-2d3d-blend-reproducible-pipeline](https://github.com/yskfuji/isles2022-2d3d-blend-reproducible-pipeline) の履歴にも含まれています。上書きによってISLES側の成果は失われていません。
- 修復前の全参照は、作者の手元に `git bundle` で退避しました（SHA-256 `342bd2c3f94b2159…`、`git bundle verify` で検証済み）。

## 2. 見つかった問題と原因

| ID | 問題 | 原因 | 対処 |
|---|---|---|---|
| F-1 | `main` にISLESの内容が表示される | 別リポジトリの履歴が `main` に強制pushされた | ISLES版を `archive/isles-misplaced-20260615` に退避し、`main` をRSNAの正しい履歴に戻す |
| F-2 | `src/datasets/rsna_ich_dataset.py` が公開されていない | `.gitignore` の `Datasets/` は位置を固定していないため、大文字小文字を区別しない環境（`core.ignorecase=true`）では `src/datasets/` にも一致した | `Datasets/` と `datasets/` はどの階層でも除外したまま、ソースのパッケージだけを `!core/pipeline/src/datasets/`（`core/pipeline` 側では `!/src/datasets/`）で例外にする。判定は `--no-index` で行い、CIで検査する |
| F-3 | `reports/latest_eval_report.*` がISLESのDice評価になっている | ISLESの評価レポートが混入した | 削除し、指標の出典を `reports/rsna_cv_summary.json` に一本化する |
| F-4 | 見出しの指標4値に、機械可読な出典がない | foldごとの値が公開物に含まれていない | 出典JSONを追加し、READMEとの一致をCIで検査する |
| F-5 | 依存のバージョンが固定されておらず、CIもない。smoke testが欠落を見逃す | `configs/env.yml` はあるが、バージョンを指定していない。smoke testは作業ツリーのファイル存在だけを確認していた | バージョンを固定した `requirements.txt`、CI、Gitの管理下にあるファイルだけで行うimport検査を追加する |
| F-6 | 不確実性評価のばらつきが母標準偏差（ddof=0）で、ほかの表（標本標準偏差、ddof=1）と定義が混在していた | 不確実性評価の集計だけが母標準偏差で計算されていた（公開値は元の値から再計算して照合した） | 全指標を標本標準偏差で再計算した。例：誤り検出AUROCは `0.9424 ± 0.0190` から `0.9424 ± 0.0200` に変わった |
| F-7 | 5-fold CVの各foldが、初期重みの学習データとほぼ重なる | 初期重みは、前処理済みの学習データ645,793スライスのうち、検査単位で98%にあたる632,808スライスで学習されており、どのfoldの検証データもその大半を含むと見込まれる（実測はしていない） | 「独立した評価」という記述を削除し、重なりを注意書きとして開示した。ImageNet重みからの再実行をROADMAPに追加した |
| F-8 | 温度スケーリングを、評価に使う同じ検証データで当てはめている | 校正用の分割を設けていない | ECEとBrierは楽観側に偏る可能性があると明記した。校正用分割の分離をROADMAPに追加した |
| F-9 | 分割の単位が検査（study）で、患者単位ではない | 前処理済みのメタデータに `PatientID` が含まれていない | 同一患者の別検査が学習と検証にまたがる可能性を明記した |
| F-10 | 重み付きloglossの「`any` の重み2」を、公式の定義として記載していた | 公式の説明は「`any` を重く扱う」とだけ述べており、重み2はコミュニティによる再現である | 出典の区別を `docs/evaluation_notes_ja.md` に明記した |
| F-11 | スクリプトに作者の手元の絶対パスが残っていた。ISLES専用の `tools/deep_research_bundle.py` が、存在しないモジュールに依存していた | 個人環境での実行スクリプトを、そのまま公開していた | スクリプトの位置からの相対パスに置き換え、`deep_research_bundle.py` を削除した。絶対パスの再混入はCIで検査する |

## 3. 合格基準の定義に対する監査

計画時に定義した合格基準のうち、次の2点は定義そのものに誤りがあったため修正しました。

- **AC-5（秘匿情報）**：計画では「DICOMタグ名が0件」としていました。しかし、読み込み処理は `SeriesInstanceUID`、`InstanceNumber`、`RescaleSlope` などの技術的なタグを正当に参照します。この定義では、正しいコードが不合格になります。そこで判定対象を、患者や施設を識別できるタグに絞りました。患者を識別するタグは `PatientName`、`PatientID`、`PatientBirthDate`、`OtherPatientIDs` です。検査や施設を識別するタグは `AccessionNumber`、`InstitutionName`、`ReferringPhysicianName` です。
- **AC-3（混入の除去）**：計画では「`rg -il 'isles|dice'` で当たるのはISLESへの案内文だけ」としていました。しかし、RSNAのコードには、セグメンテーション用の共通部品（`metrics_segmentation.py`、`unet_3d.py` など）が正当に含まれています。そこで判定を「評価レポートと見出しの指標にISLESの値が含まれないこと」に修正しました。

修復後の判定結果は、付録の合否表（[acceptance_2026-11.md](acceptance_2026-11.md)）に記録します。
