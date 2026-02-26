# rsna-ich-reproducible-pipeline

このディレクトリは **RSNA Intracranial Hemorrhage (Kaggle 2019)** の公開・監査用パッケージです。

リポジトリ名: `rsna-ich-reproducible-pipeline`

## Stable Portfolio Version（固定スナップショット）

リポジトリは継続的に開発中ですので、ポートフォリオ用のレビューを次のタグに対応させています：

✅ rsna-ich-v1.0-interview

## (はじめに)コミットメッセージに関する規約

レビューしやすさを担保するため、今後のコミットは Conventional Commits 形式（`type: summary`）で揃えます：

- fix: leakage check in group split
- feat: add calibration evaluation
- refactor: manifest validation logic
- docs: evaluation protocol clarification

## イントロダクション

- 監査マップ: `./AUDIT_MAP.md`
- RSNA監査ガイド: `./rsna_ich/AUDIT_GUIDE.md`
- 実験README（JP/EN）:
  - `./core/pipeline/README.md`
  - `./core/pipeline/README_en.md`

## コンポーネントダイレクト

- このリポジトリに含まれるもの: コード、設定、監査用ドキュメント
- 含まれないもの: `Datasets/`, `runs/`, `results/`, `logs/`

データは監査者側で準備し、READMEの再現コマンドに従って検証します。
