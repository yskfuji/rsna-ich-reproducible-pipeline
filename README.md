# rsna-ich-reproducible-pipeline

## Stable Portfolio Version

The reproducible evaluation reviewed during recruitment corresponds to:

✅ rsna-ich-v1.0-interview

Active development continues on the repository.

## Commit Message Convention (Going Forward)

To keep the repository easy to review, new commits follow a Conventional Commits style:

- fix: leakage check in group split
- feat: add calibration evaluation
- refactor: manifest validation logic
- docs: evaluation protocol clarification

このディレクトリは **RSNA Intracranial Hemorrhage (Kaggle 2019)** の公開・監査用パッケージです。

推奨リポジトリ名: `rsna-ich-reproducible-pipeline`

## 入口

- 監査マップ: `./AUDIT_MAP.md`
- RSNA監査ガイド: `./rsna_ich/AUDIT_GUIDE.md`
- 実験README（JP/EN）:
  - `./core/pipeline/README.md`
  - `./core/pipeline/README_en.md`

## 同梱方針

- 同梱: コード、設定、監査用ドキュメント
- 非同梱: `Datasets/`, `runs/`, `results/`, `logs/`

データは監査者側で準備し、READMEの再現コマンドに従って検証します。
