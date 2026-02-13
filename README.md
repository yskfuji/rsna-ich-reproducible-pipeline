# RSNA ICH Reproducible Audit Package

このディレクトリは **RSNA Intracranial Hemorrhage (Kaggle 2019)** の公開・監査用パッケージです。

## 入口

- 監査マップ: `./AUDIT_MAP.md`
- RSNA監査ガイド: `./rsna_ich/AUDIT_GUIDE.md`
- 実験README（JP/EN）:
  - `./core/ToReBrain-pipeline/README.md`
  - `./core/ToReBrain-pipeline/README_en.md`

## 同梱方針

- 同梱: コード、設定、監査用ドキュメント
- 非同梱: `Datasets/`, `runs/`, `results/`, `logs/`

データは監査者側で準備し、READMEの再現コマンドに従って検証します。
