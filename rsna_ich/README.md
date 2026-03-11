# RSNA 頭蓋内出血分類（Kaggle 2019）

**言語:** 日本語 | [英語版](README_en.md)

このフォルダは、**RSNA ICH 分類パイプラインの公開案内ページ**です。採用・監査・外部レビューの最初の接点として、何ができるか / どこを見るべきか / どう試すかを短時間で把握できるようにしています。

## まず分かること

- **何ができるか**: 学習 / 評価 / 推論 / 校正評価 / 不確実性評価 / リーク監査
- **API デモ**: FastAPI で `health` / `model metadata` / `ich inference` の公開向けスキーマを確認可能
- **誰向けか**: 採用担当、医療 AI エンジニア、再現性を重視する研究実装レビュー
- **最短確認**: `python ../scripts/smoke_test.py --use_dummy_data`
- **詳細ドキュメント**:
  - 日本語: `../core/pipeline/README.md`
  - 英語: `../core/pipeline/README_en.md`

## 成果の要点

- weighted multi-label logloss: **0.05346 ± 0.00624**
- mean AUC: **0.98815 ± 0.00311**
- error-detection AUROC (`any`): **0.9424 ± 0.0190**
- ECE (`any`): **0.0231 ± 0.0032**

## この公開物の強み

- `split_by=study` のグループ分割と監査スクリプト
- 分割前テンソルのハッシュによる重複除去でリークを抑制
- `meta.json` / `log.jsonl` / `split_stats` による再現性監査
- `subset_fingerprint_sha256` による対象集合の同一性チェック

## すぐ使うリンク

- 監査用ショートガイド: `./AUDIT_GUIDE.md`
- API デモの実行ファイル: `../core/pipeline/serve_rsna_ich_api.py`
- 引用情報: `../CITATION.cff`
- リリースノート原稿: `../docs/releases/v1.0-interview.md`
- ロードマップ: `../ROADMAP.md`

## 固定スナップショット（ポートフォリオ用）

採用選考でレビューされた「再現評価」は、次のタグに対応します：

✅ `rsna-ich-v1.0-interview`

リポジトリは継続的に開発中です。

この公開フォルダには `Datasets/`, `runs/`, `results/` を同梱していません。実データでの再現コマンドは上記の詳細 README を参照してください。
