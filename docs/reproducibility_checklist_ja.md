# 再現性点検表

本稿は、公開版 RSNA ICH リポジトリを外部レビューする際の点検項目を列挙する。

図による整理を先に確認する場合は [visual_index_ja.md](visual_index_ja.md) を参照する。

## 1. パッケージの完全性

- READMEおよびリリースノートが参照する安定スナップショットのタグを確認する。
- リポジトリに機微な医療データや非公開の実行成果物が含まれていないことを確認する。
- 必要に応じ、`core/pipeline` で `python scripts/smoke_test.py --use_dummy_data` を実行する。
- 必要に応じ、`python tools/make_manifest.py` を実行し、マニフェストを更新する。

## 2. ドキュメントの整合性

- 入口として `README.md` または `README_ja.md` を読む。
- タスク説明として `rsna_ich/README_en.md` または `rsna_ich/README.md` を読む。
- リリースノート原稿として `docs/releases/v0.6.0-rsna_ja.md` を確認する。
- logloss、AUC、不確実性、校正に関する主張が、これらファイル間で矛盾しないことを確認する。

## 3. コード経路の妥当性

- 学習、推論、APIデモのエントリポイントが存在することを確認する。
  - `core/pipeline/train_rsna_cnn2d_classifier.py`
  - `core/pipeline/predict_rsna_ich_submission.py`
  - `core/pipeline/serve_rsna_ich_api.py`
- リーク監査スクリプトおよび不確実性評価スクリプトが公開物に含まれることを確認する。

## 4. スモークテスト

- `python scripts/smoke_test.py --use_dummy_data` を実行する。
- 実データなしでコマンドが完了することを確認する。
- 出力要約が、想定どおりの公開ファイルおよびエントリポイントを指すことを確認する。

## 5. 評価手順の明示性

- READMEに `split_by=study` を用いる理由が明示されていることを確認する。
- 分類スコアに加え、不確実性および校正の見方が示されていることを確認する。
- FastAPIデモが本番運用の主張ではなく、公開用スキーマ確認デモとして説明されていることを確認する。

## 6. モデル登録の検証

- リポジトリ直下では `python core/pipeline/tools/verify_registration.py` を、代表的な実行ディレクトリと昇格条件を引数に付けて実行する。
- `core/pipeline` 配下では `python tools/verify_registration.py` を同等の引数で実行する。
- `artifacts/verification/registered_models/.../registration.json` が生成されることを確認する。
- 標準出力のJSON要約に、想定どおりの `promotion_status` およびalias結果が含まれることを確認する。

## 7. レビュー上の合格条件

- 主要指標、情報リーク対策、最初に実行するコマンドを短時間で把握できる。
- 学習、評価、監査、APIデモの流れを、非公開スクリプトに頼らず追跡できる。
- 保護データがなくても、リポジトリの配線および再現性の考え方を確認できる。

## 8. 既知の制約

- 本点検表が検証するのは公開物としての再現性導線であり、Kaggle本番スコアの完全再現そのものではない。
- 指標の完全再現には、別途用意したRSNAデータおよび対応する学習環境が必要である。
