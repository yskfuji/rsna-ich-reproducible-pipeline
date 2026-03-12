# 再現性チェックリスト

このページは、公開版 RSNA ICH リポジトリを外部レビューする際の簡易チェックリストです。

## 1. パッケージの完全性

- README とリリースノートで参照している安定スナップショットのタグを確認する。
- リポジトリに機微な医療データや private な run artifact が含まれていないことを確認する。
- 必要に応じて、`core/pipeline` で `python scripts/smoke_test.py --use_dummy_data` または `python tools/make_manifest.py` を実行し、新しいマニフェストを生成する。

## 2. ドキュメントの整合性

- 入口となるページとして `README.md` または `README_ja.md` を読む。
- タスク説明として `rsna_ich/README_en.md` または `rsna_ich/README.md` を読む。
- リリースノート原稿として `docs/releases/v0.6.0-rsna_ja.md` を確認する。
- logloss、AUC、不確実性、校正に関する主張が、これらのファイル間で矛盾していないことを確認する。

## 3. コード経路の妥当性

- 学習、推論、API デモのエントリポイントが存在することを確認する。
  - `core/pipeline/train_rsna_cnn2d_classifier.py`
  - `core/pipeline/predict_rsna_ich_submission.py`
  - `core/pipeline/serve_rsna_ich_api.py`
- リーク監査スクリプトと不確実性評価スクリプトが公開物に含まれていることを確認する。

## 4. スモークテスト

- `python scripts/smoke_test.py --use_dummy_data` を実行する。
- 実データなしでコマンドが完了することを確認する。
- 出力された summary が、想定どおりの公開ファイルとエントリポイントを指していることを確認する。

## 5. 評価レシピとしての見やすさ

- README に `split_by=study` を使う理由が明示されていることを確認する。
- 分類スコアだけでなく、不確実性や校正の見方も示されていることを確認する。
- FastAPI デモが本番運用の主張ではなく、公開用のスキーマ確認デモとして説明されていることを確認する。

## 6. モデル登録の検証

- リポジトリ直下では `python core/pipeline/tools/verify_registration.py --run-dir <REPRESENTATIVE_RUN_DIR> --model-name rsna-convnext25d --checkpoint best_wlogloss.pt --promotion-rule "val_logloss_weighted<=0.055" --registered-model-name rsna-convnext25d-verify`、`core/pipeline` では `python tools/verify_registration.py ...` を実行する。
- `artifacts/verification/registered_models/.../registration.json` が生成されることを確認する。
- 標準出力の JSON 要約に、想定どおりの `promotion_status` と alias 結果が含まれることを確認する。

## 7. レビュワー目線の合格条件

- 主要指標、リーク対策、最初に動かすコマンドを 3 分以内に把握できる。
- train → evaluate → audit → API demo の流れを、非公開スクリプトに頼らず追える。
- 保護データがなくても、リポジトリの配線と再現性の考え方を確認できる。

## 8. 既知の制約

- このチェックリストが検証するのは、公開物としての再現性導線であり、Kaggle 本番スコアの完全再現そのものではない。
- 完全な指標再現には、別途準備した RSNA データと対応する学習環境が必要になる。