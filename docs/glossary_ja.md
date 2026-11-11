# 用語集

図だけで全体を見たいときは [visual_index_ja.md](visual_index_ja.md) を先に開いてください。

このリポジトリの日本語文書で使う用語と、対応する英語表記です。文書ごとに言い方が揺れないよう、新しく書く文章はこの表に合わせます。

| 用語 | 英語表記・コード上の名前 | 意味 |
|---|---|---|
| 頭蓋内出血 | intracranial hemorrhage (ICH) | 頭蓋骨の内側で起きる出血。このリポジトリでは5つのサブタイプと `any` を分類します。 |
| サブタイプ | subtype | 硬膜外（`epidural`）、脳実質内（`intraparenchymal`）、脳室内（`intraventricular`）、くも膜下（`subarachnoid`）、硬膜下（`subdural`）の5種類です。 |
| `any` | `any` | いずれかのサブタイプの出血があることを示すラベルです。 |
| 検査 | study（`StudyInstanceUID`） | 1回のCT撮影でまとまる画像群です。分割の単位に使います。 |
| シリーズ | series（`SeriesInstanceUID`） | 1つの検査の中で、同じ条件で再構成された一連の画像です。 |
| スライス | slice | CTの1枚の断面画像です。RSNAのラベルはスライスごとに付いています。 |
| 交差検証 | cross-validation（GroupKFold） | データを複数の組（fold）に分け、1組を検証、残りを学習に使う操作を、組を入れ替えながら繰り返す評価方法です。 |
| fold | fold | 交差検証で分けた1区画、またはその区画を検証に使った1回の学習です。 |
| ホールドアウト | holdout | 学習に使わずに残しておく検証用データです。 |
| リーク | leakage | 検証データの情報が学習側に混ざり、成績が実力より良く見える現象です。 |
| 重複除去 | de-duplication | 前処理後の画像テンソルのハッシュが一致する画像を、分割の前に1枚にまとめる処理です。 |
| 初期重み | initial checkpoint（`init_from`） | 追加学習を始めるときに読み込むモデルの重みです。 |
| 追加学習 | fine-tuning | 学習済みの重みから学習を続けることです。 |
| 加重マルチラベル対数損失 | weighted multi-label logloss（`val_logloss_weighted`） | Kaggle RSNA ICHの評価指標です。定義は [評価ノート](evaluation_notes_ja.md) を参照してください。 |
| 校正 | calibration | 予測確率が、実際に陽性となる割合と一致するように確率を調整することです。「較正」とは書きません。 |
| 温度スケーリング | temperature scaling | ロジットを1つの温度で割って確率を校正する方法です。 |
| ECE | expected calibration error | 予測確率を等幅の区間に分け、区間ごとに平均予測確率と陽性の割合の差を求め、区間の件数で重み付けして平均した値です。 |
| Brierスコア | Brier score | 予測確率と正解（0または1）の差の二乗平均です。 |
| 誤り検出AUROC | error-detection AUROC | 不確実性スコアで予測の誤りをどれだけ見分けられるかを表すAUROCです。 |
| 被覆率 | coverage | 確信度の高い順に予測を採用したとき、採用した割合です。 |
| AURC | area under the risk-coverage curve | 被覆率ごとのリスクを、全被覆率にわたって平均した値です。 |
| MC Dropout | Monte Carlo dropout | 推論時にもDropoutを有効にして複数回予測し、ばらつきを不確実性とみなす方法です。 |
| 標本標準偏差 | sample standard deviation（ddof=1） | n−1で割って求める標準偏差です。このリポジトリの「±」はすべてこれです。 |
| 医療機器 | medical device | 診断や治療を目的とする機器やソフトウェアです。このリポジトリは研究用の実装で、医療機器として作ったものではありません。 |
