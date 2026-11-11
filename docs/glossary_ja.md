# 用語集

図による整理は [visual_index_ja.md](visual_index_ja.md) を先に参照する。

本リポジトリの日本語文書で用いる用語と、対応する英語表記を示す。新規に書く文章は本表に合わせ、表記の揺れを避ける。

| 用語 | 英語表記・コード上の名前 | 意味 |
|---|---|---|
| 頭蓋内出血 | intracranial hemorrhage (ICH) | 頭蓋骨の内側で起きる出血。本リポジトリでは5つのサブタイプと `any` を分類する。 |
| サブタイプ | subtype | 硬膜外（`epidural`）、脳実質内（`intraparenchymal`）、脳室内（`intraventricular`）、くも膜下（`subarachnoid`）、硬膜下（`subdural`）の5種類である。 |
| any | `any` | 上記サブタイプのいずれかの出血があることを示すラベルである。 |
| 検査 | study | DICOMの `StudyInstanceUID` で識別される検査単位である。 |
| グループ分割 | group split / `split_by=study` | 同一検査が学習と検証にまたがらないよう、検査単位で分割する方法である。 |
| 情報リーク | leakage | 検証側の情報が学習側に混入し、成績を過大評価する事態をいう。 |
| 加重マルチラベル対数損失 | weighted multi-label logloss | Kaggle課題の主指標である。`any` を重く扱う。 |
| AUC | area under the ROC curve | 順位付け性能の指標である。高いほど良い。 |
| ECE | expected calibration error | 予測確率と実際の頻度のずれを測る校正誤差である。低いほど良い。 |
| 温度スケーリング | temperature scaling | ロジットを温度パラメータで割り、確率の校正を改善する手法である。 |
| MC Dropout | Monte Carlo Dropout | 推論時にもDropoutを有効にし、複数回予測して不確実性を推定する手法である。 |
| AURC | area under the risk–coverage curve | 確信度の高い予測に限定したときのリスク推移を要約する指標である。 |
| 初期重み | initial checkpoint | 追加学習の出発点となる学習済みパラメータである。 |
| 医療機器 | medical device | 本リポジトリの実装は医療機器ではない。診断用途には用いない。 |

参照：[evaluation_notes_ja.md](evaluation_notes_ja.md)
