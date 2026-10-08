# sd2vec

**sd2vec** は、意味微分法（Semantic Differential; SD法）を使って作る、解釈可能な概念ベクトルです。
通常の word2vec が文章中の共起から分布意味を学習するのに対し、sd2vec は各概念を固定した形容詞対で評定します。

```text
概念: 猫

快い ─────────●── 不快な
安全な ───────●── 危険な
自然な ─────────● 不自然な
```

各ベクトルの次元に意味が割り当てられているため、単なる類似検索だけでなく、
「どの尺度で似ているのか」「どの概念が快い・危険・現代的なのか」を確認できます。

## 特徴

- d1-3Bなどのdecision modelで大量の概念を評定
- 各次元が固定したSD尺度に対応
- `numpy`形式とUTF-8 CSV形式で保存
- `most_similar()`によるword2vec風の最近傍検索
- `A + B - C`形式のベクトル演算に対応可能な配列形式
- chiVeなどの通常の日本語word2vecと同じ語彙上で比較可能
- JSONLキャッシュにより、長時間処理を中断後に再開可能

## 重要な位置づけ

sd2vecは、人間の心理評定の「正解」を直接収録したデータではありません。
現在の収録データは、Liquid AIの`d1-3B`が固定した尺度に対して行った評定です。
したがって、モデルの知識、プロンプト、文化的背景、学習データ由来のバイアスを含みます。

人間評定と一致することを保証するものではありません。研究・製品利用では、対象分野の人間評定で校正し、
尺度の妥当性、語義、多義語、文化差を検証してください。

## クイックスタート

### インストール

ベクトルの読み込みと検索だけなら、CPU環境で動作します。

```bash
git clone https://github.com/hirokidesuyo/sd2vec.git
cd sd2vec
python -m pip install -e .
```

推論まで実行する場合は、GPU環境で追加依存関係を入れます。

```bash
python -m pip install -e ".[inference]"
```

### Python API

```python
from sd2vec import SDVectors

model = SDVectors.load("outputs_chive_10000_phase3")

print(model.vector_size)            # 50
print(model.index_to_key[:3])       # 語彙の先頭
print(model.get_vector("猫"))       # numpy.ndarray(shape=(50,))
print(model.most_similar("猫", topn=5))
```

`SDVectors`は、主要部分でgensimの`KeyedVectors`に似たインターフェースを提供します。

| 属性・メソッド | 説明 |
|---|---|
| `index_to_key` | 語彙を並べたリスト |
| `key_to_index` | 単語から行番号への辞書 |
| `vector_size` | ベクトル次元数 |
| `vectors` | 概念×尺度の`numpy.ndarray` |
| `get_vector(word)` | 1語のベクトルを取得 |
| `most_similar(word, topn=10)` | コサイン類似度で最近傍語を取得 |
| `save_csv(path)` | ベクトルをUTF-8 BOM付きCSVで保存 |

### CLI

```bash
sd2vec-neighbors outputs_chive_10000_phase3 猫 --topn 10
```

出力例:

```text
vector_size=50
鳥    0.841299
犬    0.822558
子ども 0.806674
```

## Webアプリ: 軸で意味を比較する

ブラウザ上で2語を入力すると、word2vecの「近い・遠い」だけでなく、
次の観点で説明できます。

- **共通している軸**: 2語のSD評定が近い尺度
- **反対方向の軸**: 一方が左極、もう一方が右極にある尺度
- **差が大きい軸**: 語感・ニュアンスの違いを作る尺度
- 全50尺度の値、差分、コサイン、ユークリッド距離

依存関係を増やさず、Python標準ライブラリと既存の`numpy`だけで動作します。

```bash
python webapp.py
```

または、パッケージを編集可能インストールした後:

```bash
sd2vec-web
```

起動するとローカルでは`http://127.0.0.1:8765/`が開きます。既定では全インターフェースで
待ち受けるため、Tailscaleで接続された別端末からは、このPCのTailscale IPを確認して
`http://<Tailscale IP>:8765/`を開いてください。

```bash
# Windows
tailscale ip -4

# Tailscale IPを明示して待受する場合
python webapp.py --host 100.x.y.z --port 8765 --no-browser
```

例えばTailscale IPが`100.64.12.34`なら、
`http://100.64.12.34:8765/`にアクセスします。Windows Defender Firewallで
TCP 8765番ポートの受信がブロックされる場合は、プライベートネットワークまたは
Tailscaleインターフェースからの受信を許可してください。

このWebアプリには認証機能がありません。Tailscaleのアクセス制御されたネットワーク内だけで
使い、インターネットへポート転送しないでください。ローカル限定に戻す場合は
`python webapp.py --host 127.0.0.1`を指定します。

例えば次を試せます。

- `勝利` / `敗北`
- `春` / `秋`
- `高い` / `低い`
- `倹約` / `ケチ`
- `痩せている` / `ガリガリ`

別の成果物を使う場合:

```bash
python webapp.py --input outputs_d1 --port 8765
```

このアプリの比較は、語の辞書的な同義・反義を自動判定するものではありません。
入力された2語の評定プロフィールを比較し、「どの尺度が共通し、どの尺度が異なるか」を
表示するための探索ツールです。`outputs_chive_10000_phase3`では、収録語だけが比較できます。

## 収録データ

### d1-3B・200概念・30尺度

`outputs_d1/`には動作確認用の小規模データを収録しています。

- 200概念
- 30尺度
- 2反復
- ベクトル形状: `(200, 30)`

### d1-3B・chiVe語彙10,000語・30尺度

`outputs_chive_10000/`には、chiVe v1.3 mc90から選定した10,000語のデータを収録しています。

- 10,000語
- 30尺度
- ベクトル形状: `(10000, 30)`
- `vectors.npy`、`vectors.csv`、最近傍レポートを含む

### d1-3B・chiVe語彙10,000語・50尺度

現在の主な実験データは`outputs_chive_10000_phase3/`です。

- 10,000語
- 50尺度
- ベクトル形状: `(10000, 50)`
- 1反復
- 10,000行の`raw_scores.jsonl`
- d1の`score`出力を0〜6からSD値-3〜3へ変換

## 出力ファイル

各出力ディレクトリには、用途に応じて次のファイルが含まれます。

| ファイル | 内容 |
|---|---|
| `vectors.npy` | 推奨する機械可読形式。行が概念、列が尺度 |
| `vectors.csv` | Excel等で確認できるUTF-8 BOM付きCSV |
| `metadata.json` | 使用モデル、概念一覧、尺度一覧、変換情報 |
| `raw_scores.jsonl` | 概念ごとの生評定。再開・再計算に使用 |
| `repeat_std.npy` | 反復実行時の尺度別標準偏差 |
| `quality.json` | 件数、形状、値域などの品質情報 |
| `analysis_summary.json` | 最近傍分析の設定と要約 |
| `nearest_neighbors.csv` | 概念ごとのコサイン最近傍 |

正の値は各尺度の`left`、負の値は`right`側を表します。尺度の定義は必ず
`metadata.json`または対応する`dimensions_*.json`で確認してください。

## ベクトル演算

ベクトルは`numpy`配列なので、word2vec風の足し算・引き算ができます。

```python
import numpy as np
from sd2vec import SDVectors

model = SDVectors.load("outputs_chive_10000_phase3")

query = (
    model.get_vector("男性")
    + model.get_vector("女性")
    - model.get_vector("男")
)

norms = np.linalg.norm(model.vectors, axis=1)
scores = model.vectors @ query / (norms * np.linalg.norm(query) + 1e-8)
excluded = {"男性", "女性", "男"}
results = [
    (model.index_to_key[i], float(scores[i]))
    for i in np.argsort(-scores)
    if model.index_to_key[i] not in excluded
][:10]
print(results)
```

ただし、sd2vecの軸は評価・印象尺度なので、word2vecで知られている類推が常に成立するとは限りません。
足し算・引き算の結果は、尺度ごとの構成差として解釈してください。

## 推論データの再生成

### d1-3Bで小規模実行

```bash
python d1_sd2vec.py \
  --model LiquidAI/d1-3B \
  --concepts concepts_phase2.txt \
  --dimensions dimensions_phase2.json \
  --limit 200 \
  --repeats 2 \
  --output outputs_d1

python analyze_sd2vec.py --input outputs_d1 --top-k 10
```

### d1-3Bで10,000語・50尺度を実行

```bash
python d1_sd2vec.py \
  --model LiquidAI/d1-3B \
  --concepts concepts_chive_10000.txt \
  --dimensions dimensions_phase3.json \
  --limit 10000 \
  --repeats 1 \
  --output outputs_chive_10000_phase3

python analyze_sd2vec.py \
  --input outputs_chive_10000_phase3 \
  --top-k 10
```

`raw_scores.jsonl`は1概念ごとに追記されます。同じ出力ディレクトリを指定して再実行すると、
すでに完了した概念をスキップして途中から再開できます。

### Colab CLIで実行

Google Colab CLIを利用する場合の例です。Colab CLI自体はLinux/macOS対応で、
WindowsではWSLなどを使用してください。

```bash
colab new -s sd2vec-check --gpu L4
colab install -s sd2vec-check \
  "transformers>=5.14" torch torchvision pillow numpy
colab upload -s sd2vec-check d1_sd2vec.py /content/d1_sd2vec.py
colab upload -s sd2vec-check dimensions_phase3.json /content/dimensions_phase3.json
colab upload -s sd2vec-check concepts_chive_10000.txt /content/concepts_chive_10000.txt
colab exec -s sd2vec-check -f d1_sd2vec.py --timeout 14400
```

大規模実行では、Colabの切断に備えてJSONLを定期的に回収してください。
使用しないセッションは停止して、不要なCompute Units消費を避けてください。

## chiVeから語彙を選ぶ

chiVeは日本語のword2vec系ベクトルで、公式データでは全て300次元です。
v1.3 mc90は最小頻度90の語彙で、全体は約41万語です。語彙ファイルは頻度順に並んでいるため、
上位から走査して候補を選べます。

このリポジトリでは、まず上位50,000語を読み、URL・数値・記号・一部の機能語を除外し、
10,000語を採用しています。選定理由とchiVe順位は`data/chive_vocab_10000.csv`に保存しています。

```bash
python select_chive_vocab.py \
  data/chive/chive-1.3-mc90.tar.gz \
  --scan-limit 50000 \
  --limit 10000 \
  --output data/chive_vocab_10000.csv
```

chiVeアーカイブは数百MB〜数GBあるため、元データはGitへコミットしません。
`.gitignore`で`data/chive/*.tar.gz`を除外しています。

## chiVeとsd2vecを比較する

chiVeの300次元ベクトルとsd2vecの50次元ベクトルは、次元数が異なるため直接比較しません。
同じ語彙集合について、それぞれの空間内でコサイン類似度を計算し、類似度行列や最近傍語を比較します。

```bash
python compare_spaces.py \
  --chive data/chive/chive-1.3-mc90.tar.gz \
  --selection data/chive_vocab_10000.csv \
  --sd outputs_chive_10000_phase3 \
  --output comparison_10000 \
  --sample-size 2000 \
  --topn 10
```

出力:

- `space_comparison.json`: 空間内類似度行列の相関
- `nearest_comparison.jsonl`: 語ごとのsd2vec/chiVe最近傍比較
- `dimension_extremes.csv`: 各SD尺度の上位・下位語

既存の10,000語比較では、sd2vecとchiVeの類似度行列のPearson相関は約0.17でした。
これは、chiVeが主に共起・文脈的な意味を、sd2vecが評価・印象を捉えるという設計差と整合します。
この値は品質の優劣を単独で示すものではありません。

## Osgoodの3因子を検証する

SD法で古典的に知られる評価（Evaluation）・力量（Potency）・活動性（Activity）の3因子が、
sd2vecの全語彙に現れるかを、50尺度の相関行列から探索的に検証できます。

```bash
python analyze_osgood.py \
  --input outputs_chive_10000_phase3 \
  --output osgood_analysis
```

この分析は、尺度を標準化したPCAと、主軸法＋varimax回転による3因子分析を行います。
結果は次のファイルに保存されます。

- `osgood_analysis/report.md`: 人間が読むための要約レポート
- `osgood_analysis/summary.json`: 固有値、寄与率、マーカー尺度との相関
- `osgood_analysis/dimension_loadings.csv`: 50尺度ごとのPCA・因子負荷量

10,000語×50尺度の実行結果では、第1〜3主成分が分散の54.27%を説明しました。
第1主成分は信頼・道徳・価値、第2主成分は強さ・力・大きさ・深さ、
第3主成分は快・楽しさ・興奮・明るさに近い負荷を示しました。
ただし第4主成分も6.58%を説明し、追加診断の平行分析は10因子、MAP検定は14因子を示しました。
したがって、3因子は要約モデルの候補であり、Osgoodの3因子がそのまま再現されたとは結論しません。

この結果は、LLMの隠れ状態を直接解析したものではなく、d1-3Bが固定尺度に対して出力した評定値の構造です。
「LLMの頭の中」に人間と同じ構造があると結論するには、人間評定データとの負荷量比較や、
別モデル・別プロンプト・別語彙での再現性検証が必要です。

因子数、斜交回転、半分割の安定性を追加で確認するには次を実行します。

```bash
python analyze_osgood_diagnostics.py \
  --input outputs_chive_10000_phase3 \
  --output osgood_diagnostics
```

`osgood_diagnostics/report.md`には、平行分析、MAP検定、3〜5因子のpromax回転、
および語彙半分割によるTucker一致係数を収録しています。今回のデータでは因子負荷量は半分割に
対して安定でしたが、因子の意味は古典的な評価・力量・活動性よりも、
信頼・秩序、力量・規模・深さ、快・明るさ・情動、自然・生命性などの多因子構造として読む方が適切です。

## 類推結果のロバストネスを調べる

`王−男＋女`のような類推は、平均方向、尺度ごとの分散、語彙内のハブ語に影響されます。
次の分析では、通常のコサインに加えて、全語彙平均を引いた中心化コサイン、
尺度z-score化、CSLS(k=10)、保守的な人名候補除外、ユークリッド距離を比較します。

```bash
python analyze_analogy_robustness.py \
  --input outputs_chive_10000_phase3 \
  --output analogy_robustness \
  --topn 20 \
  --csls-k 10
```

主な出力は次のとおりです。

- `analogy_robustness/report.md`: 順位と解釈の要約
- `analogy_robustness/analogy_results.csv`: 前処理・CSLS・人名候補除外ごとの上位語
- `analogy_robustness/gender_difference_axes.csv`: `女−男`の尺度別差分
- `analogy_robustness/summary.json`: 対照実験、順位、距離比較の全結果

10,000語×50尺度データでは、`女王`は通常コサインで150位でしたが、
中心化コサインで16位、尺度z-score化コサインで25位、CSLS(k=10)で14位まで上昇しました。
一方、z-score化後のユークリッド距離では533位でした。つまり、この類推の評価は
「方向」を見るか「位置」を見るか、またハブ補正を行うかで変わります。

`女−男`のz差が大きかった尺度は`strong`(-1.50)、`calm`(+1.08)、
`creative`(+1.00)、`warm`(+1.00)、`large`(-0.90)、`powerful`(-0.81)などでした。
50尺度に「男らしい−女らしい」を直接表す軸はないため、性差は既存尺度の組合せとして現れます。
人名除外は完全な固有表現認識ではなく、明らかな姓・人名候補を使った感度分析です。

## 中心化ベクトルを反転して反義語を探す

ある語の中心化ベクトルに`-1`を掛け、その方向に近い語を検索すると、
SD尺度上で反対側の印象プロフィールを持つ語を調べられます。

```bash
python analyze_antonym_inversion.py \
  --input outputs_chive_10000_phase3 \
  --output antonym_inversion \
  --topn 20
```

10,000語×50尺度での例:

| 入力語 | 1位 | 想定反義語の順位 |
|---|---|---:|
| 勝利 | 最低 | 敗北は語彙外 |
| 光 | 手間 | 闇は293位 |
| 春 | 制約 | 秋は9,541位 |
| 上 | 出勤 | 下は5,087位 |
| 生 | `km` | 死は2,509位 |

「勝利」では最低・不足・陰性・低下・負など、負方向の印象語が上位に集まりました。
一方、「光→闇」「春→秋」のような辞書的反義語が常に上位になるわけではありません。
これは、50尺度に光/闇や春/秋を直接表す軸がなく、反転が語の辞書的関係ではなく
印象プロフィール全体の反対方向を検索する操作だからです。

詳細は次のファイルに保存されます。

- `antonym_inversion/report.md`
- `antonym_inversion/inversion_summary.csv`
- `antonym_inversion/inverted_neighbors.csv`
- `antonym_inversion/summary.json`

## 反義語ペア同士のコサインを比較する

反転近傍の順位とは別に、反義語ペアそのもののコサイン類似度を測定し、
同じ語彙から作った無関係なランダム語ペアより高いかを比較できます。

```bash
python analyze_antonym_pair_cosine.py \
  --input outputs_chive_10000_phase3 \
  --output antonym_pair_cosine \
  --random-pairs 100000 \
  --seed 42
```

語彙を確認して反義語ペアを追加した結果、31組（12カテゴリ）を利用できました。
語彙内に両方が存在する必要があるため、敗北・憎しみ・醜い・間違ったなどを含む
一部のペアは今回のデータでは評価対象外です。

| 空間 | 反義語ペア平均 | ランダムペア平均 | 差 |
|---|---:|---:|---:|
| raw | -0.0068 | 0.3986 | -0.4054 |
| 中心化 | -0.3045 | 0.0036 | -0.3081 |
| 尺度z-score | -0.3617 | 0.0048 | -0.3665 |

カテゴリ別に見ると、中心化後の「程度」8組は平均−0.4767、
「評価」4組は−0.5942、「状態」3組は−0.4026でした。一方、
「方向」7組は+0.1165で、前–後・始まり–終わりなどは反義語でも同じ文脈上の
近さが強く出ています。

全31組の中心化平均差は−0.3081で、10,000回のモンテカルロ比較では
反義語平均がランダム平均以下となる確率は`p<0.0001`でした（推定値0.0は厳密なゼロではありません）。
ただし、これは同頻度・同品詞にマッチした対照ではなく、カテゴリ間の組数も不均衡です。
したがって「反義語全般の普遍的性質」ではなく、今回の語彙・尺度セットでの探索的な効果と解釈します。

今回の50尺度では、「反義語は全般に高コサイン」という仮説は支持されず、
平均的には反義語の方がランダムペアより低コサインでした。特に評価・程度・知覚・状態の
対立は、SDプロフィール上で反対方向として現れています。一方、方向や時間の一部は、
反対語であっても共有文脈のため高コサインになり得ます。
この分析はランダムペアを頻度・品詞でマッチしていないため、次の改善では同頻度・同品詞の
対照ペアを使う必要があります。

結果は`antonym_pair_cosine/pair_cosines.csv`、`summary.json`、`report.md`に保存されます。

## 尺度の追加・変更

尺度はJSON配列で定義します。

```json
{
  "id": "pleasant",
  "left": "快い",
  "right": "不快な"
}
```

尺度を変更した場合、既存ベクトルとの次元対応が変わるため、別の出力ディレクトリを使用してください。
尺度数を増やす場合は、まず1,000語程度で実行し、尺度間相関を確認してから10,000語へ拡張するのが安全です。
相関が高すぎる尺度を大量に含めると、類似度計算で特定の意味領域が過大評価されます。

## 既知の制約

- d1はチャットモデルではなく、文章を生成しないdecision modelです。
- d1の同一入力は決定論的なため、単純な反復の標準偏差は0になることがあります。
- 不確実性を詳しく扱う場合は、d1の確率分布を保存する拡張が必要です。
- 多義語は一つのベクトルに複数の語義が混ざります。
- 固有名詞、機能語、短い形態素はSD評定に向かない場合があります。
- chiVeの語彙順位は、sd2vecの品質や人間の使用頻度を直接保証しません。
- 30次元・50次元のsd2vecを300次元chiVeへ無理に射影しても、意味軸が保存されるとは限りません。

## 開発

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
```

テストはベクトルの形状、最近傍検索、基本APIを確認します。

## ライセンスと第三者データ

このリポジトリのコードはMIT Licenseです。

chiVeのベクトル・語彙データはchiVe側のライセンスに従います。chiVe v1.3は公式READMEで
Apache License 2.0として案内されています。利用時は公式リポジトリと同梱ライセンスを確認してください。

d1-3Bのモデル重み・コード・利用条件はLiquid AIおよびHugging Faceのモデルカードに従います。
モデルを再配布・商用利用する場合は、必ず最新の公式ライセンスを確認してください。

## リンク

- GitHub: https://github.com/hirokidesuyo/sd2vec
- chiVe: https://github.com/WorksApplications/chiVe
- d1-3B: https://huggingface.co/LiquidAI/d1-3B
