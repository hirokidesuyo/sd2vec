# sd2vec

意味微分法（Semantic Differential）を使った、解釈可能なword2vec風ベクトルです。

各概念を30個の形容詞対で評定し、`vectors.npy`を概念×尺度の行列として保存します。
通常の埋め込みと違い、各次元に「快い–不快な」「強い–弱い」などの意味が対応します。

## 使い方

```python
from sd2vec import SDVectors

model = SDVectors.load("outputs_d1")
print(model.vector_size)                 # 30
print(model.get_vector("猫"))
print(model.most_similar("猫", topn=5))
```

CLIでも検索できます。

```bash
pip install -e .
sd2vec-neighbors outputs_d1 猫 --topn 5
```

## 収録データ

`outputs_d1`はLiquid AIの`d1-3B`を使った実行例です。

- 200概念 × 30尺度
- 2反復
- d1の`score`出力（0〜6）をSD値（-3〜3）へ変換
- `vectors.npy`: 推奨する機械可読ベクトル
- `vectors.csv`: Excel等で確認するためのUTF-8 CSV
- `metadata.json`: 概念・尺度・モデル情報
- `raw_scores.jsonl`: 再計算可能な生データ
- `nearest_neighbors.csv`: コサイン類似度による最近傍

## 再生成

ColabまたはGPU環境で依存関係を入れ、次を実行します。

```bash
python d1_sd2vec.py \
  --model LiquidAI/d1-3B \
  --concepts concepts_phase2.txt \
  --dimensions dimensions_phase2.json \
  --limit 200 --repeats 2 --output outputs_d1
python analyze_sd2vec.py --input outputs_d1 --top-k 10
```

`raw_scores.jsonl`は逐次追記されるため、中断後は同じコマンドで再開できます。

## chiVeとの比較

chiVeの300次元分布意味空間と、sd2vecの50次元SD空間を同じ語彙上で比較できます。
生のベクトルを直接比較せず、それぞれの空間内のコサイン類似度行列を比較します。

```bash
python compare_spaces.py \
  --chive data/chive/chive-1.3-mc90.tar.gz \
  --selection data/chive_vocab_10000.csv \
  --sd outputs_chive_10000_phase3 \
  --output comparison_10000
```

出力は、空間間の類似度相関、語ごとの最近傍比較、50尺度の上位・下位語です。

## chiVe語彙からの大規模選定

chiVe v1.3の語彙は頻度順に並んでいます。まず`mc90`（最小頻度90、全300次元）
を取得し、上位50,000語を走査して不要な機能語・数値・記号を除外し、上位10,000語を
選びます。選定結果にはchiVe順位を保存します。

```bash
python select_chive_vocab.py data/chive/chive-1.3-mc90.tar.gz \
  --scan-limit 50000 --limit 10000 \
  --output data/chive_vocab_10000.csv
```

## 注意

これは人間評定の正解データではなく、d1-3Bが生成した印象ベクトルです。
大規模利用では尺度の妥当性、文化差、多義語、モデルのバイアスを確認してください。

軽量LLMに意味微分法(SD法)の固定尺度を評定させ、解釈可能な概念ベクトルを作る最小実装です。

## 実行

```bash
python run_sd2vec.py --limit 50 --repeats 2 --output outputs
```

Colab CLIでは、依存関係を入れて同じスクリプトを実行します。

```bash
colab install -s sd2vec-check torch transformers accelerate sentencepiece numpy
colab exec -s sd2vec-check -f run_sd2vec.py --timeout 3600
```

`outputs/raw_scores.jsonl`は逐次追記されるため、切断後も同じコマンドで再開できます。完成物は `vectors.npy`、`metadata.json`、`quality.json`です。ベクトルの各次元は`dimensions.json`の尺度に対応し、正数は左側の形容詞を表します。

生成後にWindowsのExcelでも読める確認用CSVと最近傍レポートを作成できます。

```bash
python analyze_sd2vec.py --input outputs_v2 --top-k 5
```

`vectors.csv`と`nearest_neighbors.csv`はUTF-8 BOM付きで保存されるため、日本語の文字化けを避けやすくなっています。

## Phase 2

Phase 2では新しいQwen3-4Bを使い、30尺度・200概念で実行します。

```bash
python run_sd2vec.py --model Qwen/Qwen3-4B \
  --concepts concepts_phase2.txt --dimensions dimensions_phase2.json \
  --limit 200 --repeats 2 --output outputs_phase2
python analyze_sd2vec.py --input outputs_phase2 --top-k 10
```
