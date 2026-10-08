"""Find opposite-pole neighbors by inverting centered sd2vec vectors."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


DEFAULT_PAIRS = {
    "勝利": "敗北",
    "光": "闇",
    "春": "秋",
    "善": "悪",
    "上": "下",
    "生": "死",
    "愛": "憎しみ",
    "美しい": "醜い",
}


def normalize(rows: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    return rows / np.maximum(norms, 1e-12)


def top_neighbors(
    words: list[str],
    scores: np.ndarray,
    excluded: str,
    topn: int,
) -> list[dict[str, object]]:
    results = []
    for index in np.argsort(-scores):
        word = words[index]
        if word == excluded:
            continue
        results.append(
            {
                "rank": len(results) + 1,
                "word": word,
                "score": float(scores[index]),
            }
        )
        if len(results) == topn:
            break
    return results


def rank_of(words: list[str], scores: np.ndarray, target: str) -> int | None:
    if target not in words:
        return None
    order = np.argsort(-scores)
    return int(np.where(order == words.index(target))[0][0] + 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="outputs_chive_10000_phase3")
    parser.add_argument("--output", default="antonym_inversion")
    parser.add_argument("--topn", type=int, default=20)
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    vectors = np.load(input_dir / "vectors.npy").astype(float)
    metadata = json.loads((input_dir / "metadata.json").read_text(encoding="utf-8"))
    words = metadata["concepts"]
    dimensions = [item["id"] for item in metadata["dimensions"]]
    index = {word: i for i, word in enumerate(words)}

    centered = vectors - vectors.mean(axis=0)
    normalized = normalize(centered)
    all_results: list[dict[str, object]] = []
    summaries: dict[str, object] = {}

    for source, expected in DEFAULT_PAIRS.items():
        if source not in index:
            summaries[source] = {"missing_source": True, "expected": expected}
            continue
        query = -centered[index[source]]
        scores = normalized @ (query / max(np.linalg.norm(query), 1e-12))
        neighbors = top_neighbors(words, scores, source, args.topn)
        summaries[source] = {
            "expected_opposite": expected,
            "expected_in_vocabulary": expected in index,
            "expected_rank": rank_of(words, scores, expected),
            "neighbors": neighbors,
        }
        all_results.extend(
            {"source": source, "expected_opposite": expected, **row}
            for row in neighbors
        )

    with (output_dir / "inverted_neighbors.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["source", "expected_opposite", "rank", "word", "score"],
        )
        writer.writeheader()
        writer.writerows(all_results)

    axis_rows = []
    for source, expected in DEFAULT_PAIRS.items():
        if source not in index:
            continue
        query = -centered[index[source]]
        scores = normalized @ (query / max(np.linalg.norm(query), 1e-12))
        top = top_neighbors(words, scores, source, 1)[0]
        axis_rows.append(
            {
                "source": source,
                "expected_opposite": expected,
                "top_neighbor": top["word"],
                "top_score": top["score"],
                "expected_rank": rank_of(words, scores, expected),
            }
        )
    with (output_dir / "inversion_summary.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "source",
                "expected_opposite",
                "top_neighbor",
                "top_score",
                "expected_rank",
            ],
        )
        writer.writeheader()
        writer.writerows(axis_rows)

    summary = {
        "input": str(input_dir),
        "vocabulary_size": len(words),
        "dimensions": len(dimensions),
        "method": "subtract global mean vector, negate source vector, rank by cosine similarity",
        "pairs": summaries,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "# 中心化ベクトル反転による反義語探索",
        "",
        f"- 語彙: {len(words):,}語",
        f"- 尺度: {len(dimensions)}次元",
        "- 手順: 全語彙の平均ベクトルを減算 → 対象語のベクトルを反転 → コサイン類似度で検索",
        "",
        "## 結果",
        "",
        "| 入力語 | 想定反義語 | 1位 | 想定反義語の順位 |",
        "|---|---|---|---:|",
    ]
    for row in axis_rows:
        rank = row["expected_rank"] if row["expected_rank"] is not None else "語彙外"
        lines.append(
            f"| {row['source']} | {row['expected_opposite']} | "
            f"{row['top_neighbor']} ({row['top_score']:.3f}) | {rank} |"
        )
    lines += [
        "",
        "想定反義語が上位に来れば、その語の中心化ベクトルが意味空間上で両極を形成している可能性があります。",
        "ただし、語彙外の語は順位を評価できず、1位語が辞書的な反義語であることも保証されません。",
        "反転は50尺度の組み合わせ全体を反対向きにする操作であり、各尺度の反義語を直接検索する方法ではありません。",
        "",
        "## 詳細出力",
        "",
        "- `inversion_summary.csv`: 各入力語の1位と想定反義語の順位",
        "- `inverted_neighbors.csv`: 入力語ごとの上位近傍",
        "- `summary.json`: 全順位とスコア",
        "",
        "## 尺度の読み方",
        "",
        "反転で近い語が出ても、それは50尺度上の印象プロフィールが逆向きという意味です。",
        "人間の辞書的な反義語関係そのものを検証するには、語彙ごとの正解対リストや人間評定との比較が必要です。",
    ]
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
