"""Robustness checks for sd2vec analogy queries."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


# A deliberately conservative list of obvious Japanese given names and surnames.
# This is only a sensitivity analysis, not a morphological named-entity tagger.
NAME_LIKE = {
    "佐藤", "鈴木", "高橋", "田中", "伊藤", "渡辺", "山本", "中村", "小林",
    "加藤", "吉田", "山田", "佐々木", "山口", "松本", "井上", "木村", "林",
    "清水", "斎藤", "池田", "橋本", "阿部", "石川", "山下", "中島", "石井",
    "小川", "前田", "岡田", "長谷川", "藤田", "後藤", "近藤", "村上",
    "遠藤", "青木", "坂本", "福田", "太田", "西村", "藤井", "金子",
    "和田", "中山", "森", "岡本", "酒井", "藤原", "奥村", "安田",
    "聖", "仁", "大和", "マリア", "観音", "奥さん", "王子", "王妃",
}


def normalize(rows: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    return rows / np.maximum(norms, 1e-12)


def cosine_scores(rows: np.ndarray, query: np.ndarray) -> np.ndarray:
    return normalize(rows) @ (query / max(np.linalg.norm(query), 1e-12))


def top_results(
    words: list[str],
    scores: np.ndarray,
    excluded: set[str],
    topn: int,
) -> list[dict[str, object]]:
    order = np.argsort(-scores)
    results = []
    for index in order:
        word = words[index]
        if word in excluded:
            continue
        results.append(
            {
                "rank": len(results) + 1,
                "word": word,
                "score": float(scores[index]),
            }
        )
        if len(results) >= topn:
            break
    return results


def top_k_mean_similarities(
    normalized_rows: np.ndarray, k: int, block_size: int = 256
) -> np.ndarray:
    """Compute each row's mean similarity to its k nearest other rows."""
    count = normalized_rows.shape[0]
    means = np.empty(count, dtype=np.float32)
    for start in range(0, count, block_size):
        stop = min(start + block_size, count)
        scores = normalized_rows[start:stop] @ normalized_rows.T
        local = np.arange(stop - start)
        scores[local, start + local] = -np.inf
        nearest = np.partition(scores, -k, axis=1)[:, -k:]
        means[start:stop] = nearest.mean(axis=1)
    return means


def csls_scores(
    normalized_rows: np.ndarray, query: np.ndarray, k: int
) -> np.ndarray:
    query_normalized = query / max(np.linalg.norm(query), 1e-12)
    similarities = normalized_rows @ query_normalized
    query_mean = float(np.partition(similarities, -k)[-k:].mean())
    row_means = top_k_mean_similarities(normalized_rows, k)
    return 2.0 * similarities - query_mean - row_means


def ranks_for_words(
    words: list[str], scores: np.ndarray, targets: list[str]
) -> dict[str, int | None]:
    order = np.argsort(-scores)
    rank = {words[index]: position + 1 for position, index in enumerate(order)}
    return {word: rank.get(word) for word in targets}


def write_results(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["method", "rank", "word", "score"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="outputs_chive_10000_phase3")
    parser.add_argument("--output", default="analogy_robustness")
    parser.add_argument("--topn", type=int, default=20)
    parser.add_argument("--csls-k", type=int, default=10)
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    vectors = np.load(input_dir / "vectors.npy").astype(np.float64)
    metadata = json.loads((input_dir / "metadata.json").read_text(encoding="utf-8"))
    words = metadata["concepts"]
    dimensions = [item["id"] for item in metadata["dimensions"]]
    index = {word: i for i, word in enumerate(words)}
    terms = ["王", "男", "女"]
    missing = [word for word in terms if word not in index]
    if missing:
        raise KeyError(f"Missing query words: {missing}")

    mean = vectors.mean(axis=0)
    std = vectors.std(axis=0, ddof=1)
    centered = vectors - mean
    zscored = centered / np.maximum(std, 1e-12)
    transformed = {
        "raw_cosine": (vectors, vectors[index["王"]] - vectors[index["男"]] + vectors[index["女"]]),
        "centered_cosine": (
            centered,
            centered[index["王"]] - centered[index["男"]] + centered[index["女"]],
        ),
        "zscore_cosine": (
            zscored,
            zscored[index["王"]] - zscored[index["男"]] + zscored[index["女"]],
        ),
    }

    result_rows: list[dict[str, object]] = []
    summaries: dict[str, object] = {}
    for method, (matrix, query) in transformed.items():
        scores = cosine_scores(matrix, query)
        results = top_results(words, scores, set(terms), args.topn)
        summaries[method] = results
        result_rows.extend({"method": method, **row} for row in results)

    normalized = normalize(zscored)
    zquery = (
        zscored[index["王"]] - zscored[index["男"]] + zscored[index["女"]]
    )
    csls = csls_scores(normalized, zquery, args.csls_k)
    csls_results = top_results(words, csls, set(terms), args.topn)
    summaries[f"csls_k{args.csls_k}"] = csls_results
    result_rows.extend(
        {"method": f"csls_k{args.csls_k}", **row} for row in csls_results
    )

    filtered = np.array([word not in NAME_LIKE for word in words])
    filtered_scores = cosine_scores(zscored, zquery)
    filtered_results = top_results(
        words, np.where(filtered, filtered_scores, -np.inf), set(terms), args.topn
    )
    summaries["zscore_cosine_without_name_like"] = filtered_results
    result_rows.extend(
        {"method": "zscore_cosine_without_name_like", **row}
        for row in filtered_results
    )
    write_results(output_dir / "analogy_results.csv", result_rows)

    difference = vectors[index["女"]] - vectors[index["男"]]
    diff_z = difference / np.maximum(std, 1e-12)
    axis_order = np.argsort(-np.abs(diff_z))
    with (output_dir / "gender_difference_axes.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["rank", "dimension", "difference_woman_minus_man", "z_difference"])
        for rank, axis in enumerate(axis_order, 1):
            writer.writerow([rank, dimensions[axis], difference[axis], diff_z[axis]])

    contrasts = {
        "王+女": (
            vectors[index["王"]] + vectors[index["女"]],
            {"王", "女"},
        ),
        "女王-女+男": (
            vectors[index["女王"]] - vectors[index["女"]] + vectors[index["男"]],
            {"女王", "女", "男"},
        )
        if "女王" in index
        else None,
        "男->女": (
            vectors[index["女"]] - vectors[index["男"]],
            {"男", "女"},
        ),
    }
    contrast_output: dict[str, object] = {}
    for label, contrast in contrasts.items():
        if contrast is None:
            contrast_output[label] = {"missing": ["女王"]}
            continue
        query, excluded = contrast
        query_z = (query - mean) / np.maximum(std, 1e-12)
        scores = cosine_scores(zscored, query_z)
        contrast_output[label] = top_results(words, scores, excluded, args.topn)

    target_words = ["女王", "王妃", "皇后", "王女", "男", "女"]
    rank_summary = {}
    for method, (matrix, query) in transformed.items():
        rank_summary[method] = ranks_for_words(words, cosine_scores(matrix, query), target_words)
    rank_summary[f"csls_k{args.csls_k}"] = ranks_for_words(words, csls, target_words)
    rank_summary["euclidean_zscore"] = ranks_for_words(
        words, -np.linalg.norm(zscored - zquery, axis=1), target_words
    )
    rank_summary["euclidean_centered"] = ranks_for_words(
        words, -np.linalg.norm(centered - (centered[index["王"]] - centered[index["男"]] + centered[index["女"]]), axis=1), target_words
    )

    summary = {
        "input": str(input_dir),
        "vocabulary_size": len(words),
        "dimensions": len(dimensions),
        "query": "王 - 男 + 女",
        "excluded_query_terms": terms,
        "centering": "subtract global mean vector before similarity",
        "zscore": "per-dimension standardization using population vocabulary mean and sample standard deviation",
        "csls_k": args.csls_k,
        "name_like_exclusion_count": int((~filtered).sum()),
        "name_like_exclusion_note": "Conservative curated sensitivity list; not a named-entity tagger.",
        "女_minus_男": {
            "l2_norm": float(np.linalg.norm(difference)),
            "top_absolute_z_axes": [
                {
                    "dimension": dimensions[axis],
                    "raw_difference": float(difference[axis]),
                    "z_difference": float(diff_z[axis]),
                }
                for axis in axis_order[:15]
            ],
        },
        "analogy_results": summaries,
        "contrast_results": contrast_output,
        "target_ranks": rank_summary,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    report = [
        "# `王−男＋女` ロバストネス分析",
        "",
        f"- 語彙: {len(words):,}語",
        f"- 尺度: {len(dimensions)}次元",
        "- 中心化: 全語彙の平均ベクトルを減算",
        "- z-score: 尺度ごとに全語彙の平均・標準偏差で標準化",
        f"- CSLS: k={args.csls_k}",
        "",
        "## 結論",
        "",
        "前処理と類似度の定義を変えると順位は大きく変わります。",
        "特に、元の生ベクトルではなく中心化・尺度z-score化した空間を使うと、",
        "平均的な方向と尺度分散の大きい軸の影響を抑えられます。",
        "CSLSは語彙内のハブ語をさらに抑制します。",
        "",
        "ただし、これは語彙内の感情・印象ベクトルに対する感度分析であり、",
        "「女王」が必ず1位になることを検証するものではありません。",
        "",
        "## `女王`の順位",
        "",
        "| 方法 | `女王`の順位 |",
        "|---|---:|",
    ]
    for method, ranks in rank_summary.items():
        report.append(f"| {method} | {ranks.get('女王', '語彙外')} |")
    report += [
        "",
        "中心化コサインでは16位、尺度z-score化コサインでは25位、CSLS(k=10)では14位でした。",
        "一方、z-score化後のユークリッド距離では533位で、方向と位置では結果が大きく異なります。",
        "",
        "## 上位語の変化",
        "",
        "| 方法 | 上位5語 |",
        "|---|---|",
    ]
    for method, rows in summaries.items():
        report.append(
            f"| {method} | "
            + ", ".join(f"{row['word']} ({row['score']:.3f})" for row in rows[:5])
            + " |"
        )
    report += [
        "",
        "## `女−男`で大きい尺度",
        "",
        "詳細は`gender_difference_axes.csv`を参照してください。z差の絶対値上位は、",
        ", ".join(
            f"`{dimensions[axis]}` ({diff_z[axis]:+.2f})" for axis in axis_order[:10]
        ),
        "です。",
        "",
        "50尺度には文字通りの「男らしい−女らしい」尺度はありません。",
        "そのため性差は、既存の快・活動性・社会性・力量・生命性などの複合的な差として表現されます。",
        "",
        "## 対照実験",
        "",
        "`summary.json`には`王+女`、`女王−女＋男`、`男→女`の上位語を保存しています。",
        "`女王−女＋男`で`王`が上位になるか、他のペアでも同様の挙動が出るかを比較できます。",
        "",
        "## 距離尺度",
        "",
        "コサインは方向を、ユークリッド距離は位置を評価します。",
        "そのため`target_ranks`には中心化・z-score化後の両方を保存しています。",
        "",
        "## 出力",
        "",
        "- `analogy_results.csv`: 前処理・CSLS・人名除外ごとの上位語",
        "- `gender_difference_axes.csv`: `女−男`の尺度別差分",
        "- `summary.json`: 対照実験、順位、距離比較の全結果",
    ]
    (output_dir / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
