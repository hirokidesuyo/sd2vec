"""Compare antonym-pair cosine similarities with random unrelated pairs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


PAIRS = [
    ("勝利", "敗北"),
    ("光", "闇"),
    ("春", "秋"),
    ("上", "下"),
    ("生", "死"),
    ("愛", "憎しみ"),
    ("美しい", "醜い"),
    ("善", "悪"),
]


def normalize(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def cosine_pairs(matrix: np.ndarray, left: np.ndarray, right: np.ndarray) -> np.ndarray:
    normalized = normalize(matrix)
    return np.sum(normalized[left] * normalized[right], axis=1)


def random_pairs(
    size: int, count: int, forbidden: set[tuple[int, int]], rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    left = rng.integers(0, size, count)
    right = rng.integers(0, size, count)
    invalid = (left == right) | np.array(
        [(int(a), int(b)) in forbidden or (int(b), int(a)) in forbidden for a, b in zip(left, right)]
    )
    while np.any(invalid):
        amount = int(invalid.sum())
        left[invalid] = rng.integers(0, size, amount)
        right[invalid] = rng.integers(0, size, amount)
        invalid = (left == right) | np.array(
            [
                (int(a), int(b)) in forbidden or (int(b), int(a)) in forbidden
                for a, b in zip(left, right)
            ]
        )
    return left, right


def statistics(values: np.ndarray, null: np.ndarray) -> dict[str, float]:
    mean = float(null.mean())
    std = float(null.std(ddof=1))
    return {
        "cosine": float(values[0]),
        "random_mean": mean,
        "random_std": std,
        "z_vs_random": float((values[0] - mean) / max(std, 1e-12)),
        "random_percentile": float((null < values[0]).mean()),
        "random_two_sided_p": float(
            (np.abs(null - mean) >= abs(values[0] - mean)).mean()
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="outputs_chive_10000_phase3")
    parser.add_argument("--output", default="antonym_pair_cosine")
    parser.add_argument("--random-pairs", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    vectors = np.load(input_dir / "vectors.npy").astype(float)
    metadata = json.loads((input_dir / "metadata.json").read_text(encoding="utf-8"))
    words = metadata["concepts"]
    index = {word: i for i, word in enumerate(words)}
    dimensions = len(metadata["dimensions"])

    available = [(a, b) for a, b in PAIRS if a in index and b in index]
    missing = [
        {"left": a, "right": b, "missing": [word for word in (a, b) if word not in index]}
        for a, b in PAIRS
        if a not in index or b not in index
    ]
    left = np.array([index[a] for a, _ in available], dtype=int)
    right = np.array([index[b] for _, b in available], dtype=int)
    forbidden = {(int(a), int(b)) for a, b in zip(left, right)}
    rng = np.random.default_rng(args.seed)
    random_left, random_right = random_pairs(
        len(words), args.random_pairs, forbidden, rng
    )

    mean = vectors.mean(axis=0)
    std = vectors.std(axis=0, ddof=1)
    matrices = {
        "raw": vectors,
        "centered": vectors - mean,
        "zscore": (vectors - mean) / np.maximum(std, 1e-12),
    }

    results = []
    summary: dict[str, object] = {
        "input": str(input_dir),
        "vocabulary_size": len(words),
        "dimensions": dimensions,
        "random_pairs": args.random_pairs,
        "seed": args.seed,
        "antonym_pairs_requested": [{"left": a, "right": b} for a, b in PAIRS],
        "antonym_pairs_available": [{"left": a, "right": b} for a, b in available],
        "missing_pairs": missing,
        "methods": {},
    }
    for method, matrix in matrices.items():
        pair_values = cosine_pairs(matrix, left, right)
        random_values = cosine_pairs(matrix, random_left, random_right)
        method_rows = []
        for (a, b), value in zip(available, pair_values):
            stats = statistics(np.array([value]), random_values)
            row = {"method": method, "left": a, "right": b, **stats}
            method_rows.append(row)
            results.append(row)
        aggregate_value = float(pair_values.mean()) if len(pair_values) else None
        aggregate = {}
        if aggregate_value is not None:
            aggregate = {
                "antonym_mean": aggregate_value,
                "random_mean": float(random_values.mean()),
                "random_std": float(random_values.std(ddof=1)),
                "mean_difference": aggregate_value - float(random_values.mean()),
                "z_vs_random_mean": float(
                    (aggregate_value - random_values.mean())
                    / max(random_values.std(ddof=1) / np.sqrt(len(pair_values)), 1e-12)
                ),
                "antonym_above_random_fraction": float(
                    np.mean(pair_values > random_values.mean())
                ),
            }
        summary["methods"][method] = {
            "pairs": method_rows,
            "aggregate": aggregate,
        }

    with (output_dir / "pair_cosines.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        fieldnames = [
            "method", "left", "right", "cosine", "random_mean", "random_std",
            "z_vs_random", "random_percentile", "random_two_sided_p",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "# 反義語ペア間コサイン比較",
        "",
        f"- 語彙: {len(words):,}語",
        f"- 尺度: {dimensions}次元",
        f"- 無関係ペアのモンテカルロ数: {args.random_pairs:,}",
        "- 比較: raw / 全語彙平均を引いたcentered / 尺度z-score",
        "",
        "## 結論",
        "",
        "反義語ペア間のコサインがランダムな語ペアより高ければ、",
        "反義語が同じ文脈・印象軸上に配置されている可能性があります。",
        "これは反転ベクトルの近傍順位とは別の検証です。",
        "",
        "| 方法 | 利用可能ペア数 | 反義語平均 | ランダム平均 | 差 |",
        "|---|---:|---:|---:|---:|",
    ]
    for method, value in summary["methods"].items():
        aggregate = value["aggregate"]
        if aggregate:
            lines.append(
                f"| {method} | {len(available)} | {aggregate['antonym_mean']:.4f} | "
                f"{aggregate['random_mean']:.4f} | {aggregate['mean_difference']:+.4f} |"
            )
    lines += [
        "",
        "## ペア別結果",
        "",
        "| 方法 | ペア | コサイン | ランダム平均との差(z) | ランダム分位 |",
        "|---|---|---:|---:|---:|",
    ]
    for row in results:
        lines.append(
            f"| {row['method']} | {row['left']}–{row['right']} | {row['cosine']:.4f} | "
            f"{row['z_vs_random']:+.2f} | {row['random_percentile']:.1%} |"
        )
    if missing:
        lines += ["", "## 語彙外のペア", ""]
        for pair in missing:
            lines.append(
                f"- {pair['left']}–{pair['right']}: "
                f"{', '.join(pair['missing'])} が語彙外"
            )
    lines += [
        "",
        "## 注意",
        "",
        "反義語は文脈を共有するため、word2vec系の分布空間でもコサインが高くなり得ます。",
        "したがって、コサインが高いことは「同じ意味」や「反対方向の軸」を直接意味しません。",
        "反転近傍、尺度差、語彙外、ランダム対照を合わせて解釈してください。",
        "",
        "詳細は`pair_cosines.csv`と`summary.json`を参照してください。",
    ]
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
