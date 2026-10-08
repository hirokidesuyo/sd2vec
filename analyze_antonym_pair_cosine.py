"""Compare antonym-pair cosine similarities with random unrelated pairs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


PAIRS = [
    ("程度", "高い", "低い"),
    ("程度", "重い", "軽い"),
    ("時間", "新しい", "古い"),
    ("程度", "大きい", "小さい"),
    ("程度", "強い", "弱い"),
    ("活動", "早い", "遅い"),
    ("温度", "熱い", "冷たい"),
    ("知覚", "明るい", "暗い"),
    ("活動", "静か", "うるさい"),
    ("程度", "硬い", "柔らかい"),
    ("程度", "長い", "短い"),
    ("程度", "広い", "狭い"),
    ("程度", "深い", "浅い"),
    ("程度", "厚い", "薄い"),
    ("数量", "多い", "少ない"),
    ("価格", "高価", "安価"),
    ("難易度", "簡単", "難しい"),
    ("距離", "近い", "遠い"),
    ("評価", "良い", "悪い"),
    ("評価", "美しい", "醜い"),
    ("評価", "正しい", "間違った"),
    ("評価", "安全", "危険"),
    ("評価", "好き", "嫌い"),
    ("評価", "成功", "失敗"),
    ("評価", "勝利", "敗北"),
    ("状態", "生", "死"),
    ("状態", "健康", "病気"),
    ("状態", "自然", "人工的"),
    ("状態", "清潔", "汚れた"),
    ("状態", "安定", "不安定"),
    ("方向", "上", "下"),
    ("方向", "前", "後"),
    ("方向", "内", "外"),
    ("方向", "左", "右"),
    ("方向", "表", "裏"),
    ("方向", "始まり", "終わり"),
    ("方向", "出口", "入口"),
    ("方向", "進む", "戻る"),
    ("活動", "忙しい", "暇"),
    ("活動", "興奮する", "退屈な"),
    ("時間", "現代的な", "古風な"),
    ("時間", "未来的な", "過去の"),
    ("時間", "一時的な", "永続的な"),
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

    available = [
        (category, left_word, right_word)
        for category, left_word, right_word in PAIRS
        if left_word in index and right_word in index
    ]
    missing = [
        {
            "category": category,
            "left": left_word,
            "right": right_word,
            "missing": [
                word for word in (left_word, right_word) if word not in index
            ],
        }
        for category, left_word, right_word in PAIRS
        if left_word not in index or right_word not in index
    ]
    left = np.array([index[left_word] for _, left_word, _ in available], dtype=int)
    right = np.array([index[right_word] for _, _, right_word in available], dtype=int)
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
        "antonym_pairs_requested": [
            {"category": category, "left": left_word, "right": right_word}
            for category, left_word, right_word in PAIRS
        ],
        "antonym_pairs_available": [
            {"category": category, "left": left_word, "right": right_word}
            for category, left_word, right_word in available
        ],
        "missing_pairs": missing,
        "methods": {},
    }
    for method, matrix in matrices.items():
        pair_values = cosine_pairs(matrix, left, right)
        random_values = cosine_pairs(matrix, random_left, random_right)
        method_rows = []
        for (category, a, b), value in zip(available, pair_values):
            stats = statistics(np.array([value]), random_values)
            row = {
                "method": method,
                "category": category,
                "left": a,
                "right": b,
                **stats,
            }
            method_rows.append(row)
            results.append(row)
        aggregate_value = float(pair_values.mean()) if len(pair_values) else None
        aggregate = {}
        if aggregate_value is not None:
            rng_means = np.mean(
                rng.choice(random_values, size=(10_000, len(pair_values))),
                axis=1,
            )
            aggregate = {
                "antonym_mean": aggregate_value,
                "random_mean": float(random_values.mean()),
                "random_std": float(random_values.std(ddof=1)),
                "mean_difference": aggregate_value - float(random_values.mean()),
                "random_mean_95ci": [
                    float(np.percentile(rng_means, 2.5)),
                    float(np.percentile(rng_means, 97.5)),
                ],
                "one_sided_p_random_mean_at_most_antonym": float(
                    np.mean(rng_means <= aggregate_value)
                ),
                "antonym_above_random_fraction": float(
                    np.mean(pair_values > random_values.mean())
                ),
            }
        categories = sorted({category for category, _, _ in available})
        category_summary = {}
        for category in categories:
            indices = [
                i for i, (item_category, _, _) in enumerate(available)
                if item_category == category
            ]
            category_values = pair_values[indices]
            category_random_means = np.mean(
                rng.choice(random_values, size=(10_000, len(indices))), axis=1
            )
            category_summary[category] = {
                "pair_count": len(indices),
                "antonym_mean": float(category_values.mean()),
                "random_mean": float(random_values.mean()),
                "mean_difference": float(category_values.mean() - random_values.mean()),
                "random_mean_95ci": [
                    float(np.percentile(category_random_means, 2.5)),
                    float(np.percentile(category_random_means, 97.5)),
                ],
                "one_sided_p_random_mean_at_most_antonym": float(
                    np.mean(category_random_means <= category_values.mean())
                ),
            }
        summary["methods"][method] = {
            "pairs": method_rows,
            "aggregate": aggregate,
            "by_category": category_summary,
        }

    with (output_dir / "pair_cosines.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        fieldnames = [
            "method", "category", "left", "right", "cosine", "random_mean", "random_std",
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
        f"利用可能な反義語ペアは **{len(available)}組**、カテゴリは"
        f" **{len(set(category for category, _, _ in available))}種類**です。",
        "",
        "| 方法 | ペア数 | 反義語平均 | ランダム平均 | 差 | ランダム平均の95%区間 |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for method, value in summary["methods"].items():
        aggregate = value["aggregate"]
        if aggregate:
            lines.append(
                f"| {method} | {len(available)} | {aggregate['antonym_mean']:.4f} | "
                f"{aggregate['random_mean']:.4f} | {aggregate['mean_difference']:+.4f} | "
                f"{aggregate['random_mean_95ci'][0]:.4f}–{aggregate['random_mean_95ci'][1]:.4f} |"
            )
    lines.append("")
    lines.append(
        "片側モンテカルロp値（反義語平均がランダム平均以下となる確率）は"
        f"`summary.json`の`one_sided_p_random_mean_at_most_antonym`に保存しています。"
        "推定回数が10,000回なので、0.0は厳密なゼロではなくp<0.0001を表します。"
    )
    lines += [
        "",
        "## カテゴリ別（centered）",
        "",
        "| カテゴリ | ペア数 | 反義語平均 | ランダム平均 | 差 |",
        "|---|---:|---:|---:|---:|",
    ]
    centered_categories = summary["methods"]["centered"]["by_category"]
    for category, aggregate in centered_categories.items():
        lines.append(
            f"| {category} | {aggregate['pair_count']} | "
            f"{aggregate['antonym_mean']:.4f} | {aggregate['random_mean']:.4f} | "
            f"{aggregate['mean_difference']:+.4f} |"
        )
    lines += [
        "",
        "## ペア別結果",
        "",
        "| 方法 | カテゴリ | ペア | コサイン | ランダム平均との差(z) | ランダム分位 |",
        "|---|---|---|---:|---:|---:|",
    ]
    for row in results:
        lines.append(
            f"| {row['method']} | {row['category']} | {row['left']}–{row['right']} | {row['cosine']:.4f} | "
            f"{row['z_vs_random']:+.2f} | {row['random_percentile']:.1%} |"
        )
    if missing:
        lines += ["", "## 語彙外のペア", ""]
        for pair in missing:
            lines.append(
                f"- {pair['category']} / {pair['left']}–{pair['right']}: "
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
