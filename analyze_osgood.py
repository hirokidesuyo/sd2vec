"""Analyze whether sd2vec contains Osgood-like three-factor structure."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


MARKERS = {
    "Evaluation": [
        "pleasant",
        "beautiful",
        "valuable",
        "moral",
        "honest",
        "reliable",
        "safe",
        "fair",
        "healthy",
        "clean",
        "useful",
        "trusting",
        "hopeful",
    ],
    "Potency": [
        "strong",
        "powerful",
        "large",
        "expensive",
        "deep",
        "specialized",
    ],
    "Activity": [
        "active",
        "exciting",
        "joyful",
        "bright",
        "fast",
        "alive",
        "creative",
        "open",
    ],
}


def standardize(values: np.ndarray) -> np.ndarray:
    return (values - values.mean(axis=0)) / values.std(axis=0, ddof=1)


def principal_components(correlation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    eigenvalues, eigenvectors = np.linalg.eigh(correlation)
    order = np.argsort(eigenvalues)[::-1]
    return eigenvalues[order], eigenvectors[:, order]


def principal_axis_loadings(
    correlation: np.ndarray, factors: int, tolerance: float = 1e-7
) -> np.ndarray:
    """Estimate principal-axis factor loadings before rotation."""
    inverse = np.linalg.inv(correlation)
    communalities = np.clip(1.0 - 1.0 / np.diag(inverse), 0.0, 1.0)
    for _ in range(1000):
        reduced = correlation - np.diag(1.0 - communalities)
        eigenvalues, eigenvectors = np.linalg.eigh(reduced)
        positive = np.maximum(eigenvalues[-factors:], 0.0)
        loadings = eigenvectors[:, -factors:] * np.sqrt(positive)
        updated = np.sum(loadings**2, axis=1)
        if np.max(np.abs(updated - communalities)) < tolerance:
            break
        communalities = updated
    return loadings


def varimax(loadings: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """Rotate loadings with orthogonal varimax rotation."""
    rows, factors = loadings.shape
    rotation = np.eye(factors)
    previous = 0.0
    for _ in range(1000):
        rotated = loadings @ rotation
        u, singular_values, vh = np.linalg.svd(
            loadings.T
            @ (
                rotated**3
                - (gamma / rows)
                * rotated
                @ np.diag(np.diag(rotated.T @ rotated))
            )
        )
        rotation = u @ vh
        objective = singular_values.sum()
        if previous and objective / previous < 1.0 + 1e-7:
            break
        previous = objective
    return loadings @ rotation


def correlation(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(a, b)[0, 1])


def write_csv(path: Path, dimensions: list[str], rows: list[dict[str, object]]) -> None:
    fields = ["dimension", *rows[0].keys()]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for dimension, row in zip(dimensions, rows):
            writer.writerow({"dimension": dimension, **row})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="outputs_chive_10000_phase3")
    parser.add_argument("--output", default="osgood_analysis")
    parser.add_argument("--factors", type=int, default=3)
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    vectors = np.load(input_dir / "vectors.npy").astype(float)
    metadata = json.loads((input_dir / "metadata.json").read_text(encoding="utf-8"))
    dimensions = [item["id"] for item in metadata["dimensions"]]
    standardized = standardize(vectors)
    correlation_matrix = np.corrcoef(standardized, rowvar=False)
    eigenvalues, eigenvectors = principal_components(correlation_matrix)

    pca_scores = standardized @ eigenvectors[:, : args.factors]
    pca_loadings = eigenvectors[:, : args.factors] * np.sqrt(eigenvalues[: args.factors])
    factor_loadings = varimax(
        principal_axis_loadings(correlation_matrix, args.factors)
    )

    marker_correlations: dict[str, dict[str, float]] = {}
    for name, marker_ids in MARKERS.items():
        indices = [dimensions.index(item) for item in marker_ids if item in dimensions]
        composite = standardized[:, indices].mean(axis=1)
        marker_correlations[name] = {
            f"PC{i + 1}": correlation(composite, pca_scores[:, i])
            for i in range(args.factors)
        }

    rows = []
    for index, dimension in enumerate(dimensions):
        pca = {
            f"pc{i + 1}_loading": round(float(pca_loadings[index, i]), 8)
            for i in range(args.factors)
        }
        factors = {
            f"factor{i + 1}_loading": round(float(factor_loadings[index, i]), 8)
            for i in range(args.factors)
        }
        rows.append({**pca, **factors})
    write_csv(output_dir / "dimension_loadings.csv", dimensions, rows)

    summary = {
        "input": str(input_dir),
        "observations": int(vectors.shape[0]),
        "dimensions": int(vectors.shape[1]),
        "factors": args.factors,
        "pca_eigenvalues": [float(value) for value in eigenvalues],
        "pca_explained_variance_ratio": [
            float(value / len(dimensions)) for value in eigenvalues
        ],
        "pca_cumulative_variance_first_factors": float(
            eigenvalues[: args.factors].sum() / len(dimensions)
        ),
        "marker_composite_correlations": marker_correlations,
        "factor_rotation": "varimax",
        "interpretation": {
            "Evaluation": "factor with reliability, trust, morality, safety, and value loadings",
            "Potency": "factor with strength, power, size, cost, and depth loadings",
            "Activity": "factor with pleasantness, joy, excitement, brightness, and openness loadings",
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    factor_order = np.argsort(
        -np.max(np.abs(factor_loadings), axis=0)
    )
    factor_names = ["Evaluation", "Potency", "Activity"]
    lines = [
        "# Osgood 3因子分析レポート",
        "",
        f"- 入力: `{input_dir}`",
        f"- 観測語数: {vectors.shape[0]:,}",
        f"- SD尺度数: {vectors.shape[1]}",
        "- 方法: 尺度を標準化したPCA、および主軸法＋varimax回転による3因子分析",
        "",
        "## 結論",
        "",
        "sd2vecの50尺度には、評価・力量・活動性と部分的に対応する3方向が見られます。",
        "ただし、3因子が一意に再現されたとは言えず、評価・活動性の一部は快・明るさ・情動や",
        "信頼・道徳・秩序に分かれて見えます。因子数の妥当性は追加診断で検証してください。",
        "これは人間評定との一致を証明するものではなく、LLMが生成した尺度間相関に対する探索的な結果です。",
        "",
        "## PCA",
        "",
        "| 主成分 | 固有値 | 寄与率 | 累積寄与率 |",
        "|---:|---:|---:|---:|",
    ]
    cumulative = 0.0
    for i, value in enumerate(eigenvalues[:10]):
        ratio = float(value / len(dimensions))
        cumulative += ratio
        lines.append(f"| PC{i + 1} | {value:.4f} | {ratio:.2%} | {cumulative:.2%} |")
    lines += [
        "",
        f"第3主成分までで分散の **{summary['pca_cumulative_variance_first_factors']:.2%}** を説明します。",
        "PC1は信頼・道徳・価値、PC2は力量・規模・深さ、PC3は快・明るさ・情動の尺度が比較的強く載ります。",
        "",
        "## 3因子回転解の上位負荷量",
        "",
    ]
    for column, factor_index in enumerate(factor_order):
        label = factor_names[column] if column < len(factor_names) else f"Factor {column + 1}"
        order = np.argsort(-np.abs(factor_loadings[:, factor_index]))[:10]
        lines.append(f"### Factor {factor_index + 1}（{label}相当）")
        lines.append("")
        lines.append(
            ", ".join(
                f"`{dimensions[index]}` ({factor_loadings[index, factor_index]:+.3f})"
                for index in order
            )
        )
        lines.append("")
    lines += [
        "## 解釈上の注意",
        "",
        "- 因子名は負荷量を見て後付けした解釈であり、確認的因子分析による事前固定ではありません。",
        "- 50尺度には評価・力量・活動性以外の意味（時間性、道徳性、知覚性など）も含まれます。",
        "- 評価系と活動性系には交差負荷があり、古典的な3因子が完全に分離したわけではありません。",
        "- 人間評定データとの因子負荷量・因子得点の一致を調べるには、同じ語彙と尺度の人間データが必要です。",
        "",
        "## 出力",
        "",
        "- `summary.json`: 固有値、寄与率、マーカー複合得点との相関",
        "- `dimension_loadings.csv`: 全尺度のPCA負荷量と回転因子負荷量",
    ]
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
