"""Diagnostics for the number, interpretation, and stability of SD factors."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np


def standardize(x: np.ndarray) -> np.ndarray:
    return (x - x.mean(0)) / x.std(0, ddof=1)


def eigensystem(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    eigenvalues, eigenvectors = np.linalg.eigh(np.corrcoef(x, rowvar=False))
    order = np.argsort(eigenvalues)[::-1]
    return eigenvalues[order], eigenvectors[:, order]


def varimax(loadings: np.ndarray, maximum: int = 1000) -> np.ndarray:
    rows, factors = loadings.shape
    rotation = np.eye(factors)
    previous = 0.0
    for _ in range(maximum):
        rotated = loadings @ rotation
        target = loadings.T @ (
            rotated**3
            - rotated @ np.diag(np.diag(rotated.T @ rotated)) / rows
        )
        left, singular, right = np.linalg.svd(target)
        rotation = left @ right
        objective = singular.sum()
        if previous and objective / previous < 1.0 + 1e-7:
            break
        previous = objective
    return loadings @ rotation


def promax(loadings: np.ndarray, power: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """Promax oblique rotation from an orthogonal loading matrix."""
    orthogonal = varimax(loadings)
    target = np.sign(orthogonal) * np.abs(orthogonal) ** power
    regression = np.linalg.lstsq(orthogonal, target, rcond=None)[0]
    scale = np.diag(1.0 / np.sqrt(np.diag(regression.T @ regression)))
    transformation = regression @ scale
    pattern = orthogonal @ transformation
    factor_correlation = np.linalg.inv(transformation.T @ transformation)
    factor_correlation /= np.sqrt(
        np.outer(np.diag(factor_correlation), np.diag(factor_correlation))
    )
    return pattern, factor_correlation


def parallel_analysis(
    observations: int, dimensions: int, simulations: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    simulated = np.empty((simulations, dimensions))
    for index in range(simulations):
        random = rng.standard_normal((observations, dimensions))
        simulated[index], _ = eigensystem(random)
    return np.percentile(simulated, 95, axis=0), simulated.mean(axis=0)


def map_test(x: np.ndarray, maximum: int) -> list[float]:
    correlation = np.corrcoef(x, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(correlation)
    order = np.argsort(eigenvalues)[::-1]
    scores = []
    for factors in range(maximum + 1):
        if factors:
            loadings = eigenvectors[:, order[:factors]] * np.sqrt(
                eigenvalues[order[:factors]]
            )
            residual = correlation - loadings @ loadings.T
        else:
            residual = correlation.copy()
        diagonal = np.sqrt(np.maximum(np.diag(residual), 1e-12))
        partial = residual / np.outer(diagonal, diagonal)
        off_diagonal = partial[~np.eye(partial.shape[0], dtype=bool)]
        scores.append(float(np.mean(off_diagonal**2)))
    return scores


def factor_loadings(x: np.ndarray, factors: int) -> np.ndarray:
    eigenvalues, eigenvectors = eigensystem(x)
    return eigenvectors[:, :factors] * np.sqrt(np.maximum(eigenvalues[:factors], 0))


def best_tucker(
    left: np.ndarray, right: np.ndarray
) -> tuple[list[float], tuple[int, ...]]:
    left = left / np.sqrt(np.sum(left**2, axis=0))
    right = right / np.sqrt(np.sum(right**2, axis=0))
    factors = left.shape[1]
    best_score = -np.inf
    best = None
    for permutation in itertools.permutations(range(factors)):
        correlations = np.sum(left * right[:, permutation], axis=0)
        score = np.mean(np.abs(correlations))
        if score > best_score:
            best_score = score
            best = (correlations, permutation)
    assert best is not None
    return [float(value) for value in best[0]], best[1]


def top_loadings(
    dimensions: list[str], loadings: np.ndarray, count: int = 10
) -> list[list[dict[str, float | str]]]:
    result = []
    for factor in range(loadings.shape[1]):
        order = np.argsort(-np.abs(loadings[:, factor]))[:count]
        result.append(
            [
                {"dimension": dimensions[i], "loading": float(loadings[i, factor])}
                for i in order
            ]
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="outputs_chive_10000_phase3")
    parser.add_argument("--output", default="osgood_diagnostics")
    parser.add_argument("--simulations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    vectors = np.load(input_dir / "vectors.npy").astype(float)
    metadata = json.loads((input_dir / "metadata.json").read_text(encoding="utf-8"))
    dimensions = [item["id"] for item in metadata["dimensions"]]
    x = standardize(vectors)
    eigenvalues, _ = eigensystem(x)

    parallel_95, parallel_mean = parallel_analysis(
        vectors.shape[0], vectors.shape[1], args.simulations, args.seed
    )
    retained_parallel = int(np.sum(eigenvalues > parallel_95))
    map_scores = map_test(x, min(15, vectors.shape[1] - 1))
    map_factors = int(np.argmin(map_scores))

    rotations = {}
    for factors in (3, 4, 5):
        loadings = factor_loadings(x, factors)
        pattern, factor_correlation = promax(loadings)
        rotations[str(factors)] = {
            "pattern_top_loadings": top_loadings(dimensions, pattern),
            "factor_correlation": factor_correlation.tolist(),
        }

    midpoint = vectors.shape[0] // 2
    first = x[:midpoint]
    second = x[midpoint:]
    stability = {}
    for factors in (3, 4, 5):
        left = factor_loadings(first, factors)
        right = factor_loadings(second, factors)
        coefficients, permutation = best_tucker(left, right)
        stability[str(factors)] = {
            "tucker_coefficients": coefficients,
            "mean_absolute_tucker": float(np.mean(np.abs(coefficients))),
            "permutation": list(permutation),
        }

    summary = {
        "input": str(input_dir),
        "observations": int(vectors.shape[0]),
        "dimensions": int(vectors.shape[1]),
        "parallel_analysis": {
            "simulations": args.simulations,
            "seed": args.seed,
            "random_95th_percentile_eigenvalues": parallel_95.tolist(),
            "random_mean_eigenvalues": parallel_mean.tolist(),
            "observed_eigenvalues": eigenvalues.tolist(),
            "retained_components": retained_parallel,
        },
        "map_test": {
            "mean_squared_partial_correlation_by_removed_components": map_scores,
            "recommended_components": map_factors,
        },
        "promax_rotations": rotations,
        "split_half_stability": stability,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "# Osgood因子構造の追加診断",
        "",
        f"- 入力: `{input_dir}`",
        f"- 語数: {vectors.shape[0]:,}",
        f"- 尺度数: {vectors.shape[1]}",
        "",
        "## 結論",
        "",
        f"平行分析（{args.simulations}回、95パーセンタイル）では"
        f" **{retained_parallel}因子**が乱数基準を上回りました。",
        f"MAP検定の最小値は **{map_factors}因子**で得られました。",
        "したがって、3因子を先験的に採用するだけでは因子数の根拠として不十分です。",
        "このデータでは、3因子は要約モデルの候補であって、確定した再現結果とは扱いません。",
        "",
        "## スクリープロット相当の固有値",
        "",
        "| 成分 | 観測固有値 | 平行分析95%点 | 観測 > 基準 |",
        "|---:|---:|---:|:---:|",
    ]
    for i, (observed, simulated) in enumerate(
        zip(eigenvalues, parallel_95), 1
    ):
        lines.append(
            f"| {i} | {observed:.4f} | {simulated:.4f} | "
            f"{'yes' if observed > simulated else 'no'} |"
        )
    lines += [
        "",
        "固有値1超えだけで因子数を決めるKaiser基準は過剰抽出になりやすいため、"
        "平行分析とMAPの結果を優先します。",
        "",
        "## 斜交回転（promax）",
        "",
        "Osgoodの因子が相関しうることを考慮し、直交varimaxに加えてpromax回転を確認しました。",
        "詳細な負荷量と因子間相関は`summary.json`にあります。",
        "",
    ]
    for factors, result in rotations.items():
        lines.append(f"### {factors}因子")
        lines.append("")
        for factor, entries in enumerate(result["pattern_top_loadings"], 1):
            top = ", ".join(
                f"`{entry['dimension']}` ({entry['loading']:+.3f})"
                for entry in entries[:8]
            )
            lines.append(f"- Factor {factor}: {top}")
        lines.append(
            f"- 因子間相関: {np.array(result['factor_correlation']).round(3).tolist()}"
        )
        lines.append("")
    lines += [
        "3因子解で典型的な活動性（active, fast, aliveなど）がまとまらず、"
        "快・明るさ・情動と交差する場合、Activityの再現というより尺度設計に依存した別因子と読むべきです。",
        "力量因子にstrong/powerful/largeがまとまっても、simple/deep/expensiveなどが混ざるなら、"
        "純粋なPotencyではなく重み・規模・複雑さの軸の可能性があります。",
        "",
        "## 半分割の安定性",
        "",
        "語彙を前半・後半の5,000語に分け、各因子数のPCA負荷量をTucker一致係数で比較しました。",
        "",
        "| 因子数 | Tucker一致係数（各因子） | 平均絶対値 |",
        "|---:|---|---:|",
    ]
    for factors, result in stability.items():
        lines.append(
            f"| {factors} | "
            + ", ".join(f"{v:.3f}" for v in result["tucker_coefficients"])
            + f" | {result['mean_absolute_tucker']:.3f} |"
        )
    lines += [
        "",
        "一致係数が高いほど、語彙を分割しても同じ方向が再現されます。",
        "ただし今回の分割は語彙ファイルの順序に依存するため、ランダム分割を追加した確認が望まれます。",
        "",
        "## 解釈",
        "",
        "- `Evaluation`は単一因子というより、信頼・道徳・秩序と快・明るさに分かれる可能性があります。",
        "- `Potency`は一部再現する一方、重み・規模・深さ・複雑さが混ざる可能性があります。",
        "- `Activity`はactive/fast/aliveの典型尺度がまとまるかを確認しないと、再現とは言いにくいです。",
        "- 以上は評定ベクトルの構造であり、LLM内部表現や人間評定との同一性を示すものではありません。",
    ]
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
