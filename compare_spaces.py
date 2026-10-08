"""Compare chiVe 300d distributional vectors with sd2vec SD vectors."""

from __future__ import annotations

import argparse
import csv
import json
import tarfile
from pathlib import Path

import numpy as np


def read_selected(path: Path) -> list[str]:
    with path.open(encoding="utf-8") as f:
        return [row["word"] for row in csv.DictReader(f)]


def load_chive(path: Path, wanted: set[str]) -> tuple[list[str], np.ndarray]:
    found: dict[str, np.ndarray] = {}
    with tarfile.open(path, "r:gz") as archive:
        member = next(item for item in archive.getmembers() if item.name.endswith(".txt"))
        stream = archive.extractfile(member)
        if stream is None:
            raise RuntimeError(f"cannot read {member.name}")
        header = stream.readline().decode("utf-8").split()
        if len(header) != 2 or int(header[1]) != 300:
            raise ValueError(f"unexpected chiVe header: {header}")
        for raw in stream:
            parts = raw.decode("utf-8").rstrip().split()
            if parts and parts[0] in wanted:
                found[parts[0]] = np.asarray(parts[1:], dtype=np.float32)
            if len(found) == len(wanted):
                break
    words = [word for word in wanted if word in found]
    return words, np.stack([found[word] for word in words])


def normalize(matrix: np.ndarray) -> np.ndarray:
    return matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-8)


def nearest(words: list[str], matrix: np.ndarray, topn: int) -> dict[str, list[dict[str, float | str]]]:
    normalized = normalize(matrix)
    result = {}
    for i, word in enumerate(words):
        scores = normalized @ normalized[i]
        order = np.argsort(-scores)
        result[word] = [
            {"word": words[j], "cosine": round(float(scores[j]), 6)}
            for j in order
            if j != i
        ][:topn]
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chive", type=Path, required=True)
    parser.add_argument("--selection", type=Path, default=Path("data/chive_vocab_10000.csv"))
    parser.add_argument("--sd", type=Path, default=Path("outputs_chive_10000_phase3"))
    parser.add_argument("--output", type=Path, default=Path("comparison_10000"))
    parser.add_argument("--sample-size", type=int, default=2000)
    parser.add_argument("--topn", type=int, default=10)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    selected = read_selected(args.selection)
    sd_meta = json.loads((args.sd / "metadata.json").read_text(encoding="utf-8"))
    sd_words = sd_meta["concepts"]
    sd = np.load(args.sd / "vectors.npy")
    sd_index = {word: i for i, word in enumerate(sd_words)}
    wanted = set(selected) & set(sd_words)
    chive_words, chive = load_chive(args.chive, wanted)
    common = [word for word in selected if word in sd_index and word in set(chive_words)]
    chive_index = {word: i for i, word in enumerate(chive_words)}
    sd_matrix = np.stack([sd[sd_index[word]] for word in common])
    chive_matrix = np.stack([chive[chive_index[word]] for word in common])

    rng = np.random.default_rng(42)
    sample_idx = rng.choice(len(common), size=min(args.sample_size, len(common)), replace=False)
    sd_norm = normalize(sd_matrix[sample_idx])
    chive_norm = normalize(chive_matrix[sample_idx])
    upper = np.triu_indices(len(sample_idx), k=1)
    sd_sim = (sd_norm @ sd_norm.T)[upper]
    chive_sim = (chive_norm @ chive_norm.T)[upper]
    correlation = float(np.corrcoef(sd_sim, chive_sim)[0, 1])

    with (args.output / "space_comparison.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "common_words": len(common),
                "sample_size": len(sample_idx),
                "sd_dimension": int(sd_matrix.shape[1]),
                "chive_dimension": int(chive_matrix.shape[1]),
                "similarity_matrix_pearson": correlation,
                "note": "Different spaces are compared by within-space similarities, not raw vector equality.",
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    sd_neighbors = nearest(common, sd_matrix, args.topn)
    chive_neighbors = nearest(common, chive_matrix, args.topn)
    with (args.output / "nearest_comparison.jsonl").open("w", encoding="utf-8") as f:
        for word in common:
            f.write(json.dumps({"word": word, "sd2vec": sd_neighbors[word], "chiVe": chive_neighbors[word]}, ensure_ascii=False) + "\n")

    dimensions = sd_meta["dimensions"]
    with (args.output / "dimension_extremes.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["dimension", "side", "rank", "word", "score"])
        for col, dimension in enumerate(dimensions):
            order = np.argsort(sd_matrix[:, col])
            for rank, i in enumerate(order[:args.topn], 1):
                writer.writerow([dimension["id"], "right", rank, common[i], round(float(sd_matrix[i, col]), 6)])
            for rank, i in enumerate(order[::-1][:args.topn], 1):
                writer.writerow([dimension["id"], "left", rank, common[i], round(float(sd_matrix[i, col]), 6)])
    print(json.dumps({"common_words": len(common), "correlation": correlation}, ensure_ascii=False))


if __name__ == "__main__":
    main()
