"""Export sd2vec vectors as human-readable CSV and nearest-neighbor reports."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("outputs_v2"))
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    metadata = json.loads((args.input / "metadata.json").read_text(encoding="utf-8"))
    vectors = np.load(args.input / "vectors.npy")
    concepts = metadata["concepts"]
    dimensions = [d["id"] for d in metadata["dimensions"]]
    if vectors.shape != (len(concepts), len(dimensions)):
        raise ValueError(f"shape mismatch: vectors={vectors.shape}")

    with (args.input / "vectors.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["concept", *dimensions])
        writer.writerows([[concept, *map(float, vector)] for concept, vector in zip(concepts, vectors)])

    norms = np.linalg.norm(vectors, axis=1)
    similarity = vectors @ vectors.T / (norms[:, None] * norms[None, :] + 1e-8)
    with (args.input / "nearest_neighbors.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["concept", "neighbor", "cosine_similarity", "rank"])
        for i, concept in enumerate(concepts):
            candidates = np.argsort(-similarity[i])
            rank = 0
            for j in candidates:
                if i == j:
                    continue
                rank += 1
                writer.writerow([concept, concepts[j], round(float(similarity[i, j]), 6), rank])
                if rank >= args.top_k:
                    break

    quality = json.loads((args.input / "quality.json").read_text(encoding="utf-8"))
    summary = {
        **quality,
        "csv_encoding": "utf-8-sig",
        "nearest_neighbor_metric": "cosine",
        "top_k": args.top_k,
        "artifacts": ["vectors.csv", "nearest_neighbors.csv"],
    }
    (args.input / "analysis_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
