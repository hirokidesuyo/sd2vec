from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


class SDVectors:
    """A small word2vec-like interface for SD vectors."""

    def __init__(self, concepts: list[str], dimensions: list[str], vectors: np.ndarray):
        if vectors.shape != (len(concepts), len(dimensions)):
            raise ValueError(f"shape mismatch: {vectors.shape}")
        self.index_to_key = concepts
        self.key_to_index = {key: i for i, key in enumerate(concepts)}
        self.dimensions = dimensions
        self.vectors = np.asarray(vectors, dtype=np.float32)
        self._norms = np.linalg.norm(self.vectors, axis=1)

    @property
    def vector_size(self) -> int:
        return self.vectors.shape[1]

    def get_vector(self, concept: str) -> np.ndarray:
        return self.vectors[self.key_to_index[concept]]

    def most_similar(self, concept: str, topn: int = 10) -> list[tuple[str, float]]:
        if concept not in self.key_to_index:
            raise KeyError(concept)
        i = self.key_to_index[concept]
        similarities = self.vectors @ self.vectors[i] / (self._norms * self._norms[i] + 1e-8)
        order = np.argsort(-similarities)
        return [
            (self.index_to_key[j], float(similarities[j]))
            for j in order
            if j != i
        ][:topn]

    @classmethod
    def load(cls, directory: str | Path) -> "SDVectors":
        directory = Path(directory)
        metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        vectors = np.load(directory / "vectors.npy")
        dimensions = [item["id"] for item in metadata["dimensions"]]
        return cls(metadata["concepts"], dimensions, vectors)

    def save_csv(self, path: str | Path) -> None:
        with Path(path).open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["concept", *self.dimensions])
            writer.writerows(
                [[key, *map(float, self.vectors[i])] for i, key in enumerate(self.index_to_key)]
            )
