from __future__ import annotations

import argparse

from .vectors import SDVectors


def main() -> None:
    parser = argparse.ArgumentParser(description="Query an sd2vec vector store.")
    parser.add_argument("directory")
    parser.add_argument("concept")
    parser.add_argument("--topn", type=int, default=10)
    args = parser.parse_args()
    vectors = SDVectors.load(args.directory)
    print(f"vector_size={vectors.vector_size}")
    for concept, score in vectors.most_similar(args.concept, args.topn):
        print(f"{concept}\t{score:.6f}")
