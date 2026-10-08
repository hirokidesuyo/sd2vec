"""Prepare chunked concept files for a large chiVe vocabulary run."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("selection", type=Path)
    parser.add_argument("--output", type=Path, default=Path("large_vocab"))
    parser.add_argument("--chunk-size", type=int, default=10_000)
    args = parser.parse_args()
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be positive")

    words: list[str] = []
    with args.selection.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            word = row["word"].strip()
            if word:
                words.append(word)
    if not words:
        raise ValueError("selection file contains no words")

    concepts_dir = args.output / "concepts"
    concepts_dir.mkdir(parents=True, exist_ok=True)
    chunks = []
    for start in range(0, len(words), args.chunk_size):
        chunk_words = words[start : start + args.chunk_size]
        chunk_id = start // args.chunk_size
        path = concepts_dir / f"chunk_{chunk_id:04d}.txt"
        path.write_text("\n".join(chunk_words) + "\n", encoding="utf-8")
        chunks.append(
            {
                "id": chunk_id,
                "start": start,
                "end": start + len(chunk_words),
                "count": len(chunk_words),
                "concepts": (Path("concepts") / path.name).as_posix(),
                "output": (Path("vectors") / f"chunk_{chunk_id:04d}").as_posix(),
            }
        )
    manifest = {
        "source": str(args.selection),
        "concept_count": len(words),
        "chunk_size": args.chunk_size,
        "chunk_count": len(chunks),
        "chunks": chunks,
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"concepts={len(words)} chunks={len(chunks)} "
        f"chunk_size={args.chunk_size} output={args.output}"
    )


if __name__ == "__main__":
    main()
