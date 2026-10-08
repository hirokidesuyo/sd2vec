"""Run d1 SD scoring for a prepared vocabulary manifest.

Each chunk is an independent checkpoint. A completed chunk is skipped when
its quality report exists, so a disconnected Colab session can be resumed.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--dimensions", type=Path, default=Path("dimensions_phase3.json"))
    parser.add_argument("--model", default="LiquidAI/d1-3B")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int)
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    chunks = manifest["chunks"][args.start : args.stop]
    root = args.manifest.parent
    for chunk in chunks:
        output = Path(chunk["output"])
        if not output.is_absolute():
            output = root / output
        if (output / "quality.json").exists():
            print(f"skip chunk={chunk['id']} output={output}")
            continue
        concepts = Path(chunk["concepts"])
        if not concepts.is_absolute():
            concepts = root / concepts
        command = [
            args.python,
            "d1_sd2vec.py",
            "--model",
            args.model,
            "--concepts",
            str(concepts),
            "--dimensions",
            str(args.dimensions),
            "--output",
            str(output),
            "--repeats",
            str(args.repeats),
            "--batch-size",
            str(args.batch_size),
        ]
        print(f"start chunk={chunk['id']} concepts={chunk['count']}", flush=True)
        subprocess.run(command, check=True)
        print(f"done chunk={chunk['id']}", flush=True)


if __name__ == "__main__":
    main()
