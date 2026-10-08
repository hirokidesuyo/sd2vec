"""Build SD vectors with LiquidAI/d1-3B decision model.

d1 returns typed score decisions instead of generating JSON. Each concept is
one state and all SD dimensions are asked in one forward pass.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModel


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="LiquidAI/d1-3B")
    p.add_argument("--concepts", type=Path, default=Path("concepts_chive_10000.txt"))
    p.add_argument("--dimensions", type=Path, default=Path("dimensions_phase2.json"))
    p.add_argument("--output", type=Path, default=Path("outputs_chive_10000"))
    p.add_argument("--limit", type=int, default=10_000)
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=16)
    return p.parse_known_args()[0]


def load(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def question(dimension: dict[str, str]) -> dict[str, Any]:
    levels = [
        f"-3（非常に{dimension['right']}）",
        f"-2（かなり{dimension['right']}）",
        f"-1（やや{dimension['right']}）",
        "0（中立）",
        f"1（やや{dimension['left']}）",
        f"2（かなり{dimension['left']}）",
        f"3（非常に{dimension['left']}）",
    ]
    return {
        "type": "score",
        "instructions": (
            f"この概念は「{dimension['left']}」と「{dimension['right']}」のどちらの印象に近いですか。"
            "一般的な日本語話者の印象として評定してください。"
        ),
        "criteria": levels,
    }


def score_from_answer(answer: dict[str, Any]) -> float:
    value = answer.get("score")
    if value is None:
        raise ValueError(f"d1のscoreがありません: {answer}")
    # d1 score is the expected 0-based/1-based criterion position depending
    # on implementation; prefer legend when present, otherwise map index.
    numeric = float(value)
    # d1's score is the expected zero-based criterion index (0..6).
    return numeric - 3.0


def main() -> None:
    cfg = args()
    cfg.output.mkdir(parents=True, exist_ok=True)
    dimensions = load(cfg.dimensions)
    dimension_ids = [d["id"] for d in dimensions]
    concepts = [
        line.strip()
        for line in cfg.concepts.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ][: cfg.limit]
    cache_path = cfg.output / "raw_scores.jsonl"
    done: dict[tuple[str, int], dict[str, float]] = {}
    if cache_path.exists():
        for line in cache_path.read_text(encoding="utf-8").splitlines():
            if line:
                row = json.loads(line)
                done[(row["concept"], row["repeat"])] = row["scores"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = AutoModel.from_pretrained(cfg.model, trust_remote_code=True, dtype=dtype).to(device)
    model.eval()
    print(f"model={cfg.model} device={device} concepts={len(concepts)} dimensions={len(dimensions)}")

    with cache_path.open("a", encoding="utf-8") as out:
        for repeat in range(cfg.repeats):
            remaining = [c for c in concepts if (c, repeat) not in done]
            for start in range(0, len(remaining), cfg.batch_size):
                batch = remaining[start : start + cfg.batch_size]
                requests = [
                    (f"概念: {concept}", {d["id"]: question(d) for d in dimensions})
                    for concept in batch
                ]
                started = time.perf_counter()
                answers = model.system_one_batch(requests)
                elapsed = time.perf_counter() - started
                if len(answers) != len(batch):
                    raise RuntimeError(f"batch結果数が不一致: {len(answers)} != {len(batch)}")
                for concept, response in zip(batch, answers):
                    answer_map = response.get("answers", {})
                    scores = {
                        dim_id: score_from_answer(answer_map[dim_id])
                        for dim_id in dimension_ids
                    }
                    row = {"concept": concept, "repeat": repeat, "scores": scores}
                    out.write(json.dumps(row, ensure_ascii=False) + "\n")
                    out.flush()
                    done[(concept, repeat)] = scores
                print(
                    f"completed repeat={repeat + 1}/{cfg.repeats} "
                    f"batch={start // cfg.batch_size + 1} elapsed={elapsed:.3f}s"
                )

    matrix = np.array(
        [[done[(c, r)][d] for d in dimension_ids] for c in concepts for r in range(cfg.repeats)],
        dtype=np.float32,
    ).reshape(len(concepts), cfg.repeats, len(dimensions))
    means = matrix.mean(axis=1)
    stds = matrix.std(axis=1)
    np.save(cfg.output / "vectors.npy", means)
    np.save(cfg.output / "repeat_std.npy", stds)
    metadata = {
        "model": cfg.model,
        "concepts": concepts,
        "dimensions": dimensions,
        "repeats": cfg.repeats,
        "scale": [-3, 3],
        "method": "d1 system_one score decisions",
    }
    (cfg.output / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report = {
        "concept_count": len(concepts),
        "dimension_count": len(dimensions),
        "vector_shape": list(means.shape),
        "mean_repeat_std": float(stds.mean()),
        "max_repeat_std": float(stds.max()),
        "nonzero_fraction": float(np.count_nonzero(means) / means.size),
    }
    (cfg.output / "quality.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
