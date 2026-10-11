"""Build SD vectors using remote d1-3B llama-server (/v1/systemone).

Keeps all files locally on this Mac, sending only inference requests to the
remote RTX 5060 Ti server. Resumable via raw_scores.jsonl cache.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate SD vectors via remote d1-3B llama-server")
    p.add_argument("--api", default="http://133.37.59.97:8080", help="Remote llama-server URL")
    p.add_argument("--concepts", type=Path, default=Path("concepts_chive_1000.txt"), help="Concepts text file")
    p.add_argument("--dimensions", type=Path, default=Path("dimensions_phase3.json"), help="Dimensions JSON file")
    p.add_argument("--output", type=Path, default=Path("outputs_remote_bf16"), help="Output directory on Mac")
    p.add_argument("--limit", type=int, default=0, help="Limit number of concepts (0 = all)")
    p.add_argument("--repeats", type=int, default=1, help="Number of evaluation repeats per concept")
    p.add_argument("--workers", type=int, default=2, help="Concurrent workers (llama-server has multiple slots)")
    p.add_argument("--timeout", type=int, default=120, help="HTTP request timeout in seconds")
    return p.parse_args()


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def build_questions(dimensions: list[dict[str, str]]) -> dict[str, Any]:
    questions = {}
    for d in dimensions:
        criteria = [
            f"-3（非常に{d['right']}）",
            f"-2（かなり{d['right']}）",
            f"-1（やや{d['right']}）",
            "0（中立）",
            f"1（やや{d['left']}）",
            f"2（かなり{d['left']}）",
            f"3（非常に{d['left']}）",
        ]
        questions[d["id"]] = {
            "type": "score",
            "instructions": (
                f"この概念は「{d['left']}」と「{d['right']}」のどちらの印象に近いですか。"
                "一般的な日本語話者の印象として評定してください。"
            ),
            "criteria": criteria,
        }
    return questions


def query_concept(
    api_url: str,
    concept: str,
    questions: dict[str, Any],
    dimension_ids: list[str],
    timeout: int = 120,
    max_retries: int = 3,
) -> dict[str, float]:
    payload = json.dumps({"state": f"概念: {concept}", "questions": questions}).encode("utf-8")
    req = urllib.request.Request(
        f"{api_url}/v1/systemone",
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            answers = data.get("answers", {})
            scores = {}
            for dim_id in dimension_ids:
                if dim_id not in answers or "score" not in answers[dim_id]:
                    raise ValueError(f"Missing score for dimension '{dim_id}' in response: {answers}")
                scores[dim_id] = float(answers[dim_id]["score"]) - 3.0
            return scores
        except Exception as e:
            if attempt == max_retries - 1:
                raise RuntimeError(f"Failed to query concept '{concept}' after {max_retries} attempts: {e}") from e
            time.sleep(2 * (attempt + 1))


def main() -> None:
    cfg = parse_args()
    cfg.output.mkdir(parents=True, exist_ok=True)

    dimensions = load_json(cfg.dimensions)
    dimension_ids = [d["id"] for d in dimensions]
    questions = build_questions(dimensions)

    all_concepts = [
        line.strip()
        for line in cfg.concepts.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if cfg.limit > 0:
        concepts = all_concepts[: cfg.limit]
    else:
        concepts = all_concepts

    print(f"=== SD2Vec Remote Generation ===")
    print(f"API: {cfg.api}")
    print(f"Concepts: {len(concepts)} (from {cfg.concepts})")
    print(f"Dimensions: {len(dimensions)} (from {cfg.dimensions})")
    print(f"Repeats: {cfg.repeats}")
    print(f"Workers: {cfg.workers}")
    print(f"Output directory: {cfg.output}")

    cache_path = cfg.output / "raw_scores.jsonl"
    done: dict[tuple[str, int], dict[str, float]] = {}
    if cache_path.exists():
        for line in cache_path.read_text(encoding="utf-8").splitlines():
            if line:
                row = json.loads(line)
                done[(row["concept"], row["repeat"])] = row["scores"]
        print(f"Loaded {len(done)} cached concept scores from {cache_path}")

    # Determine tasks to process
    tasks_to_do: list[tuple[str, int]] = []
    for repeat in range(cfg.repeats):
        for concept in concepts:
            if (concept, repeat) not in done:
                tasks_to_do.append((concept, repeat))

    print(f"Remaining tasks to process: {len(tasks_to_do)}")

    if tasks_to_do:
        started_all = time.perf_counter()
        completed_count = 0

        with cache_path.open("a", encoding="utf-8") as out:
            with concurrent.futures.ThreadPoolExecutor(max_workers=cfg.workers) as executor:
                future_to_task = {
                    executor.submit(
                        query_concept,
                        cfg.api,
                        concept,
                        questions,
                        dimension_ids,
                        cfg.timeout,
                    ): (concept, repeat)
                    for concept, repeat in tasks_to_do
                }

                for future in concurrent.futures.as_completed(future_to_task):
                    concept, repeat = future_to_task[future]
                    try:
                        scores = future.result()
                    except Exception as e:
                        print(f"Error processing '{concept}' (repeat {repeat}): {e}")
                        continue

                    done[(concept, repeat)] = scores
                    row = {"concept": concept, "repeat": repeat, "scores": scores}
                    out.write(json.dumps(row, ensure_ascii=False) + "\n")
                    out.flush()

                    completed_count += 1
                    elapsed_total = time.perf_counter() - started_all
                    rate = completed_count / elapsed_total
                    remaining_sec = (len(tasks_to_do) - completed_count) / rate if rate > 0 else 0
                    print(
                        f"[{completed_count}/{len(tasks_to_do)}] concept='{concept}' "
                        f"speed={rate:.2f} items/s remaining={remaining_sec / 60:.1f}min"
                    )

    # Compile final vectors
    print("\nCompiling final vector matrices...")
    matrix = np.array(
        [[done[(c, r)][d] for d in dimension_ids] for c in concepts for r in range(cfg.repeats)],
        dtype=np.float32,
    ).reshape(len(concepts), cfg.repeats, len(dimensions))

    means = matrix.mean(axis=1)
    stds = matrix.std(axis=1)

    np.save(cfg.output / "vectors.npy", means)
    np.save(cfg.output / "repeat_std.npy", stds)

    # Save CSV
    csv_path = cfg.output / "vectors.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["concept"] + dimension_ids)
        for c, vec in zip(concepts, means):
            writer.writerow([c] + [f"{v:.6f}" for v in vec])

    metadata = {
        "model": "LiquidAI/d1-3B-BF16-GGUF",
        "api": cfg.api,
        "concepts": concepts,
        "dimensions": dimensions,
        "repeats": cfg.repeats,
        "scale": [-3, 3],
        "method": "d1 system_one score decisions via llama-server",
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

    print(f"Finished successfully! Results saved to: {cfg.output}")
    print(f"  vectors.npy: {cfg.output / 'vectors.npy'}")
    print(f"  vectors.csv: {cfg.output / 'vectors.csv'}")
    print(f"  quality.json: {report}")


if __name__ == "__main__":
    main()
