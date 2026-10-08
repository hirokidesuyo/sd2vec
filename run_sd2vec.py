"""Build interpretable SD vectors with a local Hugging Face causal LM.

The script is restartable: each completed batch is appended to JSONL and
reused on subsequent runs. Scores are integers in [-3, 3], where positive
means the left adjective in dimensions.json.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-4B")
    p.add_argument("--concepts", type=Path, default=Path("concepts_phase2.txt"))
    p.add_argument("--dimensions", type=Path, default=Path("dimensions_phase2.json"))
    p.add_argument("--output", type=Path, default=Path("outputs_phase2"))
    p.add_argument("--limit", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--max-new-tokens", type=int, default=2048)
    p.add_argument("--temperature", type=float, default=0.7)
    # Jupyter kernels append their own "-f <connection-file>" argument.
    return p.parse_known_args()[0]


def prompt_for(concepts: list[str], dimensions: list[dict[str, str]]) -> str:
    scale = "、".join(f"{i}:{'左' if i > 0 else '右' if i < 0 else '中立'}" for i in range(-3, 4))
    dims = "\n".join(
        f'- "{d["id"]}": 左「{d["left"]}」 / 右「{d["right"]}」'
        for d in dimensions
    )
    schema = ", ".join(f'"{d["id"]}": 0' for d in dimensions)
    items = "\n".join(f'{i}: {c}' for i, c in enumerate(concepts))
    return f"""あなたは意味微分法(Semantic Differential)の評定者です。
各概念について、あなたが持つ一般常識から具体的な印象を判断し、尺度ごとに必ず評定値を決めてください。
尺度値は整数 -3,-2,-1,0,1,2,3 のみ。正数は左の形容詞、負数は右の形容詞を意味します。
0は本当に中立の場合だけ使い、概念ごとの差が出るようにしてください。例えば「猫」は
pleasant=2, familiar=2, natural=2, large=-1 のようになります。これは例なので他の概念には適用しないでください。
尺度: {scale}
尺度一覧:
{dims}
概念一覧:
{items}
説明文は禁止し、JSON配列だけを返してください。各要素は
{{"concept_index": 0, "scores": {{{schema}}}}}
の形式にしてください。全概念を漏れなく含めてください。"""


def extract_json(text: str) -> list[dict[str, Any]]:
    text = text.strip()
    candidates = [text]
    match = re.search(r"\[[\s\S]*\]", text)
    if match:
        candidates.insert(0, match.group(0))
    for candidate in candidates:
        try:
            value = json.loads(candidate)
            if isinstance(value, list):
                return value
        except json.JSONDecodeError:
            pass
    raise ValueError(f"JSON配列を抽出できません: {text[:500]}")


def normalize_result(raw: list[dict[str, Any]], concepts: list[str], dimensions: list[dict[str, str]]) -> dict[str, dict[str, int]]:
    ids = {d["id"] for d in dimensions}
    result: dict[str, dict[str, int]] = {}
    for item in raw:
        idx = int(item["concept_index"])
        if not 0 <= idx < len(concepts):
            continue
        scores = item.get("scores", {})
        clean = {}
        for dim in dimensions:
            value = int(round(float(scores[dim["id"]])))
            clean[dim["id"]] = max(-3, min(3, value))
        if set(clean) == ids:
            result[concepts[idx]] = clean
    return result


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    dimensions = load_json(args.dimensions)
    concepts = [x.strip() for x in args.concepts.read_text(encoding="utf-8").splitlines() if x.strip()][:args.limit]
    cache_path = args.output / "raw_scores.jsonl"
    completed: dict[tuple[str, int], dict[str, int]] = {}
    if cache_path.exists():
        for line in cache_path.read_text(encoding="utf-8").splitlines():
            if line:
                row = json.loads(line)
                completed[(row["concept"], row["repeat"])] = row["scores"]

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    model.eval()
    device = next(model.parameters()).device
    print(f"model={args.model} device={device} concepts={len(concepts)} dimensions={len(dimensions)}")

    with cache_path.open("a", encoding="utf-8") as cache:
        for repeat in range(args.repeats):
            remaining = [c for c in concepts if (c, repeat) not in completed]
            for start in range(0, len(remaining), args.batch_size):
                batch = remaining[start:start + args.batch_size]
                prompt = prompt_for(batch, dimensions)
                messages = [
                    {"role": "system", "content": "JSON以外の文字を出力しないでください。"},
                    {"role": "user", "content": prompt},
                ]
                try:
                    text = tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
                    )
                except TypeError:
                    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = tokenizer(text, return_tensors="pt").to(device)
                with torch.inference_mode():
                    output = model.generate(
                        **inputs,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=args.temperature > 0,
                        temperature=args.temperature,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                generated = tokenizer.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
                scores = normalize_result(extract_json(generated), batch, dimensions)
                missing = [c for c in batch if c not in scores]
                if missing:
                    # Some models occasionally truncate one item in a batch.
                    # Retry only the missing concepts individually.
                    for concept in missing:
                        retry_prompt = prompt_for([concept], dimensions)
                        retry_messages = [
                            {"role": "system", "content": "JSON以外の文字を出力しないでください。"},
                            {"role": "user", "content": retry_prompt},
                        ]
                        try:
                            retry_text = tokenizer.apply_chat_template(
                                retry_messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
                            )
                        except TypeError:
                            retry_text = tokenizer.apply_chat_template(
                                retry_messages, tokenize=False, add_generation_prompt=True
                            )
                        retry_inputs = tokenizer(retry_text, return_tensors="pt").to(device)
                        with torch.inference_mode():
                            retry_output = model.generate(
                                **retry_inputs,
                                max_new_tokens=args.max_new_tokens,
                                do_sample=args.temperature > 0,
                                temperature=args.temperature,
                                pad_token_id=tokenizer.eos_token_id,
                            )
                        retry_generated = tokenizer.decode(
                            retry_output[0][retry_inputs.input_ids.shape[1]:], skip_special_tokens=True
                        )
                        scores.update(normalize_result(extract_json(retry_generated), [concept], dimensions))
                missing = [c for c in batch if c not in scores]
                if missing:
                    raise RuntimeError(f"評定結果が不足しています: {missing}")
                for concept in batch:
                    row = {"concept": concept, "repeat": repeat, "scores": scores[concept]}
                    cache.write(json.dumps(row, ensure_ascii=False) + "\n")
                    cache.flush()
                    completed[(concept, repeat)] = scores[concept]
                print(f"completed repeat={repeat + 1}/{args.repeats} batch={start // args.batch_size + 1}")

    matrix = np.array([[completed[(c, r)][d["id"]] for d in dimensions] for c in concepts for r in range(args.repeats)], dtype=np.float32)
    repeated = matrix.reshape(len(concepts), args.repeats, len(dimensions))
    means = repeated.mean(axis=1)
    stds = repeated.std(axis=1)
    np.save(args.output / "vectors.npy", means)
    np.save(args.output / "repeat_std.npy", stds)
    metadata = {
        "model": args.model, "concepts": concepts,
        "dimensions": dimensions, "repeats": args.repeats,
        "scale": [-3, 3], "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    report = {
        "concept_count": len(concepts), "dimension_count": len(dimensions),
        "vector_shape": list(means.shape), "mean_repeat_std": float(stds.mean()),
        "max_repeat_std": float(stds.max()), "nonzero_fraction": float(np.count_nonzero(means) / means.size),
    }
    (args.output / "quality.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
