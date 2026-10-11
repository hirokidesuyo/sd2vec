"""Continuous auto-pipeline for 200,000-word vocabulary run.

Automatically queues and advances chunks across Colab (L4) and 5060 Ti
until all 20 chunks (200,000 words) are completed or stopped by user.
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

ROOT_DIR = Path("/Users/ishikawahiroki/sd2vec")
VOCAB_DIR = ROOT_DIR / "large_vocab_200000"
CONCEPTS_DIR = VOCAB_DIR / "concepts"
VECTORS_DIR = VOCAB_DIR / "vectors"
STATE_FILE = VOCAB_DIR / "pipeline_state.json"
DIMENSIONS_FILE = ROOT_DIR / "dimensions_phase3.json"

TOTAL_CHUNKS = 20
CHUNK_SIZE = 10000


def load_state() -> dict[str, Any]:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))

    # Initial state
    chunks = {}
    for i in range(TOTAL_CHUNKS):
        cid = f"chunk_{i:04d}"
        if i == 0:
            chunks[cid] = {"status": "in_progress", "worker": "5060ti", "progress": 0}
        elif i == 1:
            chunks[cid] = {"status": "in_progress", "worker": "colab", "progress": 0}
        else:
            chunks[cid] = {"status": "pending", "worker": None, "progress": 0}

    state = {
        "total_chunks": TOTAL_CHUNKS,
        "chunk_size": CHUNK_SIZE,
        "active_colab": "chunk_0001",
        "active_5060ti": "chunk_0000",
        "chunks": chunks,
    }
    save_state(state)
    return state


def save_state(state: dict[str, Any]) -> None:
    STATE_FILE.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def get_local_count(chunk_id: str) -> int:
    path = VECTORS_DIR / chunk_id / "raw_scores.jsonl"
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def is_chunk_complete(chunk_id: str) -> bool:
    target_dir = VECTORS_DIR / chunk_id
    if (target_dir / "vectors.npy").exists() and (target_dir / "quality.json").exists():
        return True
    return get_local_count(chunk_id) >= CHUNK_SIZE


def check_and_sync_colab(active_chunk: str) -> tuple[int, bool | None, str]:
    if not active_chunk:
        return 0, False, "No active colab chunk"

    local_chunk_dir = VECTORS_DIR / active_chunk
    local_chunk_dir.mkdir(parents=True, exist_ok=True)

    check_code = f"""
import os, subprocess
cache = "/content/vectors_{active_chunk}/raw_scores.jsonl"
count = 0
if os.path.exists(cache):
    count = sum(1 for _ in open(cache))

ps_out = subprocess.check_output(["ps", "aux"], text=True)
is_running = any("d1_sd2vec" in l and "grep" not in l for l in ps_out.splitlines())
status_line = ""
log_path = "/content/run_{active_chunk}.log"
if os.path.exists(log_path):
    lines = open(log_path).readlines()
    if lines:
        status_line = lines[-1].strip()

print(f"COUNT:{{count}}")
print(f"RUNNING:{{is_running}}")
print(f"STATUS:{{status_line}}")
"""
    tmp_path = Path("/tmp/colab_auto_check.py")
    tmp_path.write_text(check_code, encoding="utf-8")

    out = None
    for attempt in range(1, 4):
        try:
            out = subprocess.check_output(
                ["colab", "exec", "-s", "sd2vec-colab", "-f", str(tmp_path), "--timeout", "50"],
                text=True,
                timeout=60,
                stderr=subprocess.DEVNULL,
            )
            break
        except Exception as e:
            if attempt < 3:
                time.sleep(2)
            else:
                return get_local_count(active_chunk), None, f"Colab check error: {e}"

    count = 0
    is_running = False
    status = ""
    for line in out.splitlines():
        if line.startswith("COUNT:"):
            count = int(line.split(":", 1)[1])
        elif line.startswith("RUNNING:"):
            is_running = line.split(":", 1)[1].strip() == "True"
        elif line.startswith("STATUS:"):
            status = line.split(":", 1)[1].strip()

    # Sync raw_scores.jsonl
    if count > 0:
        try:
            subprocess.run(
                [
                    "colab", "download", "-s", "sd2vec-colab",
                    f"/content/vectors_{active_chunk}/raw_scores.jsonl",
                    str(local_chunk_dir / "raw_scores.jsonl"),
                ],
                capture_output=True,
                check=False,
                timeout=45,
            )
        except Exception:
            pass

    return count, is_running, status


def finalize_colab_chunk(active_chunk: str) -> None:
    print(f"[Pipeline] Finalizing completed Colab chunk '{active_chunk}'...")
    local_chunk_dir = VECTORS_DIR / active_chunk
    local_chunk_dir.mkdir(parents=True, exist_ok=True)

    for fname in ["vectors.npy", "repeat_std.npy", "metadata.json", "quality.json", "raw_scores.jsonl"]:
        try:
            subprocess.run(
                [
                    "colab", "download", "-s", "sd2vec-colab",
                    f"/content/vectors_{active_chunk}/{fname}",
                    str(local_chunk_dir / fname),
                ],
                capture_output=True,
                check=False,
                timeout=45,
            )
        except Exception:
            pass


def start_colab_chunk(next_chunk: str) -> None:
    print(f"[Pipeline] Starting next Colab chunk '{next_chunk}'...")
    local_concept_file = CONCEPTS_DIR / f"{next_chunk}.txt"
    remote_concept_file = f"/content/{next_chunk}.txt"

    # Upload concept file with retry
    uploaded = False
    for attempt in range(1, 4):
        try:
            subprocess.run(
                ["colab", "upload", "-s", "sd2vec-colab", str(local_concept_file), remote_concept_file],
                check=True,
                timeout=45,
            )
            uploaded = True
            break
        except Exception as e:
            print(f"[Pipeline] Colab upload attempt {attempt} failed: {e}")
            time.sleep(3)

    if not uploaded:
        print(f"[Pipeline] Failed to upload '{next_chunk}' after 3 attempts.")
        return

    # Launch background run with retry
    start_code = f"""
import os
cmd = "nohup python3 /content/d1_sd2vec.py --model LiquidAI/d1-3B --concepts {remote_concept_file} --dimensions /content/dimensions_phase3.json --output /content/vectors_{next_chunk} --batch-size 16 > /content/run_{next_chunk}.log 2>&1 &"
os.system(cmd)
"""
    tmp_path = Path("/tmp/colab_auto_start.py")
    tmp_path.write_text(start_code, encoding="utf-8")
    for attempt in range(1, 4):
        try:
            subprocess.run(
                ["colab", "exec", "-s", "sd2vec-colab", "-f", str(tmp_path), "--timeout", "30"],
                check=True,
                timeout=35,
            )
            print(f"[Pipeline] Successfully launched '{next_chunk}' on Colab.")
            return
        except Exception as e:
            print(f"[Pipeline] Colab launch exec attempt {attempt} failed: {e}")
            time.sleep(3)


def start_5060ti_chunk(next_chunk: str) -> None:
    print(f"[Pipeline] Starting next 5060 Ti chunk '{next_chunk}'...")
    concept_file = CONCEPTS_DIR / f"{next_chunk}.txt"
    output_dir = VECTORS_DIR / next_chunk
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(ROOT_DIR / ".venv/bin/python"),
        str(ROOT_DIR / "build_sd2vec_remote.py"),
        "--concepts", str(concept_file),
        "--dimensions", str(DIMENSIONS_FILE),
        "--output", str(output_dir),
        "--workers", "2",
    ]
    log_file = output_dir / "run.log"
    out = open(log_file, "a", encoding="utf-8")
    subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
    print(f"[Pipeline] Successfully launched '{next_chunk}' on 5060 Ti (log: {log_file}).")


def find_next_pending(state: dict[str, Any]) -> str | None:
    for cid, info in state["chunks"].items():
        if info["status"] == "pending":
            return cid
    return None


def main() -> None:
    state = load_state()
    active_colab = state.get("active_colab")
    active_5060ti = state.get("active_5060ti")

    # 1. Update 5060 Ti
    if active_5060ti:
        c0 = get_local_count(active_5060ti)
        state["chunks"][active_5060ti]["progress"] = c0
        if is_chunk_complete(active_5060ti):
            print(f"[Pipeline] 5060 Ti finished '{active_5060ti}'!")
            state["chunks"][active_5060ti]["status"] = "completed"
            next_chunk = find_next_pending(state)
            if next_chunk:
                state["chunks"][next_chunk]["status"] = "in_progress"
                state["chunks"][next_chunk]["worker"] = "5060ti"
                state["active_5060ti"] = next_chunk
                start_5060ti_chunk(next_chunk)
            else:
                state["active_5060ti"] = None

    # 2. Update Colab
    if active_colab:
        c1, is_running, status = check_and_sync_colab(active_colab)
        state["chunks"][active_colab]["progress"] = c1
        if c1 >= CHUNK_SIZE or is_chunk_complete(active_colab):
            print(f"[Pipeline] Colab finished '{active_colab}'!")
            finalize_colab_chunk(active_colab)
            state["chunks"][active_colab]["status"] = "completed"
            next_chunk = find_next_pending(state)
            if next_chunk:
                state["chunks"][next_chunk]["status"] = "in_progress"
                state["chunks"][next_chunk]["worker"] = "colab"
                state["active_colab"] = next_chunk
                start_colab_chunk(next_chunk)
            else:
                state["active_colab"] = None
        elif is_running is False and c1 < CHUNK_SIZE:
            print(f"[Pipeline] Colab process confirmed stopped for '{active_colab}' (progress={c1}). Resuming...")
            start_colab_chunk(active_colab)
        elif is_running is None:
            print(f"[Pipeline] Transient Colab connection issue; skipping process restart check for '{active_colab}'.")

    save_state(state)

    # Summary
    completed_chunks = sum(1 for c in state["chunks"].values() if c["status"] == "completed")
    total_progress = sum(c["progress"] for c in state["chunks"].values())

    print("\n=== SD2Vec 200,000-Word Pipeline Progress ===")
    print(f"  Completed Chunks: {completed_chunks} / {TOTAL_CHUNKS}")
    print(f"  Total Words Processed: {total_progress:,} / 200,000 ({total_progress / 2000:.1f}%)")
    if state["active_5060ti"]:
        p0 = state["chunks"][state["active_5060ti"]]["progress"]
        print(f"  [5060 Ti] Active: {state['active_5060ti']} ({p0:,} / 10,000)")
    if state["active_colab"]:
        p1 = state["chunks"][state["active_colab"]]["progress"]
        print(f"  [Colab L4] Active: {state['active_colab']} ({p1:,} / 10,000)")


if __name__ == "__main__":
    main()
