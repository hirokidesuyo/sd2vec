"""Periodic monitor and sync script for 20,000-word run.

Syncs Colab chunk_0001 progress to Mac, monitors 5060 Ti chunk_0000 progress,
and restarts Colab if disconnected.
"""

import json
import os
import subprocess
import time
from pathlib import Path

BASE_DIR = Path("/Users/ishikawahiroki/sd2vec/large_vocab_200000/vectors")
CHUNK0_DIR = BASE_DIR / "chunk_0000"
CHUNK1_DIR = BASE_DIR / "chunk_0001"
CHUNK1_DIR.mkdir(parents=True, exist_ok=True)


def get_local_chunk0_count() -> int:
    path = CHUNK0_DIR / "raw_scores.jsonl"
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def check_and_sync_colab() -> tuple[int, bool, str]:
    # Check status and process
    check_code = """
import os, subprocess
cache = "/content/vectors_chunk_0001/raw_scores.jsonl"
count = 0
if os.path.exists(cache):
    count = sum(1 for _ in open(cache))

ps_out = subprocess.check_output(["ps", "aux"], text=True)
is_running = any("d1_sd2vec" in l and "grep" not in l for l in ps_out.splitlines())
status_line = ""
if os.path.exists("/content/run_chunk_0001.log"):
    lines = open("/content/run_chunk_0001.log").readlines()
    if lines:
        status_line = lines[-1].strip()

print(f"COUNT:{count}")
print(f"RUNNING:{is_running}")
print(f"STATUS:{status_line}")
"""
    tmp_path = Path("/tmp/colab_monitor_check.py")
    tmp_path.write_text(check_code, encoding="utf-8")

    try:
        out = subprocess.check_output(
            ["colab", "exec", "-s", "sd2vec-colab", "-f", str(tmp_path), "--timeout", "60"],
            text=True,
        )
    except Exception as e:
        return 0, False, f"Colab exec error: {e}"

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

    # Sync raw_scores.jsonl if count > 0
    if count > 0:
        try:
            subprocess.run(
                [
                    "colab", "download", "-s", "sd2vec-colab",
                    "/content/vectors_chunk_0001/raw_scores.jsonl",
                    str(CHUNK1_DIR / "raw_scores.jsonl"),
                ],
                capture_output=True,
                check=False,
            )
        except Exception:
            pass

    return count, is_running, status


def restart_colab() -> None:
    print("[Colab Monitor] Restarting Colab background run...")
    start_code = """
import os
cmd = "nohup python3 /content/d1_sd2vec.py --model LiquidAI/d1-3B --concepts /content/chunk_0001.txt --dimensions /content/dimensions_phase3.json --output /content/vectors_chunk_0001 --batch-size 16 > /content/run_chunk_0001.log 2>&1 &"
os.system(cmd)
"""
    tmp_path = Path("/tmp/colab_restart.py")
    tmp_path.write_text(start_code, encoding="utf-8")
    try:
        subprocess.run(
            ["colab", "exec", "-s", "sd2vec-colab", "-f", str(tmp_path), "--timeout", "60"],
            check=True,
        )
        print("[Colab Monitor] Restart command sent successfully.")
    except Exception as e:
        print(f"[Colab Monitor] Restart failed: {e}")


def main() -> None:
    c0 = get_local_chunk0_count()
    c1, is_running, status = check_and_sync_colab()

    print(f"=== SD2Vec 20,000-Word Progress ===")
    print(f"  [5060 Ti]  chunk_0000: {c0:>5} / 10,000 ({c0 / 100:>5.1f}%)")
    print(f"  [Colab L4] chunk_0001: {c1:>5} / 10,000 ({c1 / 100:>5.1f}%) | Running: {is_running} | {status}")
    print(f"  Total Progress: {c0 + c1:>5} / 20,000 ({(c0 + c1) / 200:>5.1f}%)")

    # If Colab not running and not finished, restart
    if not is_running and c1 < 10000:
        restart_colab()


if __name__ == "__main__":
    main()
