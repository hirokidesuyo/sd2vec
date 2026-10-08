"""Select a reproducible concept vocabulary from a chiVe text model.

The chiVe text file is frequency ordered. We scan only the first `scan_limit`
rows, exclude obvious non-concepts, and retain the first `limit` candidates.
"""

from __future__ import annotations

import argparse
import csv
import re
import tarfile
from pathlib import Path

URL_RE = re.compile(r"(https?://|www\.|[@＠])", re.I)
NUMBER_RE = re.compile(r"^[0-9０-９.,，．・/／:+＋−_＿\-]+$")
SYMBOL_RE = re.compile(r"^[^\wぁ-んァ-ヶ一-龠々ー]+$", re.UNICODE)
FUNCTION_WORDS = {
    "の", "に", "は", "を", "が", "と", "も", "へ", "で", "や", "から",
    "まで", "より", "だけ", "しか", "ほど", "など", "こと", "もの", "ため",
    "よう", "そう", "これ", "それ", "あれ", "ここ", "そこ", "あそこ",
    "する", "いる", "ある", "なる", "できる", "れる", "られる", "せる",
    "させる", "です", "ます", "だ", "だった", "ない", "なく", "たい",
    "た", "て", "ます", "という", "及び", "または", "しかし", "そして",
}


def valid(word: str) -> tuple[bool, str]:
    if not word:
        return False, "empty"
    if URL_RE.search(word):
        return False, "url_or_handle"
    if NUMBER_RE.fullmatch(word):
        return False, "number"
    if SYMBOL_RE.fullmatch(word):
        return False, "symbol"
    if word in FUNCTION_WORDS:
        return False, "function_word"
    if len(word) > 30:
        return False, "too_long"
    return True, ""


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("archive", type=Path)
    p.add_argument("--limit", type=int, default=10_000)
    p.add_argument("--scan-limit", type=int, default=50_000)
    p.add_argument("--output", type=Path, default=Path("chive_vocab_10000.csv"))
    args = p.parse_args()

    selected = []
    scanned = 0
    with tarfile.open(args.archive, "r:gz") as archive:
        member = next(item for item in archive.getmembers() if item.name.endswith(".txt"))
        stream = archive.extractfile(member)
        if stream is None:
            raise RuntimeError(f"cannot read {member.name}")
        header = stream.readline().decode("utf-8").strip().split()
        if len(header) != 2 or int(header[1]) != 300:
            raise ValueError(f"unexpected chiVe header: {header}")
        for raw in stream:
            if scanned >= args.scan_limit or len(selected) >= args.limit:
                break
            scanned += 1
            word = raw.decode("utf-8").split(" ", 1)[0]
            ok, reason = valid(word)
            if ok:
                selected.append((len(selected) + 1, scanned, word, reason))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["selected_rank", "chive_rank", "word", "filter_reason"])
        writer.writerows(selected)
    print(f"selected={len(selected)} scanned={scanned} output={args.output}")


if __name__ == "__main__":
    main()
