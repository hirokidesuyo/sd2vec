import json
import time
import urllib.request
from pathlib import Path

def main():
    dim_path = Path("/Users/ishikawahiroki/sd2vec/dimensions_phase3.json")
    dimensions = json.loads(dim_path.read_text(encoding="utf-8"))

    def question(d):
        return {
            "type": "score",
            "instructions": f"この概念は「{d['left']}」と「{d['right']}」のどちらの印象に近いですか。一般的な日本語話者の印象として評定してください。",
            "criteria": [
                f"-3（非常に{d['right']}）",
                f"-2（かなり{d['right']}）",
                f"-1（やや{d['right']}）",
                "0（中立）",
                f"1（やや{d['left']}）",
                f"2（かなり{d['left']}）",
                f"3（非常に{d['left']}）",
            ],
        }

    questions = {d["id"]: question(d) for d in dimensions}
    payload = json.dumps({"state": "概念: 猫", "questions": questions}).encode("utf-8")

    req = urllib.request.Request(
        "http://133.37.59.97:8080/v1/systemone",
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    print("Sending request for '概念: 猫' (50 dimensions) to 5060 Ti...")
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=120) as resp:
        res = json.loads(resp.read().decode("utf-8"))
    t1 = time.time()

    print(f"Success! Elapsed time: {t1 - t0:.3f} seconds")
    answers = res.get("answers", {})
    print(f"Total dimensions answered: {len(answers)} / {len(dimensions)}")
    print("Sample scores:")
    for d in dimensions[:8]:
        score = answers[d["id"]]["score"] - 3.0
        print(f"  {d['id']:<12} ({d['left']} ↔ {d['right']}): {score:+.3f}")

if __name__ == "__main__":
    main()
