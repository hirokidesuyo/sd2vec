"""Small dependency-free web app for explaining sd2vec pair relationships."""

from __future__ import annotations

import argparse
import json
import mimetypes
import threading
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import numpy as np


DEFAULT_DATA = Path(__file__).parent / "outputs_chive_10000_phase3"
PAGE_TITLE = "sd2vec 軸で見る意味比較"


def load_model(directory: Path) -> tuple[list[str], list[dict[str, str]], np.ndarray]:
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    vectors = np.load(directory / "vectors.npy").astype(np.float64)
    dimensions = metadata["dimensions"]
    return metadata["concepts"], dimensions, vectors


def explain_pair(
    left_word: str,
    right_word: str,
    concepts: list[str],
    dimensions: list[dict[str, str]],
    vectors: np.ndarray,
) -> dict[str, object]:
    index = {word: i for i, word in enumerate(concepts)}
    missing = [word for word in (left_word, right_word) if word not in index]
    if missing:
        raise KeyError(f"語彙にありません: {', '.join(missing)}")
    left = vectors[index[left_word]]
    right = vectors[index[right_word]]
    difference = left - right
    left_norm = np.linalg.norm(left)
    right_norm = np.linalg.norm(right)
    cosine = float(left @ right / max(left_norm * right_norm, 1e-12))
    euclidean = float(np.linalg.norm(difference))
    rows = []
    for item, left_value, right_value, delta in zip(
        dimensions, left, right, difference
    ):
        scale = max(abs(float(left_value)), abs(float(right_value)), 1e-12)
        if float(left_value) * float(right_value) < 0:
            relation = "opposite"
        elif abs(float(delta)) <= 0.35:
            relation = "shared"
        else:
            relation = "different"
        rows.append(
            {
                "id": item["id"],
                "left_label": item["left"],
                "right_label": item["right"],
                "left": round(float(left_value), 4),
                "right": round(float(right_value), 4),
                "difference": round(float(delta), 4),
                "relative_difference": round(float(delta) / scale, 4),
                "relation": relation,
            }
        )
    shared = sorted(
        (row for row in rows if row["relation"] == "shared"),
        key=lambda row: abs(row["difference"]),
    )
    opposite = sorted(
        (row for row in rows if row["relation"] == "opposite"),
        key=lambda row: abs(row["difference"]),
        reverse=True,
    )
    different = sorted(
        (row for row in rows if row["relation"] != "shared"),
        key=lambda row: abs(row["difference"]),
        reverse=True,
    )
    return {
        "left_word": left_word,
        "right_word": right_word,
        "cosine": round(cosine, 6),
        "euclidean": round(euclidean, 6),
        "mean_absolute_difference": round(float(np.mean(np.abs(difference))), 6),
        "shared_axes": shared[:10],
        "opposite_axes": opposite[:10],
        "different_axes": different[:10],
        "all_axes": rows,
    }


HTML = r"""<!doctype html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>sd2vec 軸で見る意味比較</title>
<style>
:root { color-scheme: light; --ink:#172033; --muted:#687386; --line:#dce2eb; --accent:#4263eb; --soft:#f5f7fb; --bad:#d9485f; --good:#087f5b; }
* { box-sizing:border-box; }
body { margin:0; background:var(--soft); color:var(--ink); font-family:system-ui,-apple-system,"Segoe UI","Noto Sans JP",sans-serif; }
main { max-width:1120px; margin:0 auto; padding:32px 18px 64px; }
h1 { margin:0 0 8px; font-size:clamp(26px,4vw,40px); }
h2 { margin:0 0 14px; font-size:20px; }
p { color:var(--muted); line-height:1.7; }
.hero { margin-bottom:24px; }
.card { background:white; border:1px solid var(--line); border-radius:16px; padding:20px; margin:16px 0; box-shadow:0 4px 16px #1720330a; }
.form { display:grid; grid-template-columns:1fr auto 1fr auto; gap:10px; align-items:end; }
label { display:block; color:var(--muted); font-size:13px; margin-bottom:6px; }
input { width:100%; border:1px solid #b9c2d0; border-radius:9px; padding:12px; font-size:17px; }
button { border:0; border-radius:9px; padding:12px 18px; background:var(--accent); color:white; cursor:pointer; font-size:15px; font-weight:600; }
button:hover { filter:brightness(.94); }
.presets { display:flex; flex-wrap:wrap; gap:8px; margin-top:14px; }
.preset { background:#edf2ff; color:#364fc7; padding:8px 11px; font-size:13px; }
.error { background:#fff0f1; color:#a61e2b; border:1px solid #ffc9ce; padding:12px; border-radius:9px; }
.metrics { display:grid; grid-template-columns:repeat(3,1fr); gap:12px; }
.metric { background:var(--soft); border-radius:12px; padding:15px; }
.metric b { display:block; font-size:27px; margin-top:5px; }
.metric small { color:var(--muted); }
.grid { display:grid; grid-template-columns:repeat(3,1fr); gap:14px; }
.panel { border:1px solid var(--line); border-radius:12px; padding:14px; }
.panel h3 { margin:0 0 5px; font-size:16px; }
.panel p { font-size:13px; margin:0 0 12px; }
.axis { padding:10px 0; border-top:1px solid var(--line); }
.axis:first-of-type { border-top:0; }
.axis-title { display:flex; justify-content:space-between; gap:8px; font-weight:650; }
.axis-label { color:var(--muted); font-size:12px; margin-top:3px; }
.bar { height:7px; background:#edf0f5; border-radius:8px; margin-top:7px; overflow:hidden; }
.bar > i { display:block; height:100%; border-radius:8px; background:var(--accent); }
.bar > i.negative { background:#e67700; }
.badge { border-radius:99px; padding:3px 8px; font-size:11px; }
.badge.opposite { background:#fff0f1; color:var(--bad); }
.badge.shared { background:#e6fcf5; color:var(--good); }
.badge.different { background:#fff4e6; color:#ad5d00; }
table { width:100%; border-collapse:collapse; font-size:13px; }
th,td { text-align:left; padding:9px 7px; border-bottom:1px solid var(--line); }
th { color:var(--muted); font-weight:600; }
.hidden { display:none; }
details summary { cursor:pointer; color:var(--accent); }
@media(max-width:760px) { .form,.metrics,.grid { grid-template-columns:1fr; } .form > span { display:none; } main { padding-top:20px; } }
</style>
</head>
<body>
<main>
  <section class="hero">
    <h1>sd2vec 軸で見る意味比較</h1>
    <p>2語が「近い・遠い」だけでなく、どのSD尺度を共有し、どの尺度で異なるのかを調べます。</p>
  </section>
  <section class="card">
    <div class="form">
      <div><label for="left">左の語</label><input id="left" value="勝利" list="words"></div>
      <span>と</span>
      <div><label for="right">右の語</label><input id="right" value="敗北" list="words"></div>
      <button onclick="analyze()">比較する</button>
    </div>
    <div class="presets">
      <button class="preset" onclick="setPair('勝利','敗北')">勝利 / 敗北</button>
      <button class="preset" onclick="setPair('春','秋')">春 / 秋</button>
      <button class="preset" onclick="setPair('高い','低い')">高い / 低い</button>
      <button class="preset" onclick="setPair('倹約','ケチ')">倹約 / ケチ</button>
      <button class="preset" onclick="setPair('痩せている','ガリガリ')">痩せている / ガリガリ</button>
    </div>
    <datalist id="words"></datalist>
    <div id="error" class="error hidden"></div>
  </section>
  <section id="result" class="hidden">
    <section class="card">
      <h2 id="headline"></h2>
      <div class="metrics">
        <div class="metric"><small>コサイン類似度</small><b id="cosine"></b></div>
        <div class="metric"><small>ユークリッド距離</small><b id="euclidean"></b></div>
        <div class="metric"><small>平均絶対差</small><b id="mad"></b></div>
      </div>
      <p>コサインは全体の方向、ユークリッド距離は位置の離れ具合を表します。下の軸別表示が「なぜ」を説明します。</p>
    </section>
    <section class="grid">
      <div class="panel"><h3>共通している軸</h3><p>2語の評定が近い尺度。意味の共通部分です。</p><div id="shared"></div></div>
      <div class="panel"><h3>反対方向の軸</h3><p>一方が左極、もう一方が右極にある尺度。</p><div id="opposite"></div></div>
      <div class="panel"><h3>差が大きい軸</h3><p>語感・ニュアンスの違いを作る尺度。</p><div id="different"></div></div>
    </section>
    <section class="card"><h2>全尺度</h2><div id="all"></div></section>
  </section>
</main>
<script>
const $ = id => document.getElementById(id);
function setPair(a,b) { $('left').value=a; $('right').value=b; analyze(); }
function esc(s) { return String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])); }
function axis(row) {
  const width = Math.min(100, Math.abs(row.difference) / 3 * 100);
  const cls = row.relation;
  const bar = `<div class="bar"><i class="${row.difference < 0 ? 'negative':''}" style="width:${width}%"></i></div>`;
  return `<div class="axis"><div class="axis-title"><span>${esc(row.id)}</span><span class="badge ${cls}">${cls==='shared'?'共通':cls==='opposite'?'反対':'差'}</span></div><div class="axis-label">${esc(row.left_label)} / ${esc(row.right_label)}　${row.left.toFixed(2)} / ${row.right.toFixed(2)}　差 ${row.difference>0?'+':''}${row.difference.toFixed(2)}</div>${bar}</div>`;
}
function renderList(id, rows) { $(id).innerHTML = rows.length ? rows.map(axis).join('') : '<p>該当する軸はありません。</p>'; }
function analyze() {
  const left = $('left').value.trim(), right = $('right').value.trim();
  $('error').classList.add('hidden');
  fetch(`/api/explain?left=${encodeURIComponent(left)}&right=${encodeURIComponent(right)}`)
    .then(async r => { const data=await r.json(); if(!r.ok) throw new Error(data.error); return data; })
    .then(data => {
      $('result').classList.remove('hidden'); $('headline').textContent=`${data.left_word} と ${data.right_word}`;
      $('cosine').textContent=data.cosine.toFixed(4); $('euclidean').textContent=data.euclidean.toFixed(4); $('mad').textContent=data.mean_absolute_difference.toFixed(4);
      renderList('shared', data.shared_axes); renderList('opposite', data.opposite_axes); renderList('different', data.different_axes);
      $('all').innerHTML=`<table><thead><tr><th>尺度</th><th>${esc(data.left_word)}</th><th>${esc(data.right_word)}</th><th>差</th><th>関係</th></tr></thead><tbody>${data.all_axes.map(r=>`<tr><td>${esc(r.id)}</td><td>${r.left.toFixed(3)}</td><td>${r.right.toFixed(3)}</td><td>${r.difference>0?'+':''}${r.difference.toFixed(3)}</td><td>${r.relation}</td></tr>`).join('')}</tbody></table>`;
    }).catch(e => { $('error').textContent=e.message; $('error').classList.remove('hidden'); $('result').classList.add('hidden'); });
}
fetch('/api/words').then(r=>r.json()).then(words => $('words').innerHTML=words.map(w=>`<option value="${esc(w)}">`).join(''));
analyze();
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    concepts: list[str] = []
    dimensions: list[dict[str, str]] = []
    vectors: np.ndarray

    def send_json(self, payload: object, status: int = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            body = HTML.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if parsed.path == "/api/words":
            self.send_json(self.concepts)
            return
        if parsed.path == "/api/explain":
            from urllib.parse import parse_qs

            query = parse_qs(parsed.query)
            left = query.get("left", [""])[0]
            right = query.get("right", [""])[0]
            if not left or not right:
                self.send_json({"error": "2語を入力してください。"}, HTTPStatus.BAD_REQUEST)
                return
            try:
                result = explain_pair(
                    left, right, self.concepts, self.dimensions, self.vectors
                )
            except KeyError as error:
                self.send_json({"error": str(error).strip("'")}, HTTPStatus.NOT_FOUND)
                return
            self.send_json(result)
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def log_message(self, format: str, *args: object) -> None:
        print(f"[web] {format % args}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_DATA)
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="待受アドレス。Tailscaleから接続するには0.0.0.0またはTailscale IPを指定",
    )
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    concepts, dimensions, vectors = load_model(args.input)
    Handler.concepts = concepts
    Handler.dimensions = dimensions
    Handler.vectors = vectors
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    browser_host = "127.0.0.1" if args.host == "0.0.0.0" else args.host
    url = f"http://{browser_host}:{args.port}/"
    print(f"sd2vec web app: {url}")
    if args.host == "0.0.0.0":
        print(
            "全インターフェースで待受中です。Tailscale端末からは "
            f"http://<このPCのTailscale IP>:{args.port}/ に接続してください。"
        )
    if not args.no_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n停止しました。")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
