const DATA_ROOT = "data/";
const $ = (id) => document.getElementById(id);
let model = null;

const PRESETS = [
  ["春", "秋"], ["高い", "低い"], ["強い", "弱い"], ["明るい", "暗い"],
  ["上", "下"], ["生", "死"], ["成功", "失敗"], ["王", "女王"],
];

const MATH_PRESETS = [
  { a: "王", b: "男", c: "女", label: "王 − 男 ＋ 女" },
  { a: "日本", b: "東京", c: "パリ", label: "日本 − 東京 ＋ パリ" },
  { a: "猫", b: "", c: "大きい", label: "猫 ＋ 大きい" },
  { a: "夏", b: "暑い", c: "寒い", label: "夏 − 暑い ＋ 寒い" },
  { a: "太陽", b: "昼", c: "夜", label: "太陽 − 昼 ＋ 夜" },
];

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, (char) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  })[char]);
}

// ---------------------------------------------------------------------------
// 2-Word Comparison Logic
// ---------------------------------------------------------------------------
function explain(leftWord, rightWord) {
  const leftIndex = model.index[leftWord];
  const rightIndex = model.index[rightWord];
  if (leftIndex === undefined || rightIndex === undefined) {
    const missing = [leftIndex === undefined ? leftWord : null, rightIndex === undefined ? rightWord : null]
      .filter(Boolean).join("、");
    throw new Error(`語彙にありません: ${missing}`);
  }
  const left = model.vectors.subarray(leftIndex * model.dimension, (leftIndex + 1) * model.dimension);
  const right = model.vectors.subarray(rightIndex * model.dimension, (rightIndex + 1) * model.dimension);
  const difference = new Float64Array(model.dimension);
  let dot = 0, leftNorm = 0, rightNorm = 0, distance = 0, absSum = 0;
  const rows = [];
  for (let i = 0; i < model.dimension; i += 1) {
    const delta = left[i] - right[i];
    difference[i] = delta;
    dot += left[i] * right[i];
    leftNorm += left[i] * left[i];
    rightNorm += right[i] * right[i];
    distance += delta * delta;
    absSum += Math.abs(delta);
    let relation = "different";
    if (left[i] * right[i] < 0) relation = "opposite";
    else if (Math.abs(delta) <= 0.35) relation = "shared";
    rows.push({
      id: model.dimensions[i].id, leftLabel: model.dimensions[i].left, rightLabel: model.dimensions[i].right,
      left: left[i], right: right[i], difference: delta, relation,
    });
  }
  const byDifference = (a, b) => Math.abs(b.difference) - Math.abs(a.difference);
  return {
    leftWord, rightWord,
    cosine: dot / Math.max(Math.sqrt(leftNorm * rightNorm), 1e-12),
    euclidean: Math.sqrt(distance), meanAbsoluteDifference: absSum / model.dimension,
    shared: rows.filter((row) => row.relation === "shared").sort((a, b) => Math.abs(a.difference) - Math.abs(b.difference)).slice(0, 10),
    opposite: rows.filter((row) => row.relation === "opposite").sort(byDifference).slice(0, 10),
    different: rows.filter((row) => row.relation !== "shared").sort(byDifference).slice(0, 10),
    all: rows,
  };
}

function renderAxis(row) {
  const width = Math.min(100, Math.abs(row.difference) / 3 * 100);
  const label = row.relation === "shared" ? "共通" : row.relation === "opposite" ? "反対" : "差";
  return `<div class="axis"><div class="axis-title"><span>${escapeHtml(row.id)}</span><span class="badge ${row.relation}">${label}</span></div><div class="axis-label">${escapeHtml(row.leftLabel)} / ${escapeHtml(row.rightLabel)}　${row.left.toFixed(2)} / ${row.right.toFixed(2)}　差 ${row.difference >= 0 ? "+" : ""}${row.difference.toFixed(2)}</div><div class="bar"><i class="${row.difference < 0 ? "negative" : ""}" style="width:${width}%"></i></div></div>`;
}

function renderList(id, rows) {
  $(id).innerHTML = rows.length ? rows.map(renderAxis).join("") : "<p>該当する軸はありません。</p>";
}

function render(result) {
  $("result").classList.remove("hidden");
  $("headline").textContent = `${result.leftWord} と ${result.rightWord}`;
  $("cosine").textContent = result.cosine.toFixed(4);
  $("euclidean").textContent = result.euclidean.toFixed(4);
  $("mad").textContent = result.meanAbsoluteDifference.toFixed(4);
  renderList("shared", result.shared);
  renderList("opposite", result.opposite);
  renderList("different", result.different);
  $("all").innerHTML = `<table><thead><tr><th>尺度</th><th>${escapeHtml(result.leftWord)}</th><th>${escapeHtml(result.rightWord)}</th><th>差</th><th>関係</th></tr></thead><tbody>${result.all.map((row) => `<tr><td>${escapeHtml(row.id)}</td><td>${row.left.toFixed(3)}</td><td>${row.right.toFixed(3)}</td><td>${row.difference >= 0 ? "+" : ""}${row.difference.toFixed(3)}</td><td>${row.relation}</td></tr>`).join("")}</tbody></table>`;
}

function analyze() {
  $("error").classList.add("hidden");
  try {
    render(explain($("left").value.trim(), $("right").value.trim()));
  } catch (error) {
    $("result").classList.add("hidden");
    $("error").textContent = error.message;
    $("error").classList.remove("hidden");
  }
}

function setPair(left, right) {
  $("left").value = left; $("right").value = right;
  switchTab("tab-compare");
  analyze();
}
window.setPair = setPair;

// ---------------------------------------------------------------------------
// Concept Math (Analogy: A - B + C) Logic
// ---------------------------------------------------------------------------
function getWordDirection(word, useCenter) {
  const idx = model.index[word];
  if (idx === undefined) throw new Error(`語彙にありません: ${word}`);
  const dim = model.dimension;
  const offset = idx * dim;
  const vec = new Float32Array(dim);
  let normSq = 0;
  for (let d = 0; d < dim; d++) {
    const val = useCenter ? (model.vectors[offset + d] - model.meanVector[d]) : model.vectors[offset + d];
    vec[d] = val;
    normSq += val * val;
  }
  const norm = Math.max(Math.sqrt(normSq), 1e-12);
  for (let d = 0; d < dim; d++) vec[d] /= norm;
  return vec;
}

function runMath() {
  $("math-error").classList.add("hidden");
  try {
    const wordA = $("math-a").value.trim();
    const wordB = $("math-b").value.trim();
    const wordC = $("math-c").value.trim();

    if (!wordA) throw new Error("基準の語 (A) を入力してください。");

    const useCenter = $("math-center").checked;
    const dim = model.dimension;
    const query = new Float32Array(dim);
    const excluded = new Set([wordA]);

    const vecA = getWordDirection(wordA, useCenter);
    for (let d = 0; d < dim; d++) query[d] += vecA[d];

    if (wordB) {
      excluded.add(wordB);
      const vecB = getWordDirection(wordB, useCenter);
      for (let d = 0; d < dim; d++) query[d] -= vecB[d];
    }

    if (wordC) {
      excluded.add(wordC);
      const vecC = getWordDirection(wordC, useCenter);
      for (let d = 0; d < dim; d++) query[d] += vecC[d];
    }

    // Normalize query vector
    let qNormSq = 0;
    for (let d = 0; d < dim; d++) qNormSq += query[d] * query[d];
    const qNorm = Math.max(Math.sqrt(qNormSq), 1e-12);
    for (let d = 0; d < dim; d++) query[d] /= qNorm;

    // Fast inner product across 200,000 concepts
    const count = model.count;
    const scores = new Float32Array(count);
    for (let i = 0; i < count; i++) {
      const offset = i * dim;
      let dot = 0;
      if (useCenter) {
        for (let d = 0; d < dim; d++) {
          dot += (model.vectors[offset + d] - model.meanVector[d]) * query[d];
        }
        scores[i] = dot / Math.max(model.centeredNorms[i], 1e-12);
      } else {
        for (let d = 0; d < dim; d++) {
          dot += model.vectors[offset + d] * query[d];
        }
        scores[i] = dot / Math.max(model.rawNorms[i], 1e-12);
      }
    }

    // Extract top 10 (skipping excluded words)
    // Find top-10 using min-heap / threshold
    const topIndices = [];
    for (let i = 0; i < count; i++) {
      const word = model.concepts[i];
      if (excluded.has(word)) continue;
      const s = scores[i];
      if (topIndices.length < 10) {
        topIndices.push({ idx: i, word, score: s });
        if (topIndices.length === 10) topIndices.sort((a, b) => b.score - a.score);
      } else if (s > topIndices[9].score) {
        topIndices[9] = { idx: i, word, score: s };
        topIndices.sort((a, b) => b.score - a.score);
      }
    }

    renderMathResult({
      formula: `${wordA}${wordB ? ` − ${wordB}` : ""}${wordC ? ` ＋ ${wordC}` : ""}`,
      baseWord: wordA,
      neighbors: topIndices,
      queryVector: query,
    });
  } catch (err) {
    $("math-result").classList.add("hidden");
    $("math-error").textContent = err.message;
    $("math-error").classList.remove("hidden");
  }
}

function renderMathResult(res) {
  $("math-result").classList.remove("hidden");
  $("math-headline").textContent = `計算結果: ${res.formula}`;

  // Render neighbors
  $("math-neighbors").innerHTML = res.neighbors.map((n, i) => `
    <div class="neighbor-item">
      <div class="neighbor-left">
        <span class="neighbor-rank">#${i + 1}</span>
        <span class="neighbor-word">${escapeHtml(n.word)}</span>
      </div>
      <div class="neighbor-right">
        <span class="neighbor-score">${n.score.toFixed(4)}</span>
        <button class="neighbor-btn" onclick="setPair('${escapeHtml(res.baseWord)}', '${escapeHtml(n.word)}')">2語比較</button>
      </div>
    </div>
  `).join("");

  // Axes profile of composite vector
  const dim = model.dimension;
  const axes = [];
  for (let d = 0; d < dim; d++) {
    axes.push({
      id: model.dimensions[d].id,
      leftLabel: model.dimensions[d].left,
      rightLabel: model.dimensions[d].right,
      value: res.queryVector[d],
    });
  }

  const sortedByVal = [...axes].sort((a, b) => b.value - a.value);
  const posAxes = sortedByVal.slice(0, 8);
  const negAxes = sortedByVal.slice(-8).reverse();
  const neutralAxes = [...axes].sort((a, b) => Math.abs(a.value) - Math.abs(b.value)).slice(0, 8);

  function renderQueryAxis(a) {
    const width = Math.min(100, Math.abs(a.value) * 120);
    return `
      <div class="axis">
        <div class="axis-title">
          <span>${escapeHtml(a.id)}</span>
          <span style="font-size:12px;color:var(--muted);">${a.value >= 0 ? "+" : ""}${a.value.toFixed(3)}</span>
        </div>
        <div class="axis-label">${escapeHtml(a.leftLabel)} (正) / ${escapeHtml(a.rightLabel)} (負)</div>
        <div class="bar"><i class="${a.value < 0 ? "negative" : ""}" style="width:${width}%"></i></div>
      </div>
    `;
  }

  $("math-pos-axes").innerHTML = posAxes.map(renderQueryAxis).join("");
  $("math-neg-axes").innerHTML = negAxes.map(renderQueryAxis).join("");
  $("math-neutral-axes").innerHTML = neutralAxes.map(renderQueryAxis).join("");

  $("math-all-axes").innerHTML = `
    <table>
      <thead><tr><th>尺度</th><th>左極 (正)</th><th>右極 (負)</th><th>合成値</th></tr></thead>
      <tbody>
        ${axes.map((a) => `
          <tr>
            <td>${escapeHtml(a.id)}</td>
            <td>${escapeHtml(a.leftLabel)}</td>
            <td>${escapeHtml(a.rightLabel)}</td>
            <td>${a.value >= 0 ? "+" : ""}${a.value.toFixed(4)}</td>
          </tr>
        `).join("")}
      </tbody>
    </table>
  `;
}

function setMathPreset(a, b, c) {
  $("math-a").value = a;
  $("math-b").value = b;
  $("math-c").value = c;
  runMath();
}

// ---------------------------------------------------------------------------
// Dynamic Autocomplete across 200,000 words
// ---------------------------------------------------------------------------
function setupAutocomplete() {
  const inputs = [
    $("left"), $("right"),
    $("math-a"), $("math-b"), $("math-c"),
  ].filter(Boolean);

  const suggestList = $("suggest-list");
  let debounceTimer = null;

  function updateSuggestions(val) {
    if (!model || !model.concepts) return;
    const q = val.trim();
    if (!q) {
      suggestList.innerHTML = model.concepts.slice(0, 30).map((w) => `<option value="${escapeHtml(w)}">`).join("");
      return;
    }
    const matches = [];
    const len = model.concepts.length;
    // Fast scan prioritizing startsWith, then includes
    for (let i = 0; i < len && matches.length < 35; i++) {
      const word = model.concepts[i];
      if (word.startsWith(q)) matches.push(word);
    }
    if (matches.length < 25) {
      for (let i = 0; i < len && matches.length < 35; i++) {
        const word = model.concepts[i];
        if (!word.startsWith(q) && word.includes(q)) matches.push(word);
      }
    }
    suggestList.innerHTML = matches.map((w) => `<option value="${escapeHtml(w)}">`).join("");
  }

  inputs.forEach((input) => {
    input.addEventListener("input", (e) => {
      clearTimeout(debounceTimer);
      debounceTimer = setTimeout(() => updateSuggestions(e.target.value), 40);
    });
    input.addEventListener("focus", (e) => {
      updateSuggestions(e.target.value);
    });
  });
}

// ---------------------------------------------------------------------------
// Tabs & Initialization
// ---------------------------------------------------------------------------
function switchTab(tabId) {
  document.querySelectorAll(".tab-btn").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.tab === tabId);
  });
  document.querySelectorAll(".tab-content").forEach((content) => {
    content.classList.toggle("hidden", content.id !== tabId);
  });
}

document.querySelectorAll(".tab-btn").forEach((btn) => {
  btn.addEventListener("click", () => switchTab(btn.dataset.tab));
});

async function loadModel() {
  const [metadataResponse, vectorsResponse] = await Promise.all([
    fetch(`${DATA_ROOT}metadata.json`), fetch(`${DATA_ROOT}vectors.bin`),
  ]);
  if (!metadataResponse.ok || !vectorsResponse.ok) throw new Error("データを読み込めませんでした。");
  const metadata = await metadataResponse.json();
  const bytes = await vectorsResponse.arrayBuffer();
  const vectors = new Float32Array(bytes);
  const dimension = metadata.dimensions.length;
  const count = metadata.concepts.length;
  if (vectors.length !== count * dimension) throw new Error("ベクトルデータのサイズが一致しません。");

  // Precompute mean vector & norms for ultra-fast analogies
  const meanVector = new Float32Array(dimension);
  for (let i = 0; i < count; i++) {
    const offset = i * dimension;
    for (let d = 0; d < dimension; d++) meanVector[d] += vectors[offset + d];
  }
  for (let d = 0; d < dimension; d++) meanVector[d] /= count;

  const rawNorms = new Float32Array(count);
  const centeredNorms = new Float32Array(count);
  for (let i = 0; i < count; i++) {
    const offset = i * dimension;
    let rSum = 0, cSum = 0;
    for (let d = 0; d < dimension; d++) {
      const v = vectors[offset + d];
      const cv = v - meanVector[d];
      rSum += v * v;
      cSum += cv * cv;
    }
    rawNorms[i] = Math.max(Math.sqrt(rSum), 1e-12);
    centeredNorms[i] = Math.max(Math.sqrt(cSum), 1e-12);
  }

  model = {
    concepts: metadata.concepts,
    dimensions: metadata.dimensions,
    vectors,
    dimension,
    count,
    index: Object.fromEntries(metadata.concepts.map((word, i) => [word, i])),
    meanVector,
    rawNorms,
    centeredNorms,
  };

  // Populate presets
  $("presets").innerHTML = PRESETS.map(([left, right]) => `
    <button class="preset" data-left="${escapeHtml(left)}" data-right="${escapeHtml(right)}">${escapeHtml(left)} / ${escapeHtml(right)}</button>
  `).join("");
  document.querySelectorAll("#presets .preset").forEach((button) => {
    button.addEventListener("click", () => setPair(button.dataset.left, button.dataset.right));
  });

  $("math-presets").innerHTML = MATH_PRESETS.map((p) => `
    <button class="preset math-preset-btn" data-a="${escapeHtml(p.a)}" data-b="${escapeHtml(p.b)}" data-c="${escapeHtml(p.c)}">${escapeHtml(p.label)}</button>
  `).join("");
  document.querySelectorAll(".math-preset-btn").forEach((button) => {
    button.addEventListener("click", () => setMathPreset(button.dataset.a, button.dataset.b, button.dataset.c));
  });

  setupAutocomplete();
  $("loading").classList.add("hidden");
  analyze();
  runMath();
}

$("compare").addEventListener("click", analyze);
[$("left"), $("right")].forEach((input) => {
  input.addEventListener("keydown", (event) => { if (event.key === "Enter") analyze(); });
});

$("calc-btn").addEventListener("click", runMath);
[$("math-a"), $("math-b"), $("math-c")].forEach((input) => {
  input.addEventListener("keydown", (event) => { if (event.key === "Enter") runMath(); });
});
$("math-center").addEventListener("change", runMath);

loadModel().catch((error) => {
  $("loading").classList.add("hidden");
  $("error").textContent = error.message;
  $("error").classList.remove("hidden");
});
