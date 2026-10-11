const DATA_ROOT = "data/";
const $ = (id) => document.getElementById(id);
let model = null;

// 2-Word Comparison Presets
const COMPARE_PRESETS = [
  ["春", "秋"], ["高い", "低い"], ["強い", "弱い"], ["明るい", "暗い"],
  ["上", "下"], ["生", "死"], ["成功", "失敗"], ["王", "女王"],
];

// Equalizer (Dimension Manipulator) Analysis Presets
const EQUALIZER_PRESETS = [
  {
    label: "ラーメン (高価・健康的・現代的)",
    base: "ラーメン",
    offsets: { expensive: 3.0, healthy: 3.0, modern: 2.0 },
  },
  {
    label: "ラーメン (混沌・刺激的・非日常)",
    base: "ラーメン",
    offsets: { orderly: -3.0, safe: -3.0, exciting: 3.0 },
  },
  {
    label: "政治家 (誠実・道徳的・穏やか)",
    base: "政治家",
    offsets: { honest: 4.0, moral: 4.0, calm: 3.0 },
  },
  {
    label: "学校 (混沌・活動的・刺激)",
    base: "学校",
    offsets: { orderly: -4.0, safe: -4.0, exciting: 3.0 },
  },
  {
    label: "白湯 (力強さ・現代的・活動的)",
    base: "白湯",
    offsets: { strong: 3.5, modern: 3.0, active: 2.5 },
  },
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
  $("left").value = left;
  $("right").value = right;
  switchTab("tab-compare");
  analyze();
}
window.setPair = setPair;

// ---------------------------------------------------------------------------
// Dimension Equalizer (尺度調整探索) Logic
// ---------------------------------------------------------------------------
const sliderOffsets = new Float32Array(50);
let equalizerDebounceTimer = null;

function renderSliderControls() {
  const container = $("eq-sliders-container");
  container.innerHTML = model.dimensions.map((dim, i) => `
    <div class="slider-group" id="slider-group-${i}">
      <div class="slider-header">
        <span class="slider-name">${escapeHtml(dim.id)}</span>
        <span class="slider-val-badge" id="slider-val-${i}">0.0</span>
      </div>
      <div class="slider-labels">
        <span>${escapeHtml(dim.left)} (+)</span>
        <span>${escapeHtml(dim.right)} (-)</span>
      </div>
      <input type="range" class="slider-control" id="slider-input-${i}"
             min="-3" max="3" step="0.5" value="0"
             data-index="${i}">
    </div>
  `).join("");

  document.querySelectorAll(".slider-control").forEach((input) => {
    input.addEventListener("input", (e) => {
      const idx = parseInt(e.target.dataset.index, 10);
      const val = parseFloat(e.target.value);
      sliderOffsets[idx] = val;
      updateSliderUI(idx, val);
      scheduleEqualizerUpdate();
    });
  });
}

function updateSliderUI(idx, val) {
  const badge = $(`slider-val-${idx}`);
  const group = $(`slider-group-${idx}`);
  if (badge) badge.textContent = (val > 0 ? "+" : "") + val.toFixed(1);
  if (group) group.classList.toggle("is-modified", val !== 0);
}

function resetAllSliders() {
  for (let i = 0; i < model.dimension; i++) {
    sliderOffsets[i] = 0;
    const input = $(`slider-input-${i}`);
    if (input) input.value = "0";
    updateSliderUI(i, 0);
  }
  scheduleEqualizerUpdate();
}

function setEqualizerPreset(preset) {
  $("eq-base").value = preset.base;
  for (let i = 0; i < model.dimension; i++) {
    const dimId = model.dimensions[i].id;
    const val = preset.offsets[dimId] || 0;
    sliderOffsets[i] = val;
    const input = $(`slider-input-${i}`);
    if (input) input.value = val.toString();
    updateSliderUI(i, val);
  }
  runEqualizer();
}

function scheduleEqualizerUpdate() {
  clearTimeout(equalizerDebounceTimer);
  equalizerDebounceTimer = setTimeout(runEqualizer, 60);
}

function runEqualizer() {
  $("eq-error").classList.add("hidden");
  try {
    const baseWord = $("eq-base").value.trim();
    if (!baseWord) throw new Error("基準概念を入力してください。");

    const baseIndex = model.index[baseWord];
    if (baseIndex === undefined) throw new Error(`語彙にありません: ${baseWord}`);

    const useCenter = $("eq-center").checked;
    const dim = model.dimension;
    const offsetBase = baseIndex * dim;

    // Create modified query vector
    const query = new Float32Array(dim);
    let qNormSq = 0;
    for (let d = 0; d < dim; d++) {
      let val = model.vectors[offsetBase + d];
      if (useCenter) val -= model.meanVector[d];
      val += sliderOffsets[d];
      query[d] = val;
      qNormSq += val * val;
    }

    const qNorm = Math.max(Math.sqrt(qNormSq), 1e-12);
    for (let d = 0; d < dim; d++) query[d] /= qNorm;

    // Fast inner product across 200,000 concepts
    const count = model.count;
    const scores = new Float32Array(count);
    for (let i = 0; i < count; i++) {
      const off = i * dim;
      let dot = 0;
      if (useCenter) {
        for (let d = 0; d < dim; d++) {
          dot += (model.vectors[off + d] - model.meanVector[d]) * query[d];
        }
        scores[i] = dot / Math.max(model.centeredNorms[i], 1e-12);
      } else {
        for (let d = 0; d < dim; d++) {
          dot += model.vectors[off + d] * query[d];
        }
        scores[i] = dot / Math.max(model.rawNorms[i], 1e-12);
      }
    }

    // Extract top 10 (skipping base word)
    const topIndices = [];
    for (let i = 0; i < count; i++) {
      if (i === baseIndex) continue;
      const word = model.concepts[i];
      const s = scores[i];
      if (topIndices.length < 10) {
        topIndices.push({ word, score: s });
        if (topIndices.length === 10) topIndices.sort((a, b) => b.score - a.score);
      } else if (s > topIndices[9].score) {
        topIndices[9] = { word, score: s };
        topIndices.sort((a, b) => b.score - a.score);
      }
    }

    renderEqualizerResults(baseWord, topIndices);
  } catch (err) {
    $("eq-error").textContent = err.message;
    $("eq-error").classList.remove("hidden");
  }
}

function renderEqualizerResults(baseWord, neighbors) {
  // Render neighbors
  $("eq-neighbors").innerHTML = neighbors.map((n, i) => `
    <div class="neighbor-item">
      <div class="neighbor-left">
        <span class="neighbor-rank">#${i + 1}</span>
        <span class="neighbor-word">${escapeHtml(n.word)}</span>
      </div>
      <div class="neighbor-right">
        <span class="neighbor-score">${n.score.toFixed(4)}</span>
        <button class="neighbor-btn" onclick="setPair('${escapeHtml(baseWord)}', '${escapeHtml(n.word)}')">比較</button>
      </div>
    </div>
  `).join("");

  // Render modified axes summary
  const modifiedAxes = [];
  const baseIdx = model.index[baseWord];
  const dim = model.dimension;
  for (let d = 0; d < dim; d++) {
    const offset = sliderOffsets[d];
    if (offset !== 0) {
      const baseVal = model.vectors[baseIdx * dim + d];
      modifiedAxes.push({
        id: model.dimensions[d].id,
        leftLabel: model.dimensions[d].left,
        rightLabel: model.dimensions[d].right,
        baseVal,
        offset,
        finalVal: baseVal + offset,
      });
    }
  }

  if (modifiedAxes.length === 0) {
    $("eq-modified-axes").innerHTML = `<p class="muted" style="font-size:13px;">現在調整されている尺度はありません（基準概念の原点最近傍です）。</p>`;
  } else {
    $("eq-modified-axes").innerHTML = `
      <table>
        <thead>
          <tr><th>尺度</th><th>原点値</th><th>調整オフセット</th><th>合成値</th></tr>
        </thead>
        <tbody>
          ${modifiedAxes.map((a) => `
            <tr>
              <td><b>${escapeHtml(a.id)}</b> (${escapeHtml(a.leftLabel)} / ${escapeHtml(a.rightLabel)})</td>
              <td>${a.baseVal.toFixed(2)}</td>
              <td><span style="color:${a.offset > 0 ? '#4263eb' : '#e67700'};font-weight:600;">${a.offset > 0 ? '+' : ''}${a.offset.toFixed(1)}</span></td>
              <td><b>${a.finalVal.toFixed(2)}</b></td>
            </tr>
          `).join("")}
        </tbody>
      </table>
    `;
  }
}

// ---------------------------------------------------------------------------
// Dynamic Autocomplete across 200,000 words
// ---------------------------------------------------------------------------
function setupAutocomplete() {
  const inputs = [$("left"), $("right"), $("eq-base")].filter(Boolean);
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

  // Populate comparison presets
  $("presets").innerHTML = COMPARE_PRESETS.map(([left, right]) => `
    <button class="preset" data-left="${escapeHtml(left)}" data-right="${escapeHtml(right)}">${escapeHtml(left)} / ${escapeHtml(right)}</button>
  `).join("");
  document.querySelectorAll("#presets .preset").forEach((button) => {
    button.addEventListener("click", () => setPair(button.dataset.left, button.dataset.right));
  });

  // Populate equalizer presets
  $("eq-presets").innerHTML = EQUALIZER_PRESETS.map((p, i) => `
    <button class="preset eq-preset-btn" data-preset-idx="${i}">${escapeHtml(p.label)}</button>
  `).join("");
  document.querySelectorAll(".eq-preset-btn").forEach((button) => {
    button.addEventListener("click", (e) => {
      const p = EQUALIZER_PRESETS[parseInt(e.target.dataset.presetIdx, 10)];
      setEqualizerPreset(p);
    });
  });

  renderSliderControls();
  setupAutocomplete();

  $("eq-reset-btn").addEventListener("click", resetAllSliders);
  $("eq-base").addEventListener("keydown", (e) => { if (e.key === "Enter") runEqualizer(); });
  $("eq-center").addEventListener("change", runEqualizer);

  $("loading").classList.add("hidden");
  analyze();
  runEqualizer();
}

$("compare").addEventListener("click", analyze);
[$("left"), $("right")].forEach((input) => {
  input.addEventListener("keydown", (event) => { if (event.key === "Enter") analyze(); });
});

loadModel().catch((error) => {
  $("loading").classList.add("hidden");
  $("error").textContent = error.message;
  $("error").classList.remove("hidden");
});
