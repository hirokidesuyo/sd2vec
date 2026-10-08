const DATA_ROOT = "data/";
const $ = (id) => document.getElementById(id);
let model = null;

const PRESETS = [
  ["春", "秋"], ["高い", "低い"], ["強い", "弱い"], ["明るい", "暗い"],
  ["上", "下"], ["生", "死"], ["成功", "失敗"], ["王", "女王"],
];

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, (char) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  })[char]);
}

function normalize(vector) {
  let sum = 0;
  for (const value of vector) sum += value * value;
  return Math.sqrt(sum);
}

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
  $("left").value = left; $("right").value = right; analyze();
}

async function loadModel() {
  const [metadataResponse, vectorsResponse] = await Promise.all([
    fetch(`${DATA_ROOT}metadata.json`), fetch(`${DATA_ROOT}vectors.bin`),
  ]);
  if (!metadataResponse.ok || !vectorsResponse.ok) throw new Error("データを読み込めませんでした。");
  const metadata = await metadataResponse.json();
  const bytes = await vectorsResponse.arrayBuffer();
  const vectors = new Float32Array(bytes);
  const dimension = metadata.dimensions.length;
  if (vectors.length !== metadata.concepts.length * dimension) throw new Error("ベクトルデータのサイズが一致しません。");
  model = { concepts: metadata.concepts, dimensions: metadata.dimensions, vectors, dimension, index: Object.fromEntries(metadata.concepts.map((word, i) => [word, i])) };
  $("words").innerHTML = metadata.concepts.map((word) => `<option value="${escapeHtml(word)}">`).join("");
  $("presets").innerHTML = PRESETS.map(([left, right]) => `<button class="preset" data-left="${escapeHtml(left)}" data-right="${escapeHtml(right)}">${escapeHtml(left)} / ${escapeHtml(right)}</button>`).join("");
  document.querySelectorAll(".preset").forEach((button) => button.addEventListener("click", () => setPair(button.dataset.left, button.dataset.right)));
  $("loading").classList.add("hidden");
  analyze();
}

$("compare").addEventListener("click", analyze);
[$("left"), $("right")].forEach((input) => input.addEventListener("keydown", (event) => { if (event.key === "Enter") analyze(); }));
loadModel().catch((error) => { $("loading").classList.add("hidden"); $("error").textContent = error.message; $("error").classList.remove("hidden"); });
