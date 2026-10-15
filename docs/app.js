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
        <span>${escapeHtml(dim.right)} (−)</span>
        <span>${escapeHtml(dim.left)} (＋)</span>
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
              <td><b>${escapeHtml(a.id)}</b> (${escapeHtml(a.rightLabel)} [−] / ${escapeHtml(a.leftLabel)} [＋])</td>
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
  setupFlowModule();

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

// ---------------------------------------------------------------------------
// 3. Text Vector Trajectory (Flow) Analysis
// ---------------------------------------------------------------------------

const FLOW_PRESETS = {
  melos: `メロスは激怒した。必ず、かの邪智暴虐の王を除かなければならぬと決意した。メロスには政治がわからぬ。けれども邪悪に対しては、人一倍に敏感であった。シラクスの市に入ったメロスは、王ディオニスが人を殺していると聞き、王城に入り捕らえられた。メロスは妹の結婚式のため三日間の猶予を請い、友のセリヌンティウスを人質として差し出した。
メロスは村へ帰り、雨の降るなか妹の結婚式を無事に挙げた。満天の星の下、祝宴が静かに終わり、安堵と喜びが村を包んだ。翌朝、未明にメロスは友を救うため、雨のやんだ村を出発した。
正午、豪雨による山崩れで川は氾濫し、濁流が逆巻いていた。橋は落ち、激流が吠えていた。メロスは覚悟を決め、濁流へと飛び込み、死闘の末に対岸へ泳ぎ着いた。そこへ山賊が襲いかかったが、メロスは渾身の力で打ち倒し、前へ進んだ。
しかし真夏の太陽が照りつけ、メロスの肉体は疲労困憊し、一歩も歩けなくなった。「もはやこれまで。友よ、すまぬ。私は力尽きた」と叫び、草原に倒れ伏して絶望の涙を流した。
ふと耳を澄ますと、岩の裂け目から清水が湧き出ていた。メロスは泉の水を飲み干し、失われた気力を劇的に回復した。「まだ間に合う、太陽は沈んでいない！」メロスは風のように再び疾走を始めた。
夕陽が西の空を赤く染める中、メロスは最後の力を振り絞り、処刑場へと駆け込んだ。刑台のセリヌンティウスを抱きしめ、二人は涙を流して互いの真の信頼を称え合った。暴君は感動に打たれ、改心して二人を許した。`,

  rashomon: `ある日の暮方の事である。一人の下人が、羅生門の下で雨やみを待っていた。広い門の下には、この男のほかに誰もいない。ただ、所々丹塗の剥げた、大きな円柱に、蟋蟀が一匹とまっている。京都の街は天災や飢饉で荒れ果て、下人も主人から暇を出されたばかりで、途方に暮れていた。
夕闇が迫る中、下人は門の二階の楼上へ登った。そこには身寄りのない死骸が無造作に打ち捨てられていた。腐臭と陰惨な気配が漂う中、下人は一本の火影を見た。見ると、白髪の老婆が死骸の髪の毛を一本ずつ抜いていた。下人は激しい憎悪と正義感を燃やし、太刀を抜いて老婆に躍りかかり、取り押さえた。
老婆は息も絶え絶えに「この髪を抜いて鬘にしようとした。この死んだ女は蛇を魚と偽って売っていたが、生きるために仕方なかった。自分も生きるためにこの悪事をしている」と言い訳した。
それを聞いた下人の心からは憎悪が消え、ある激しい悪の勇気が湧き上がった。「では、己が引剥をしようと恨むまいな。己もそうしなければ餓死する体なのだ」下人は老婆の着物を素早く剥ぎ取り、それを抱えて夜の暗闇の中へと駆け下りて消え去った。`,

  momotaro: `昔々ある所に、お爺さんとお婆さんが住んでいました。お婆さんが川で洗濯をしていると、川上から大きな桃が流れてきました。桃を持ち帰って割ると、中から元気な男の子が生まれ、桃太郎と名付けられました。
桃太郎はすくすくと育ち、強く優しい若者になりました。ある日、村を荒らし人々を苦しめる鬼ヶ島の鬼を退治するため、旅立ちを決意しました。お婆さんが作った美味しい日本一の黍団子を腰につけて出発しました。
道中で犬、猿、雉に出会い、黍団子を分け与えて信頼で結ばれた仲間になりました。一行は舟を漕ぎ出し、険しい海を越えて鬼ヶ島へと渡りました。
鬼ヶ島では激しい戦いが始まりました。犬が鬼の足を噛み、猿が背中を引っ掻き、雉が目を突いて飛び回り、桃太郎は力強く鬼の親分を打ち倒しました。鬼たちは降参し、涙を流してこれまでの悪事を詫び、奪った宝物を全て差し出しました。
桃太郎たちは宝車を引いて誇らしげに村へと帰り、村人たちと平和で幸福な日々を暮らしました。`,

  business: `新規事業の立ち上げ当初、私たちのチームは画期的な技術革新と輝かしい未来への希望に満ち溢れていました。市場の反応も上々で、順風満帆に見えました。
しかし半年後、基幹システムに致命的な欠陥が発覚し、顧客からの抗議と巨額の負債が一気に押し寄せました。資金は底をつきかけ、チーム内には疑心暗鬼と疲労困憊が広がり、まさに絶望の淵に立たされました。
それでも私たちは諦めませんでした。夜を徹して全員で原因を究明し、徹底的な構造改革と誠実な顧客対応を粘り強く続けました。結束を取り戻したチームは、従来の枠組みを打ち破る新しいソリューションを生み出しました。
その結果、危機を完全に乗り越え、前年比三倍の利益を達成する奇跡的なV字回復を果たしました。困難を共に乗り越えた仲間への深い信頼と誇りが、私たちの最大の財産となりました。`
};

const FLOW_STOP_WORDS = new Set([
  "する", "いる", "ある", "なる", "いう", "これ", "それ", "あれ", "よう", "こと",
  "もの", "とき", "ため", "くる", "いく", "見る", "思う", "言う", "できる",
  "し", "い", "あり", "なり", "あっ", "いっ", "き", "来", "行っ", "思っ", "私", "おれ", "自分",
  "の", "に", "は", "を", "が", "と", "も", "へ", "で", "や", "から", "まで",
  "より", "だけ", "しか", "ほど", "など", "そして", "しかし", "だが", "また",
  "その", "この", "あの", "どの", "ここ", "そこ", "あそこ", "彼", "彼女"
]);

const FLOW_DEFAULT_DIMS = ["hopeful", "safe", "fast", "moral", "calm"];
const FLOW_DIM_PALETTE = {
  hopeful:   { color: "#2ecc71", label: "希望 ↔ 絶望" },
  safe:      { color: "#3498db", label: "安全 ↔ 危機" },
  fast:      { color: "#e74c3c", label: "速さ・疾走 ↔ 静止" },
  moral:     { color: "#9b59b6", label: "善・道徳 ↔ 悪・背徳" },
  calm:      { color: "#f39c12", label: "平穏 ↔ 緊迫" },
  pleasant:  { color: "#1abc9c", label: "快い ↔ 不快" },
  strong:    { color: "#34495e", label: "強い・力 ↔ 弱い" },
  active:    { color: "#e67e22", label: "活動的 ↔ 静的" },
};

let flowActiveDims = new Set(FLOW_DEFAULT_DIMS);
let currentFlowResult = null;

function setupFlowModule() {
  // Preset buttons
  document.querySelectorAll(".flow-preset-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      const presetKey = btn.dataset.preset;
      if (FLOW_PRESETS[presetKey]) {
        $("flow-input").value = FLOW_PRESETS[presetKey];
        runFlowAnalysis();
      }
    });
  });

  // Action buttons
  $("flow-run-btn").addEventListener("click", runFlowAnalysis);
  $("flow-center").addEventListener("change", () => {
    if (currentFlowResult) {
      renderFlowGraphAndHighlights();
    }
  });

  // Render dimension selector toggles
  renderFlowDimSelectors();

  // Set default initial text to Melos
  if (!$("flow-input").value) {
    $("flow-input").value = FLOW_PRESETS.melos;
  }
}

function renderFlowDimSelectors() {
  const container = $("flow-dim-selectors");
  if (!container) return;
  container.innerHTML = Object.entries(FLOW_DIM_PALETTE).map(([id, info]) => {
    const isActive = flowActiveDims.has(id);
    return `
      <button class="dim-toggle-btn ${isActive ? 'active' : ''}" data-dim="${id}">
        <span class="dim-color-dot" style="background:${info.color}"></span>
        <span>${escapeHtml(info.label)}</span>
      </button>
    `;
  }).join("");

  container.querySelectorAll(".dim-toggle-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      const dimId = btn.dataset.dim;
      if (flowActiveDims.has(dimId)) {
        if (flowActiveDims.size > 1) {
          flowActiveDims.delete(dimId);
          btn.classList.remove("active");
        }
      } else {
        flowActiveDims.add(dimId);
        btn.classList.add("active");
      }
      if (currentFlowResult) {
        renderFlowGraphAndHighlights();
      }
    });
  });
}

function tokenizeFlowText(text) {
  if (!text || !model) return [];
  let rawTokens = [];
  if (typeof Intl !== "undefined" && Intl.Segmenter) {
    const segmenter = new Intl.Segmenter("ja", { granularity: "word" });
    for (const { segment, isWordLike } of segmenter.segment(text)) {
      if (isWordLike) {
        const clean = segment.trim();
        if (clean) rawTokens.push(clean);
      }
    }
  } else {
    const matched = text.match(/[\p{Script=Han}\p{Script=Hiragana}\p{Script=Katakana}\w]+/gu);
    if (matched) rawTokens = matched.map((t) => t.trim()).filter(Boolean);
  }

  const results = [];
  for (let i = 0; i < rawTokens.length; i++) {
    const t1 = rawTokens[i];
    if (FLOW_STOP_WORDS.has(t1) || t1.length <= 1) continue;

    // Check 2-word compound if available in 200k vocab
    if (i + 1 < rawTokens.length) {
      const compound = t1 + rawTokens[i + 1];
      if (model.index[compound] !== undefined) {
        results.push({ word: compound, index: model.index[compound] });
        i++;
        continue;
      }
    }

    if (model.index[t1] !== undefined) {
      results.push({ word: t1, index: model.index[t1] });
    }
  }
  return results;
}

function computeTrajectory(wordItems, applyCenter) {
  const n = wordItems.length;
  if (n === 0) return { points: [], words: [] };

  const winSize = Math.max(4, Math.min(30, Math.floor(n / 4)));
  const step = Math.max(1, Math.floor(winSize / 3));
  const dimCount = model.dimension;
  const points = [];

  for (let start = 0; start <= n - winSize; start += step) {
    const end = start + winSize;
    const winItems = wordItems.slice(start, end);
    const progress = Math.round(((start + end) / (2 * n)) * 100);

    const avgVec = new Float32Array(dimCount);
    for (const item of winItems) {
      const offset = item.index * dimCount;
      for (let d = 0; d < dimCount; d++) {
        let v = model.vectors[offset + d];
        if (applyCenter) v -= model.meanVector[d];
        avgVec[d] += v;
      }
    }
    for (let d = 0; d < dimCount; d++) {
      avgVec[d] /= winItems.length;
    }

    points.push({
      progress,
      start,
      end,
      vector: avgVec,
      words: winItems.map((it) => it.word),
    });
  }

  if (points.length > 0 && points[points.length - 1].progress < 95 && n >= winSize) {
    const winItems = wordItems.slice(n - winSize, n);
    const avgVec = new Float32Array(dimCount);
    for (const item of winItems) {
      const offset = item.index * dimCount;
      for (let d = 0; d < dimCount; d++) {
        let v = model.vectors[offset + d];
        if (applyCenter) v -= model.meanVector[d];
        avgVec[d] += v;
      }
    }
    for (let d = 0; d < dimCount; d++) avgVec[d] /= winItems.length;
    points.push({
      progress: 100,
      start: n - winSize,
      end: n,
      vector: avgVec,
      words: winItems.map((it) => it.word),
    });
  }

  return { points, words: wordItems };
}

function runFlowAnalysis() {
  const errorEl = $("flow-error");
  errorEl.classList.add("hidden");

  const rawText = $("flow-input").value.trim();
  if (!rawText) {
    errorEl.textContent = "解析するテキストを入力してください。";
    errorEl.classList.remove("hidden");
    return;
  }

  const wordItems = tokenizeFlowText(rawText);
  if (wordItems.length < 5) {
    errorEl.textContent = `認識された単語が少なすぎます（${wordItems.length}語）。より長い文章を入力してください。`;
    errorEl.classList.remove("hidden");
    return;
  }

  // Update metrics
  $("flow-char-count").textContent = rawText.length.toLocaleString();
  $("flow-concept-count").textContent = wordItems.length.toLocaleString();
  const approxTotalWords = rawText.length / 2.5;
  const coveragePct = Math.min(100, Math.round((wordItems.length / approxTotalWords) * 100));
  $("flow-coverage").textContent = `${coveragePct}%`;

  currentFlowResult = {
    text: rawText,
    wordItems: wordItems
  };

  renderFlowGraphAndHighlights();
  renderWordTimeline(wordItems);
  $("flow-result").classList.remove("hidden");
}

function renderFlowGraphAndHighlights() {
  if (!currentFlowResult) return;
  const applyCenter = $("flow-center").checked;
  const { points, words } = computeTrajectory(currentFlowResult.wordItems, applyCenter);

  if (points.length < 2) {
    $("flow-error").textContent = "文章が短すぎるため十分な軌跡データポイントを生成できませんでした。";
    $("flow-error").classList.remove("hidden");
    return;
  }

  // Map dimension id to index
  const dimIdToIndex = Object.fromEntries(model.dimensions.map((d, i) => [d.id, i]));

  // Calculate min and max y for scaling
  let minY = 0, maxY = 0;
  points.forEach((pt) => {
    flowActiveDims.forEach((dimId) => {
      const idx = dimIdToIndex[dimId];
      if (idx !== undefined) {
        const val = pt.vector[idx];
        if (val < minY) minY = val;
        if (val > maxY) maxY = val;
      }
    });
  });

  // Add padding to y bounds
  const span = Math.max(0.5, maxY - minY);
  minY -= span * 0.15;
  maxY += span * 0.15;

  // SVG dimensions
  const svgWidth = 860;
  const svgHeight = 360;
  const padLeft = 45;
  const padRight = 30;
  const padTop = 30;
  const padBottom = 40;
  const plotW = svgWidth - padLeft - padRight;
  const plotH = svgHeight - padTop - padBottom;

  const getX = (pct) => padLeft + (pct / 100) * plotW;
  const getY = (val) => padTop + plotH - ((val - minY) / (maxY - minY)) * plotH;

  // Build SVG XML
  let svg = `<svg viewBox="0 0 ${svgWidth} ${svgHeight}" xmlns="http://www.w3.org/2000/svg" style="background:#fafbfc; border-radius:8px;">`;

  // Horizontal Grid Lines & Y Labels
  const gridSteps = 5;
  for (let i = 0; i <= gridSteps; i++) {
    const val = minY + (i / gridSteps) * (maxY - minY);
    const yPos = getY(val);
    const isZero = Math.abs(val) < (maxY - minY) / (gridSteps * 2);
    svg += `
      <line x1="${padLeft}" y1="${yPos}" x2="${svgWidth - padRight}" y2="${yPos}" 
            stroke="${isZero ? '#adb5bd' : '#e9ecef'}" stroke-width="${isZero ? 1.5 : 1}" 
            stroke-dasharray="${isZero ? '4,3' : 'none'}" />
      <text x="${padLeft - 8}" y="${yPos + 4}" fill="#868e96" font-size="11" text-anchor="end" font-family="sans-serif">${val > 0 ? '+' : ''}${val.toFixed(2)}</text>
    `;
  }

  // Vertical Grid Lines & Progress Labels
  for (let pct = 0; pct <= 100; pct += 25) {
    const xPos = getX(pct);
    svg += `
      <line x1="${xPos}" y1="${padTop}" x2="${xPos}" y2="${svgHeight - padBottom}" stroke="#f1f3f5" stroke-width="1" />
      <text x="${xPos}" y="${svgHeight - padBottom + 18}" fill="#868e96" font-size="11" text-anchor="middle" font-family="sans-serif">${pct}%</text>
    `;
  }
  svg += `<text x="${svgWidth / 2}" y="${svgHeight - 10}" fill="#495057" font-size="12" font-weight="600" text-anchor="middle" font-family="sans-serif">文章の進行度 (%)</text>`;

  // Draw Line Paths for Each Active Dimension
  const renderedLines = [];
  flowActiveDims.forEach((dimId) => {
    const idx = dimIdToIndex[dimId];
    if (idx === undefined) return;
    const info = FLOW_DIM_PALETTE[dimId] || { color: "#4263eb", label: dimId };

    const ptsStr = points.map((pt) => `${getX(pt.progress).toFixed(1)},${getY(pt.vector[idx]).toFixed(1)}`).join(" L ");
    const pathD = `M ${ptsStr}`;
    svg += `
      <path d="${pathD}" fill="none" stroke="${info.color}" stroke-width="2.6" stroke-linejoin="round" stroke-linecap="round" opacity="0.9" />
    `;

    // Data points
    points.forEach((pt) => {
      svg += `
        <circle cx="${getX(pt.progress).toFixed(1)}" cy="${getY(pt.vector[idx]).toFixed(1)}" r="3.2" fill="${info.color}" stroke="#ffffff" stroke-width="1" />
      `;
    });
  });

  // Crosshair group (hidden by default)
  svg += `<line id="flow-crosshair" x1="0" y1="${padTop}" x2="0" y2="${svgHeight - padBottom}" stroke="#343a40" stroke-width="1.2" stroke-dasharray="3,3" style="display:none;" />`;
  svg += `</svg>`;

  $("flow-chart-svg-wrapper").innerHTML = svg;

  // Attach interactive hover listener to the container
  setupChartHoverInteraction(points, getX, getY, padLeft, padRight, svgWidth);

  // Render highlights
  renderHighlights(points, dimIdToIndex);
}

function setupChartHoverInteraction(points, getX, getY, padLeft, padRight, svgWidth) {
  const container = $("flow-chart-container");
  const tooltip = $("flow-tooltip");
  const crosshair = $("flow-crosshair");

  container.onmousemove = (e) => {
    const rect = container.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const relX = (mouseX / rect.width) * svgWidth;

    if (relX < padLeft || relX > svgWidth - padRight) {
      tooltip.classList.add("hidden");
      if (crosshair) crosshair.style.display = "none";
      return;
    }

    // Find closest point by progress
    let bestPt = points[0];
    let bestDist = 99999;
    points.forEach((pt) => {
      const px = getX(pt.progress);
      const dist = Math.abs(px - relX);
      if (dist < bestDist) {
        bestDist = dist;
        bestPt = pt;
      }
    });

    const px = getX(bestPt.progress);
    if (crosshair) {
      crosshair.setAttribute("x1", px);
      crosshair.setAttribute("x2", px);
      crosshair.style.display = "block";
    }

    // Tooltip HTML
    const dimIdToIndex = Object.fromEntries(model.dimensions.map((d, i) => [d.id, i]));
    let rowsHtml = "";
    flowActiveDims.forEach((dimId) => {
      const idx = dimIdToIndex[dimId];
      if (idx !== undefined) {
        const info = FLOW_DIM_PALETTE[dimId];
        const val = bestPt.vector[idx];
        rowsHtml += `
          <div class="flow-tooltip-row">
            <span style="display:inline-flex; align-items:center; gap:5px;">
              <span style="width:8px; height:8px; border-radius:50%; background:${info.color};"></span>
              ${escapeHtml(info.label)}:
            </span>
            <b>${val > 0 ? '+' : ''}${val.toFixed(2)}</b>
          </div>
        `;
      }
    });

    const sampleWords = bestPt.words.slice(0, 8).join("、 ");
    tooltip.innerHTML = `
      <div class="flow-tooltip-title">進行度: ${bestPt.progress}% 付近</div>
      ${rowsHtml}
      <div class="flow-tooltip-words">代表語: ${escapeHtml(sampleWords)}</div>
    `;

    // Position tooltip
    const leftPercent = (px / svgWidth) * 100;
    tooltip.style.left = `${leftPercent}%`;
    tooltip.style.top = "18%";
    tooltip.classList.remove("hidden");
  };

  container.onmouseleave = () => {
    tooltip.classList.add("hidden");
    if (crosshair) crosshair.style.display = "none";
  };
}

function renderHighlights(points, dimIdToIndex) {
  const container = $("flow-highlights");
  if (!container) return;

  const highlights = [];

  // 1. Hopeful Peak
  if (dimIdToIndex["hopeful"] !== undefined) {
    const idx = dimIdToIndex["hopeful"];
    let maxPt = points[0], minPt = points[0];
    points.forEach((pt) => {
      if (pt.vector[idx] > maxPt.vector[idx]) maxPt = pt;
      if (pt.vector[idx] < minPt.vector[idx]) minPt = pt;
    });

    highlights.push({
      badge: "希望のピーク",
      bg: "#e6fcf5",
      color: "#087f5b",
      title: `希望の高まり (${maxPt.vector[idx] > 0 ? '+' : ''}${maxPt.vector[idx].toFixed(2)})`,
      meta: `物語の ${maxPt.progress}% 付近`,
      words: maxPt.words.slice(0, 6),
    });

    highlights.push({
      badge: "絶望の谷底",
      bg: "#fff5f5",
      color: "#c92a2a",
      title: `最大の危機・絶望 (${minPt.vector[idx] > 0 ? '+' : ''}${minPt.vector[idx].toFixed(2)})`,
      meta: `物語の ${minPt.progress}% 付近`,
      words: minPt.words.slice(0, 6),
    });
  }

  // 2. Fast / Tension Peak
  if (dimIdToIndex["fast"] !== undefined) {
    const idx = dimIdToIndex["fast"];
    let maxPt = points[0];
    points.forEach((pt) => {
      if (pt.vector[idx] > maxPt.vector[idx]) maxPt = pt;
    });
    highlights.push({
      badge: "疾走・クライマックス",
      bg: "#fff0f6",
      color: "#a61e4d",
      title: `速さ・緊迫の頂点 (${maxPt.vector[idx] > 0 ? '+' : ''}${maxPt.vector[idx].toFixed(2)})`,
      meta: `物語の ${maxPt.progress}% 付近`,
      words: maxPt.words.slice(0, 6),
    });
  }

  // 3. Calm Peak (Peace / Relief)
  if (dimIdToIndex["calm"] !== undefined) {
    const idx = dimIdToIndex["calm"];
    let maxPt = points[0];
    points.forEach((pt) => {
      if (pt.vector[idx] > maxPt.vector[idx]) maxPt = pt;
    });
    highlights.push({
      badge: "安堵・静寂",
      bg: "#fff9db",
      color: "#f08c00",
      title: `平穏のピーク (${maxPt.vector[idx] > 0 ? '+' : ''}${maxPt.vector[idx].toFixed(2)})`,
      meta: `物語の ${maxPt.progress}% 付近`,
      words: maxPt.words.slice(0, 6),
    });
  }

  container.innerHTML = highlights.map((h) => `
    <div class="flow-highlight-card">
      <span class="flow-highlight-badge" style="background:${h.bg}; color:${h.color};">${escapeHtml(h.badge)}</span>
      <div class="flow-highlight-title">${escapeHtml(h.title)}</div>
      <div class="flow-highlight-meta">${escapeHtml(h.meta)}</div>
      <div class="flow-highlight-words">
        ${h.words.map((w) => `<span class="word-tag">${escapeHtml(w)}</span>`).join("")}
      </div>
    </div>
  `).join("");
}

function renderWordTimeline(wordItems) {
  const container = $("flow-word-timeline");
  if (!container) return;
  container.innerHTML = wordItems.map((item, i) => `
    <span class="timeline-word-item">
      <span>${escapeHtml(item.word)}</span>
      <small>#${i + 1}</small>
    </span>
  `).join("");
}

loadModel().catch((error) => {
  $("loading").classList.add("hidden");
  $("error").textContent = error.message;
  $("error").classList.remove("hidden");
});
