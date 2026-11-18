// UI: model loading with a progress bar (download once, then Cache Storage), text input, heatmap card.
const $ = (id) => document.getElementById(id);
const worker = new Worker(new URL("./worker.js", import.meta.url), { type: "module" });
const SAMPLES = {
  human: "It was the best of times, it was the worst of times, it was the age of wisdom, it was the age of foolishness, it was the epoch of belief, it was the epoch of incredulity, it was the season of Light, it was the season of Darkness, it was the spring of hope, it was the winter of despair. We had everything before us, we had nothing before us, we were all going direct to Heaven, we were all going direct the other way. In short, the period was so far like the present period, that some of its noisiest authorities insisted on its being received, for good or for evil, in the superlative degree of comparison only.",
  ai: "Artificial intelligence has become an increasingly important topic in modern society, influencing a wide range of industries and aspects of daily life. In today's rapidly evolving technological landscape, it is essential to understand both the opportunities and the challenges that AI presents. On the one hand, AI-powered tools can enhance productivity, streamline complex workflows, and unlock new insights from large datasets. On the other hand, they raise important questions about privacy, fairness, and accountability. Ultimately, a balanced approach that leverages the benefits of AI while addressing its risks will be crucial for building a more equitable and sustainable future.",
  mixed: "I grew up two streets from the harbour, and every morning my grandfather would carry the same battered thermos down to the quay, sit on an upturned crate, and say nothing at all to anyone until the first boat came in. I never understood it as a child. Now, when I think about that ritual, I realize that it represented something deeper: a quiet commitment to routine in the face of uncertainty. It highlights the importance of finding stability in a changing world, and it serves as a reminder that small, consistent habits can foster a profound sense of connection, resilience, and purpose. His boots were always wet by seven.",
};
let ready = false, nextId = 1, pending = null;

// ---- colors: same scale as the report heatmaps
const PROB = [[0, [251, 191, 36, 0]], [0.35, [251, 191, 36, 0.22]], [0.65, [251, 113, 133, 0.5]], [1, [225, 29, 72, 0.82]]];
function colorOf(p) {
  p = Math.min(Math.max(p, 0), 1);
  for (let i = 0; i < PROB.length - 1; i++) {
    const [a, ca] = PROB[i], [b, cb] = PROB[i + 1];
    if (p <= b) { const t = (p - a) / (b - a), c = ca.map((v, k) => v + t * (cb[k] - v)); return `rgba(${c[0].toFixed(0)},${c[1].toFixed(0)},${c[2].toFixed(0)},${c[3].toFixed(2)})`; }
  }
  return "rgba(225,29,72,.82)";
}

// ---- loading bar
const fmtMB = (b) => (b / 1e6).toFixed(0);
function setBar(frac, indeterminate) {
  const bar = $("loader").querySelector(".bar"); bar.classList.toggle("indet", !!indeterminate);
  $("loader-fill").style.width = Math.round(100 * frac) + "%"; bar.setAttribute("aria-valuenow", Math.round(100 * frac));
}
function say(msg, warn) { $("msg").textContent = msg || ""; $("msg").style.color = warn ? "#9a3412" : "#44403c"; }

worker.onmessage = (e) => {
  const m = e.data;
  if (m.type === "status") $("loader-title").textContent = m.message;
  else if (m.type === "progress") {
    if (m.phase === "download") { setBar(m.loaded / m.total * 0.95); $("loader-detail").textContent = `${fmtMB(m.loaded)} of ${fmtMB(m.total)} MB`; }
    else { setBar(1, true); $("loader-detail").textContent = "initializing"; }
  } else if (m.type === "ready") {
    ready = true; setBar(1); $("loader").classList.add("done"); $("loader-title").textContent = m.cached ? "Model ready (loaded from your browser cache)" : "Model ready (downloaded and stored in your browser)";
    $("loader-detail").textContent = m.threads > 1 ? `${m.threads} threads` : "1 thread";
    $("runinfo").textContent = `Model version ${m.version}, ${m.threads} CPU thread${m.threads > 1 ? "s" : ""}${m.isolated ? "" : " (cross-origin isolation is off, so this runs on one thread and is slower)"}.`;
    refreshButtons(); if (pending) { const t = pending; pending = null; run(t); }
  } else if (m.type === "analyzing") say(`Analyzing: window ${m.done} of ${m.total}`);
  else if (m.type === "result") show(m);
  else if (m.type === "warn") say(m.message, true);
  else if (m.type === "cleared") { say("The downloaded model was removed from this browser. Reload the page to download it again."); ready = false; refreshButtons(); }
  else if (m.type === "error") {
    $("analyze").disabled = false; $("result").querySelector(".hmx")?.classList.remove("busy");
    if (m.where === "load") { $("loader-title").textContent = "The model could not be loaded"; $("loader-detail").textContent = ""; setBar(0); say(m.message + (location.protocol === "file:" ? " Open the page through a web server; browsers do not allow this from a file." : ""), true); }
    else say(m.message, true);
  }
};
worker.onerror = (e) => { say("The worker failed: " + (e.message || "unknown error"), true); };

// ---- analysis
function refreshButtons() {}
function run(text) {
  say(ready ? "Analyzing" : "Waiting for the model to finish loading");
  $("analyze").disabled = true; $("result").querySelector(".hmx")?.classList.add("busy");
  worker.postMessage({ type: "analyze", id: nextId++, text });
}
function onAnalyze() {
  const text = $("input").value.trim();
  if (!text) { say("Paste or type some text first.", true); return; }
  if (!ready) { pending = text; say("Waiting for the model to finish loading, then analyzing"); return; }
  run(text);
}

function verdict(p) { return p < 0.2 ? "Looks human-written" : p < 0.6 ? "Mixed or uncertain" : "Looks AI-generated"; }

function show(m) {
  $("analyze").disabled = false; say(m.truncated ? "The text was cut to its first 30,000 characters." : (m.tokens < 60 ? "This text is short; scores on short text are unreliable." : ""), m.tokens < 60 || m.truncated);
  const card = document.createElement("div"); card.className = "hmx";
  const pct = Math.round(100 * m.docProb);
  card.innerHTML = `<div class="top"><div><div class="ttl">Document score</div><div class="verdict"></div></div><div class="score">${pct}%<small>AI-generated</small></div></div>
    <div class="meter"><i style="left:${(100 * m.docProb).toFixed(1)}%"></i></div><div class="ends"><span>human</span><span>AI</span></div>
    <div class="map"></div><div class="maplab"></div><div class="txt"></div>
    <div class="leg"><span>P(AI) 0%</span><span class="bar2"></span><span>100%</span><span>&middot; hover a word for its probability</span></div>
    <div class="foot"></div>`;
  card.querySelector(".verdict").textContent = verdict(m.docProb);
  const words = m.words, meanW = words.length ? words.reduce((a, w) => a + w[2], 0) / words.length : 0;
  const map = card.querySelector(".map"), k = Math.min(64, words.length);
  for (let i = 0; i < k; i++) {
    const a = Math.floor(i * words.length / k), b = Math.floor((i + 1) * words.length / k); let s = 0; for (let j = a; j < b; j++) s += words[j][2];
    const v = b > a ? s / (b - a) : 0, el = document.createElement("b"); if (v > 0.04) el.style.background = colorOf(v); map.appendChild(el);
  }
  card.querySelector(".maplab").textContent = `Document map: ${m.tokens} tokens, mean word probability ${(100 * meanW).toFixed(0)}%`;
  const txt = card.querySelector(".txt"); let pos = 0, last = null;
  for (const [s, e, v] of words) {            // plain spaces after a word belong to its chip so neighbouring highlights form a ribbon
    const gap = m.text.slice(pos, s);
    if (last && gap && gap.trim() === "" && !gap.includes("\n")) last.textContent += gap; else if (gap) txt.appendChild(document.createTextNode(gap));
    const sp = document.createElement("span"); sp.className = "w"; sp.style.setProperty("--c", colorOf(v)); sp.title = `P(AI) = ${(100 * v).toFixed(0)}%`; sp.textContent = m.text.slice(s, e);
    txt.appendChild(sp); last = sp; pos = e;
  }
  if (pos < m.text.length) txt.appendChild(document.createTextNode(m.text.slice(pos)));
  card.querySelector(".foot").textContent = `${m.tokens} tokens in ${m.windows} window${m.windows > 1 ? "s" : ""}, analyzed in ${(m.ms / 1000).toFixed(1)} s on this device.`;
  $("result").replaceChildren(card);
  window.__lastResult = m;   // for tests
}

// ---- wiring
$("analyze").addEventListener("click", onAnalyze);
$("clear").addEventListener("click", () => { $("input").value = ""; $("result").replaceChildren(); say(""); updateCount(); $("input").focus(); });
document.querySelectorAll("[data-sample]").forEach((b) => b.addEventListener("click", () => { $("input").value = SAMPLES[b.dataset.sample]; updateCount(); }));
$("input").addEventListener("keydown", (e) => { if ((e.ctrlKey || e.metaKey) && e.key === "Enter") onAnalyze(); });
$("input").addEventListener("input", updateCount);
$("clear-cache").addEventListener("click", () => worker.postMessage({ type: "clear" }));
function updateCount() { const t = $("input").value.trim(); $("count").textContent = t ? `${t.split(/\s+/).length} words` : ""; }
$("analyze").disabled = false;

if (!window.isSecureContext) say("This page must be served over https (or localhost) so the model can be cached.", true);
navigator.storage?.persist?.();
worker.postMessage({ type: "init", base: new URL("./model/", location.href).href });
window.__app = { SAMPLES };
