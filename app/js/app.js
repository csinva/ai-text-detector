// Live AI-text detector UI. The heat map is drawn on a layer directly behind the textarea, so the words you type are the words that light up.
// The text is re-read a moment after you pause (or after a few finished words, or at once on paste); words waiting for the model shimmer and a light sweeps across them as they are scored.
const $ = (id) => document.getElementById(id);
const worker = new Worker(new URL("./worker.js", import.meta.url), { type: "module" });
const ta = $("input"), mirror = $("mirror"), tip = $("tip"), MIN_WORDS = 6;
const SAMPLES = {
  human: "Call me Ishmael. Some years ago, never mind how long precisely, having little or no money in my purse, and nothing particular to interest me on shore, I thought I would sail about a little and see the watery part of the world. It is a way I have of driving off the spleen and regulating the circulation. Whenever I find myself growing grim about the mouth, whenever it is a damp, drizzly November in my soul, then I account it high time to get to sea as soon as I can.",
  ai: "Artificial intelligence has become an increasingly important topic in modern society, influencing a wide range of industries and aspects of daily life. In today's rapidly evolving technological landscape, it is essential to understand both the opportunities and the challenges that AI presents. On the one hand, AI-powered tools can enhance productivity, streamline complex workflows, and unlock new insights from large datasets. On the other hand, they raise important questions about privacy, fairness, and accountability. Ultimately, a balanced approach that leverages the benefits of AI while addressing its risks will be crucial for building a more equitable and sustainable future.",
  mixed: "Call me Ishmael. Some years ago, never mind how long precisely, having little or no money in my purse, and nothing particular to interest me on shore, I thought I would sail about a little and see the watery part of the world. It is a way I have of driving off the spleen and regulating the circulation. Whenever I find myself growing grim about the mouth, whenever it is a damp, drizzly November in my soul, then I account it high time to get to sea as soon as I can. This decision reflects a broader commitment to exploration and personal growth. It highlights the importance of embracing new experiences in order to foster resilience, creativity, and a deeper sense of purpose. Ultimately, the sea serves as a powerful reminder that meaningful journeys can unlock new perspectives and inspire lasting change.",
};
// state: `snap` is the text the model last scored, `probs` maps a word's start offset in `snap` to its P(AI); unchanged words keep their colour while you edit around them
const S = { ready: false, sentId: 0, doneId: 0, snap: "", probs: new Map(), timer: null, timerOn: false, lastRun: 0, analyzedWc: 0, lastLen: 0, wantRun: false, shown: new Map(), reveal: false };

// ---- colour scale (the same as the report heatmaps)
const PROB = [[0, [230, 159, 0, 0]], [0.35, [230, 159, 0, 0.3]], [0.7, [230, 159, 0, 0.62]], [1, [213, 94, 0, 0.85]]];
function colorOf(p) {
  p = Math.min(Math.max(p, 0), 1);
  for (let i = 0; i < PROB.length - 1; i++) {
    const [a, ca] = PROB[i], [b, cb] = PROB[i + 1];
    if (p <= b) { const t = (p - a) / (b - a), c = ca.map((v, k) => v + t * (cb[k] - v)); return `rgba(${c[0].toFixed(0)},${c[1].toFixed(0)},${c[2].toFixed(0)},${c[3].toFixed(2)})`; }
  }
  return "rgba(213,94,0,.85)";
}
const esc = (s) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
const wordsOf = (t) => Array.from(t.matchAll(/\S+/g), (m) => [m.index, m.index + m[0].length]);

// ---- the heat layer
function lookup(t) {
  const old = S.snap; if (!old || !S.probs.size) return () => undefined;
  const m = Math.min(old.length, t.length); let pre = 0; while (pre < m && old.charCodeAt(pre) === t.charCodeAt(pre)) pre++;
  let suf = 0; while (suf < m - pre && old.charCodeAt(old.length - 1 - suf) === t.charCodeAt(t.length - 1 - suf)) suf++;
  const delta = t.length - old.length;
  return (s, e) => (e <= pre ? S.probs.get(s) : s >= t.length - suf ? S.probs.get(s - delta) : undefined);
}
function render() {
  const t = ta.value, words = wordsOf(t), look = lookup(t), live = S.ready && words.length >= MIN_WORDS, shown = new Map();
  const items = words.map(([s, e], k) => {
    const nxt = k + 1 < words.length ? words[k + 1][0] : -1, end = nxt >= 0 && /^[ \t]*$/.test(t.slice(e, nxt)) ? nxt : e;   // spaces belong to the chip, so highlights form a ribbon
    const p = look(s, e); return { s, end, p, c: p === undefined ? null : colorOf(p) };
  });
  const fresh = S.reveal ? items.filter((w) => w.c && S.shown.get(w.s) !== w.c).length : 0, step = Math.max(3, Math.min(34, 950 / Math.max(fresh, 1)));
  let html = "", pos = 0, nf = 0;
  items.forEach((w, k) => {
    if (w.s > pos) html += esc(t.slice(pos, w.s));
    const txt = esc(t.slice(w.s, w.end));
    if (w.c) {
      const isNew = S.reveal && S.shown.get(w.s) !== w.c;
      html += `<span class="w${isNew ? " rv" : ""}" data-p="${w.p.toFixed(3)}" style="--c:${w.c}${isNew ? `;--d:${Math.round(nf++ * step)}ms` : ""}">${txt}</span>`; shown.set(w.s, w.c);
    } else html += live ? `<span class="w pend" style="--i:${k % 24}">${txt}</span>` : `<span class="w">${txt}</span>`;
    pos = w.end;
  });
  html += esc(t.slice(pos)); if (t.endsWith("\n")) html += " ";
  mirror.innerHTML = html; S.shown = shown; S.reveal = false;
}
function autosize() { ta.style.height = "auto"; ta.style.height = Math.max(ta.scrollHeight, 240) + "px"; }

// ---- score card
let shownScore = 0;
function setScore(p, text) {
  $("scorecard").classList.toggle("idle", p == null); $("verdict").textContent = text;
  if (p == null) { $("score").textContent = "–"; shownScore = 0; return; }
  const from = shownScore, to = 100 * p, t0 = performance.now(); $("meter-dot").style.left = to.toFixed(1) + "%";
  (function f(t) { const k = Math.min(1, (t - t0) / 550), v = from + (to - from) * (1 - Math.pow(1 - k, 3)); $("score").textContent = Math.round(v); shownScore = v; if (k < 1) requestAnimationFrame(f); })(t0);
}
function setMap(words) {
  const map = $("map"), n = Math.min(56, words.length); map.replaceChildren();
  for (let i = 0; i < n; i++) {
    const a = Math.floor(i * words.length / n), b = Math.floor((i + 1) * words.length / n); let s = 0; for (let j = a; j < b; j++) s += words[j][2];
    const v = b > a ? s / (b - a) : 0, el = document.createElement("b"); el.style.setProperty("--k", i); if (v > 0.04) el.style.background = colorOf(v); map.appendChild(el);
  }
}
const label = (p) => (p < 0.2 ? "Looks human-written" : p < 0.6 ? "Mixed or uncertain" : "Looks AI-generated");
function setStatus(state, text) { $("status").dataset.state = state; $("status-text").textContent = text; }
function say(msg, warn) { $("msg").textContent = msg || ""; $("msg").style.color = warn ? "#9a3412" : "#4d4a45"; }
const busy = () => S.sentId !== S.doneId;

// ---- worker messages
const fmtMB = (b) => (b / 1e6).toFixed(0);
function setBar(frac, indeterminate) { const bar = $("loader").querySelector(".bar"); bar.classList.toggle("indet", !!indeterminate); $("loader-fill").style.width = Math.round(100 * frac) + "%"; bar.setAttribute("aria-valuenow", Math.round(100 * frac)); }
function apply(m) { S.snap = m.text; S.probs = new Map(m.words.map((w) => [w[0], w[2]])); S.reveal = true; }
worker.onmessage = (e) => {
  const m = e.data;
  if (m.type === "status") $("loader-title").textContent = m.message;
  else if (m.type === "progress") {
    if (m.phase === "download") { setBar(m.loaded / m.total * 0.95); $("loader-detail").textContent = `${fmtMB(m.loaded)} of ${fmtMB(m.total)} MB`; } else { setBar(1, true); $("loader-detail").textContent = "initializing"; }
  } else if (m.type === "ready") {
    S.ready = true; setBar(1); $("loader").classList.add("done"); $("loader-title").textContent = m.cached ? "Model loaded from your browser cache" : "Model downloaded and stored in your browser"; $("loader-detail").textContent = m.threads > 1 ? `${m.threads} threads` : "1 thread";
    $("runinfo").textContent = `Model version ${m.version}, ${m.threads} CPU thread${m.threads > 1 ? "s" : ""}${m.isolated ? "" : " (cross-origin isolation is off, so this runs on one thread and is slower)"}.`;
    setStatus("ready", "Ready"); render(); if (S.wantRun || wordsOf(ta.value).length >= MIN_WORDS) run();
  } else if (m.type === "analyzing") { if (m.id === S.sentId) setStatus("busy", m.total > 1 ? `Reading, part ${m.done} of ${m.total}` : "Reading"); }
  else if (m.type === "partial") { if (m.id === S.sentId) { apply(m); render(); } }
  else if (m.type === "result") {
    if (m.id !== S.sentId) return;                                  // an older text: a newer one is already on its way
    S.doneId = m.id; apply(m); const flagged = m.words.filter((w) => w[2] > 0.5).length;
    setScore(m.docProb, `${label(m.docProb)} · ${m.words.length ? Math.round(100 * flagged / m.words.length) : 0}% of words flagged`); setMap(m.words); render();
    setStatus("ready", `Read ${m.tokens} tokens in ${m.ms < 1000 ? Math.round(m.ms) + " ms" : (m.ms / 1000).toFixed(1) + " s"}${m.cachedWindows ? " (part reused)" : ""}`);
    say(m.truncated ? "The text was cut to its first 30,000 characters." : m.tokens < 60 ? "Short text: keep writing for a steadier score." : "", m.truncated); window.__lastResult = m;
    if (ta.value.slice(0, 30000).normalize("NFC") !== m.text) run();               // changed while reading
  } else if (m.type === "skipped") { if (m.id === S.sentId) { S.doneId = m.id; run(); } }
  else if (m.type === "warn") say(m.message, true);
  else if (m.type === "cleared") { say("The downloaded model was removed from this browser. Reload the page to download it again."); S.ready = false; setStatus("loading", "Model removed"); }
  else if (m.type === "error") {
    if (m.where === "load") {
      $("loader-title").textContent = "The model could not be loaded"; $("loader-detail").textContent = ""; setBar(0); setStatus("error", "Model not loaded");
      const hint = /memory|allocation|RangeError|out of/i.test(m.message) ? " The device ran out of memory; the model needs about 1 GB free, so try a desktop browser and close other tabs." : location.protocol === "file:" ? " Open the page through a web server; browsers do not allow this from a file." : "";
      say(m.message + hint, true);
    } else { S.doneId = S.sentId; setStatus("error", "Could not read the text"); say(m.message, true); render(); }
  }
};
worker.onerror = (e) => { $("loader-title").textContent = "The model could not be started"; setStatus("error", "Not available"); say("The background worker failed (" + (e.message || "unknown error") + "). This page needs a recent Chrome, Edge, Firefox (114 or later) or Safari (16.4 or later) with WebAssembly and module workers.", true); };

// ---- reading the text: a moment after you pause, after a few finished words, or at once on paste
function run() {
  clearTimeout(S.timer); S.timerOn = false;
  const t = ta.value; if (wordsOf(t).length < MIN_WORDS) return;
  if (!S.ready) { S.wantRun = true; return; }
  S.wantRun = false; S.lastRun = performance.now(); S.analyzedWc = wordsOf(t).length;
  worker.postMessage({ type: "analyze", id: ++S.sentId, text: t });   // the worker drops any older request still waiting
  setStatus("busy", "Reading"); render();
}
function onInput(e) {
  const t = ta.value, wc = wordsOf(t).length; autosize(); updateCount(); render();
  if (wc < MIN_WORDS) { clearTimeout(S.timer); S.timerOn = false; setScore(null, wc ? "A few more words…" : "Start typing or paste text"); setStatus(S.ready ? "ready" : "loading", S.ready ? "Ready" : "Loading the model"); say(""); S.lastLen = t.length; return; }
  const type = (e && e.inputType) || "", pasted = type.startsWith("insertFromPaste") || type.startsWith("insertFromDrop") || Math.abs(t.length - S.lastLen) > 30;
  S.lastLen = t.length;
  const boundary = /[\s.,;:!?)\]"'”’]/.test(t.slice(-1));
  let delay = 650;                                                     // a pause
  if (pasted) delay = 30; else if (boundary && Math.abs(wc - S.analyzedWc) >= 3 && performance.now() - S.lastRun > 900) delay = 0;   // a few finished words
  clearTimeout(S.timer); S.timerOn = true; S.timer = setTimeout(run, delay); render();
}
function updateCount() { const n = wordsOf(ta.value).length; $("count").textContent = n ? `${n} word${n === 1 ? "" : "s"}` : ""; }

// ---- wiring
ta.addEventListener("input", onInput);
ta.addEventListener("keydown", (e) => { if ((e.ctrlKey || e.metaKey) && e.key === "Enter") run(); });
$("analyze").addEventListener("click", run);
$("clear").addEventListener("click", () => { ta.value = ""; S.snap = ""; S.probs = new Map(); S.shown = new Map(); $("map").replaceChildren(); onInput(null); ta.focus(); });
document.querySelectorAll("[data-sample]").forEach((b) => b.addEventListener("click", () => { ta.value = SAMPLES[b.dataset.sample]; onInput({ inputType: "insertFromPaste" }); ta.focus(); }));
$("clear-cache").addEventListener("click", () => worker.postMessage({ type: "clear" }));
$("editor").addEventListener("mousemove", (e) => {                     // the textarea is on top, so look through it for the word chip underneath
  const w = document.elementsFromPoint(e.clientX, e.clientY).find((x) => x.classList && x.classList.contains("w") && x.dataset.p);
  if (!w) { tip.classList.remove("on"); return; }
  tip.textContent = `P(AI) ${Math.round(100 * w.dataset.p)}%`; tip.style.left = e.clientX + "px"; tip.style.top = e.clientY + "px"; tip.classList.add("on");
});
$("editor").addEventListener("mouseleave", () => tip.classList.remove("on"));
window.addEventListener("resize", () => { autosize(); render(); });

if (typeof WebAssembly !== "object" || typeof Worker !== "function") { $("loader-title").textContent = "This browser cannot run the model"; setStatus("error", "Not supported"); say("WebAssembly and web workers are required. Use a recent Chrome, Edge, Firefox or Safari.", true); }
else if (!window.isSecureContext) say("This page must be served over https (or localhost) so the model can be cached.", true);
navigator.storage?.persist?.();
autosize(); render();
worker.postMessage({ type: "init", base: new URL("./model/", location.href).href });
window.__app = { SAMPLES };
