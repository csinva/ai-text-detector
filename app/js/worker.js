// Inference worker: downloads the model once (progress messages), keeps it in Cache Storage, runs it with onnxruntime-web (WASM, SIMD, threads when cross-origin isolated).
import * as ort from "../ort/ort.wasm.min.mjs";
import { BPETokenizer } from "./tokenizer.js";
import { scoreIds, tokenToWord } from "./detector.js";

const CACHE = "ai-text-detector-model";
const MAX_CHARS = 30000;
let session = null, tok = null, manifest = null, windowSize = 1024, loading = null;
const post = (m) => self.postMessage(m);

async function sha256(buf) {
  const d = new Uint8Array(await crypto.subtle.digest("SHA-256", buf));
  return Array.from(d, (b) => b.toString(16).padStart(2, "0")).join("");
}

async function fetchInto(url, key, size, sha, dest, cache, onBytes) {
  const req = new Request(key);
  const hit = cache ? await cache.match(req) : null;
  if (hit) {
    const buf = new Uint8Array(await hit.arrayBuffer());
    if (buf.length === size) { dest.set(buf); onBytes(size); return { cached: true }; }
    await cache.delete(req);
  }
  const resp = await fetch(url);
  if (!resp.ok) throw new Error(`download failed (${resp.status}) for ${url}`);
  const reader = resp.body.getReader(); let off = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    if (off + value.length > size) throw new Error("download larger than expected: " + url);
    dest.set(value, off); off += value.length; onBytes(value.length);
  }
  if (off !== size) throw new Error(`incomplete download for ${url} (${off} of ${size} bytes)`);
  if (sha && (await sha256(dest)) !== sha) throw new Error("checksum mismatch for " + url);
  try { await cache?.put(req, new Response(dest.slice(), { headers: { "content-type": "application/octet-stream" } })); }
  catch (e) { post({ type: "warn", message: "Could not store the model in the browser cache (" + (e.name || e) + "); it will be downloaded again next time." }); }
  return { cached: false };
}

async function load(base) {
  const root = new URL(base, self.location.href).href;
  manifest = await (await fetch(root + "manifest.json", { cache: "no-cache" })).json();
  windowSize = manifest.window || 1024;
  const cache = self.caches ? await caches.open(CACHE) : null;
  for (const k of cache ? await cache.keys() : []) if (!k.url.includes("?v=" + manifest.version)) await cache.delete(k);  // drop older model versions
  const total = manifest.total_bytes + manifest.tokenizer_bytes; let done = 0, anyDownload = false;
  const bump = (n) => { done += n; post({ type: "progress", phase: "download", loaded: done, total }); };
  post({ type: "status", message: "Checking for a cached model" });
  const tokBuf = new Uint8Array(manifest.tokenizer_bytes);
  const r0 = await fetchInto(root + manifest.tokenizer, root + manifest.tokenizer + "?v=" + manifest.version, manifest.tokenizer_bytes, null, tokBuf, cache, bump);
  anyDownload ||= !r0.cached;
  const model = new Uint8Array(manifest.total_bytes); let at = 0;
  for (const p of manifest.parts) {
    const r = await fetchInto(root + p.name, root + p.name + "?v=" + manifest.version, p.bytes, p.sha256, model.subarray(at, at + p.bytes), cache, bump);
    anyDownload ||= !r.cached; at += p.bytes;
    post({ type: "status", message: anyDownload ? "Downloading the model (first visit only)" : "Reading the cached model" });
  }
  post({ type: "status", message: "Starting the model" }); post({ type: "progress", phase: "init", loaded: total, total });
  tok = new BPETokenizer(JSON.parse(new TextDecoder().decode(tokBuf)));
  const threads = self.crossOriginIsolated ? Math.max(1, Math.min(8, (navigator.hardwareConcurrency || 4) - 1)) : 1;
  ort.env.wasm.wasmPaths = new URL("../ort/", import.meta.url).href; ort.env.wasm.numThreads = threads; ort.env.wasm.proxy = false;
  session = await ort.InferenceSession.create(model, { executionProviders: ["wasm"], graphOptimizationLevel: "all" });
  post({ type: "ready", cached: !anyDownload, threads, isolated: !!self.crossOriginIsolated, version: manifest.version });
}

async function analyze(id, rawText) {
  const t0 = performance.now();
  let text = rawText.length > MAX_CHARS ? rawText.slice(0, MAX_CHARS) : rawText;
  const enc = tok.encode(text); text = enc.text;
  if (!enc.ids.length) throw new Error("Nothing to analyze.");
  const run = async (ids) => {
    const out = await session.run({ ids: new ort.Tensor("int64", BigInt64Array.from(ids, (v) => BigInt(v)), [ids.length]) });
    return { tok: out.tok.data, doc: out.doc.data[0] };
  };
  const r = await scoreIds(enc.ids, windowSize, run, (k, n) => post({ type: "analyzing", id, done: k, total: n }));
  post({ type: "result", id, text, truncated: rawText.length > MAX_CHARS, tokens: enc.ids.length, windows: r.windows, docProb: r.docProb, docLogit: r.docLogit, tokProbMean: r.tokProbMean,
         words: tokenToWord(text, enc.offsets, r.tokProb), ms: performance.now() - t0 });
}

let chain = Promise.resolve();
self.onmessage = (e) => {
  const m = e.data;
  if (m.type === "init") {
    loading ||= load(m.base).catch((err) => { loading = null; post({ type: "error", where: "load", message: String(err.message || err) }); });
  } else if (m.type === "analyze") {
    chain = chain.then(async () => { await loading; if (!session) return; try { await analyze(m.id, m.text); } catch (err) { post({ type: "error", where: "analyze", id: m.id, message: String(err.message || err) }); } });
  } else if (m.type === "clear") {
    (async () => { await caches.delete(CACHE); post({ type: "cleared" }); })();
  }
};
