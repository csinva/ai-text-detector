// Windowing and word aggregation, ported from aidetect/infer.py.

export function windows(n, window) {
  if (n <= window) return [[0, n]];
  const st = []; for (let s = 0; s < n - window; s += window) st.push(s); st.push(n - window);
  return st.map((s) => [s, s + window]);
}

// mean token probability per whitespace-delimited word -> [[start, end, prob], ...]
export function tokenToWord(text, offsets, tokProb) {
  const words = []; for (const m of text.matchAll(/\S+/g)) words.push([m.index, m.index + m[0].length]);
  if (!words.length) return [];
  const wid = new Int32Array(text.length + 1).fill(-1);
  words.forEach(([s, e], i) => { for (let j = s; j < e; j++) wid[j] = i; });
  const num = new Float64Array(words.length), den = new Float64Array(words.length), clip = (v) => Math.min(Math.max(v, 0), text.length - 1);
  for (let t = 0; t < offsets.length; t++) {
    let w = wid[clip(offsets[t][1] - 1)]; if (w < 0) w = wid[clip(offsets[t][0])];
    if (w >= 0) { num[w] += tokProb[t]; den[w] += 1; }
  }
  const prob = Array.from(num, (v, i) => (den[i] > 0 ? v / den[i] : NaN));
  const firstOk = prob.find((v) => !Number.isNaN(v));
  for (let i = 0; i < prob.length; i++) if (Number.isNaN(prob[i])) prob[i] = i ? prob[i - 1] : (firstOk ?? 0.5);
  return words.map(([s, e], i) => [s, e, prob[i]]);
}

export const sigmoid = (x) => 1 / (1 + Math.exp(-x));

// run `runWindow(ids) -> {tok: Float32Array, doc: number}` over the windows of one text
export async function scoreIds(ids, window, runWindow, onProgress, shouldAbort) {
  const n = ids.length, tokSum = new Float64Array(n), tokCnt = new Float64Array(n), ws = windows(n, window);
  let dl = 0, dw = 0, k = 0;
  for (const [s, e] of ws) {
    if (shouldAbort?.()) throw new Error("superseded");
    const { tok, doc } = await runWindow(ids.slice(s, e));
    for (let i = 0; i < e - s; i++) { tokSum[s + i] += sigmoid(tok[i]); tokCnt[s + i] += 1; }
    dl += doc * (e - s); dw += e - s; k++; onProgress?.(k, ws.length, { tokSum, tokCnt, upTo: e });
  }
  const tp = Float64Array.from(tokSum, (v, i) => v / Math.max(tokCnt[i], 1)); let mean = 0; for (const v of tp) mean += v;
  return { docLogit: dl / dw, docProb: sigmoid(dl / dw), tokProb: tp, tokProbMean: tp.length ? mean / tp.length : 0.5, windows: ws.length };
}
