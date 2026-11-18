// Byte-level BPE tokenizer for the Qwen2/Qwen3 vocabulary (voyage-4-nano), with character offsets.
// Mirrors the `tokenizers` pipeline of tokenizer.json: NFC normalizer, regex split (isolated), GPT-2 byte-to-unicode, BPE merges, no special tokens added.
// Offsets are UTF-16 code unit offsets into the NFC-normalized text (equal to the `tokenizers` character offsets for text without astral characters).

const PRETOK = new RegExp(String.raw`(?:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`, "giu");

function byteToChar() {
  const bs = [];
  for (let b = 33; b <= 126; b++) bs.push(b);
  for (let b = 161; b <= 172; b++) bs.push(b);
  for (let b = 174; b <= 255; b++) bs.push(b);
  const cs = bs.slice(); let n = 0;
  for (let b = 0; b < 256; b++) if (!bs.includes(b)) { bs.push(b); cs.push(256 + n); n++; }
  const m = new Array(256);
  bs.forEach((b, i) => { m[b] = String.fromCharCode(cs[i]); });
  return m;
}

export class BPETokenizer {
  constructor(json) {
    this.vocab = new Map(Object.entries(json.vocab));
    this.ranks = new Map(); json.merges.forEach((m, i) => this.ranks.set(m, i));
    this.b2c = byteToChar(); this.cache = new Map(); this.enc = new TextEncoder();
  }

  bpe(mapped) {
    let parts = this.cache.get(mapped);
    if (parts) return parts;
    parts = Array.from(mapped);
    while (parts.length > 1) {
      let best = Infinity, bi = -1;
      for (let i = 0; i < parts.length - 1; i++) {
        const r = this.ranks.get(parts[i] + " " + parts[i + 1]);
        if (r !== undefined && r < best) { best = r; bi = i; }
      }
      if (bi < 0) break;
      const a = parts[bi], b = parts[bi + 1], out = [];
      for (let i = 0; i < parts.length; i++) {
        if (i < parts.length - 1 && parts[i] === a && parts[i + 1] === b) { out.push(a + b); i++; } else out.push(parts[i]);
      }
      parts = out;
    }
    if (this.cache.size < 200000) this.cache.set(mapped, parts);
    return parts;
  }

  encode(rawText) {
    const text = rawText.normalize("NFC"); const ids = [], offsets = [];
    const pieces = []; let last = 0;
    for (const m of text.matchAll(PRETOK)) {
      if (m.index > last) pieces.push([last, text.slice(last, m.index)]);
      if (m[0].length) pieces.push([m.index, m[0]]);
      last = m.index + m[0].length;
    }
    if (last < text.length) pieces.push([last, text.slice(last)]);
    for (const [start, piece] of pieces) {
      const bytes = this.enc.encode(piece), us = new Int32Array(bytes.length), ue = new Int32Array(bytes.length);
      let bi = 0, u = 0;
      for (const ch of piece) {                                  // iterates code points
        const nb = this.enc.encode(ch).length, nu = ch.length;
        for (let k = 0; k < nb; k++) { us[bi + k] = u; ue[bi + k] = u + nu; }
        bi += nb; u += nu;
      }
      let mapped = ""; for (let i = 0; i < bytes.length; i++) mapped += this.b2c[bytes[i]];
      let pos = 0;
      for (const tokStr of this.bpe(mapped)) {
        const id = this.vocab.get(tokStr), n = tokStr.length;      // one mapped char per byte
        if (id === undefined) throw new Error("token not in vocabulary: " + JSON.stringify(tokStr));
        ids.push(id); offsets.push([start + us[pos], start + ue[pos + n - 1]]); pos += n;
      }
    }
    return { ids, offsets, text };
  }
}
