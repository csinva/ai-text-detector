"""Inference for the AI-text detector: document probability + word-level probabilities, on GPU or CPU.

    from infer import Predictor
    p = Predictor("<run dir>/model.pt", device="cpu")
    out = p.predict(["text one", "text two"])      # [{'doc_prob': .., 'words': [(start, end, prob), ...]}, ...]

Long texts are cut into windows of at most `window` tokens (the last window is anchored to the end); token
probabilities in an overlap are averaged, the document probability is the token-count weighted mean of window probabilities.
A word's probability is the mean probability of the tokens that end inside it.
"""
from __future__ import annotations
import html, os, re, sys
import numpy as np, torch
from tokenizers import Tokenizer

sys.path.insert(0, os.path.dirname(__file__))
from model import Cfg, Detector, pack

TOK = "/data/new/prod_models/voyage-4-nano/tokenizer.json"
WORD = re.compile(r"\S+")


def windows(n, window):
    if n <= window:
        return [(0, n)]
    st = list(range(0, n - window, window)) + [n - window]
    return [(s, s + window) for s in st]


def token_to_word(text, offsets, tok_prob):
    """mean token probability per whitespace-delimited word. Returns [(start, end, prob)]."""
    words = [(m.start(), m.end()) for m in WORD.finditer(text)]
    if not words:
        return []
    wid = np.full(len(text) + 1, -1, dtype=np.int64)
    for i, (s, e) in enumerate(words):
        wid[s:e] = i
    off = np.asarray(offsets, dtype=np.int64).reshape(-1, 2)
    last = np.clip(off[:, 1] - 1, 0, len(text) - 1)
    w = wid[last]
    first = wid[np.clip(off[:, 0], 0, len(text) - 1)]
    w = np.where(w >= 0, w, first)
    ok = w >= 0
    num = np.bincount(w[ok], weights=tok_prob[ok], minlength=len(words)); den = np.bincount(w[ok], minlength=len(words))
    prob = np.where(den > 0, num / np.maximum(den, 1), np.nan)
    # words without a token (rare) take the previous word's value
    for i in range(len(prob)):
        if np.isnan(prob[i]):
            prob[i] = prob[i - 1] if i else (prob[~np.isnan(prob)][0] if (~np.isnan(prob)).any() else 0.5)
    return [(s, e, float(p)) for (s, e), p in zip(words, prob)]


class Predictor:
    def __init__(self, path, device="cuda", window=1024, dtype=None, n_layers=12, causal=False, quant=False):
        self.dev = torch.device(device); self.window = window
        self.tk = Tokenizer.from_file(TOK)
        m = Detector(Cfg(causal=causal), n_layers)
        sd = torch.load(path, map_location="cpu")
        m.load_state_dict(sd, strict=True)
        self.dtype = dtype or (torch.bfloat16 if self.dev.type == "cuda" else torch.float32)
        m.to(self.dev); m.init_rope(self.dev); m.eval()
        if self.dev.type == "cuda":
            m.attn_impl = "flash"
        else:
            m.attn_impl = "sdpa"
        if self.dev.type != "cuda" and self.dtype != torch.float32:
            m.to(self.dtype)
        if quant:  # dynamic int8 (CPU). quant=1: all linear layers; 2: all but the MLP down projection and the heads; 3: all but the heads
            skip = {1: (), 2: ("down", "head"), 3: ("head",)}[int(quant)]
            names = {n for n, mod in m.named_modules() if isinstance(mod, torch.nn.Linear) and not any(s in n for s in skip)}
            m = torch.ao.quantization.quantize_dynamic(m, names, dtype=torch.qint8)
        self.m = m

    @torch.no_grad()
    def predict(self, texts, batch_tokens=32768, with_words=True, max_chars=200000):
        texts = [t[:max_chars] for t in texts]
        encs = self.tk.encode_batch(texts)
        items = []  # (doc, start, end)
        for d, e in enumerate(encs):
            for s, en in windows(len(e.ids), self.window):
                if en > s:
                    items.append((d, s, en))
        order = sorted(range(len(items)), key=lambda i: items[i][2] - items[i][1])  # length-bucketed batches, no padding anyway
        tok_sum = [np.zeros(len(e.ids)) for e in encs]; tok_cnt = [np.zeros(len(e.ids)) for e in encs]
        doc_logit = np.zeros(len(texts)); doc_w = np.zeros(len(texts))
        i = 0
        while i < len(order):
            batch, tot = [], 0
            while i < len(order) and (not batch or tot + items[order[i]][2] - items[order[i]][1] <= batch_tokens):
                batch.append(order[i]); tot += items[order[i]][2] - items[order[i]][1]; i += 1
            seqs = [encs[items[j][0]].ids[items[j][1]:items[j][2]] for j in batch]
            ids, pos, cu, seg, ms, n = pack(seqs, self.dev)
            with torch.autocast(self.dev.type, dtype=self.dtype, enabled=self.dev.type == "cuda"):
                tok, doc = self.m(ids, pos, cu, seg, ms, n)
            tp = torch.sigmoid(tok.float()).cpu().numpy(); dl = doc.float().cpu().numpy()
            off = 0
            for k, (j, s) in enumerate(zip(batch, seqs)):
                d, a, b = items[j]
                tok_sum[d][a:b] += tp[off:off + len(s)]; tok_cnt[d][a:b] += 1
                doc_logit[d] += dl[k] * len(s); doc_w[d] += len(s)
                off += len(s)
        out = []
        for d, t in enumerate(texts):
            r = dict(doc_logit=float(doc_logit[d] / max(doc_w[d], 1)), n_tokens=len(encs[d].ids))
            r["doc_prob"] = float(1 / (1 + np.exp(-r["doc_logit"])))
            tp = tok_sum[d] / np.maximum(tok_cnt[d], 1)
            r["tok_prob_mean"] = float(tp.mean()) if len(tp) else 0.5
            if with_words:
                r["words"] = token_to_word(t, encs[d].offsets, tp)
            out.append(r)
        return out


def heatmap_html(text, words, doc_prob=None):
    """word-level heatmap: background alpha proportional to P(AI)."""
    out, pos = [], 0
    for s, e, p in words:
        out.append(html.escape(text[pos:s]))
        out.append(f'<span style="background:rgba(230,70,50,{p:.2f})" title="P(AI)={p:.2f}">{html.escape(text[s:e])}</span>')
        pos = e
    out.append(html.escape(text[pos:]))
    head = f"<p><b>P(AI-generated) = {doc_prob:.3f}</b></p>" if doc_prob is not None else ""
    return f'<div style="font-family:sans-serif;line-height:1.7;white-space:pre-wrap;max-width:50em">{head}{"".join(out)}</div>'
