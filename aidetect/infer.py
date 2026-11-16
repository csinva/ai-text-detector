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


def token_to_word(text, offsets, tok_prob, reduce="mean"):
    """per whitespace-delimited word: mean (probabilities) or sum (attributions) over its tokens. Returns [(start, end, value)]."""
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
    if reduce == "sum":
        return [(s, e, float(v)) for (s, e), v in zip(words, num)]
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

    def explain(self, text, steps=24, max_tokens=1024, attn_layers=12):
        """Why does the document head say what it says? Two per-word views of the *document* logit (needs an fp32 model, quant=0):
        'ig'   integrated gradients of the document logit with respect to the input embeddings (baseline: the end-of-text token at every position), summed over
               embedding dimensions and over the tokens of a word. Positive = pushes toward AI, negative = toward human. The sum over
               all tokens equals logit(text) minus logit(baseline) (completeness; `ig_gap` reports the residual).
        'attn' attention rollout (head-averaged attention, residual connections mixed in, product over layers): how much of every
               token's final representation flows from each token, averaged over query tokens.
        Only the first `max_tokens` tokens are explained (the matrices are quadratic in length)."""
        enc = self.tk.encode(text[:200000])
        n = min(len(enc.ids), max_tokens)
        ids = torch.tensor(enc.ids[:n]).to(self.dev)
        pos = torch.arange(n).to(self.dev); cu = torch.tensor([0, n], dtype=torch.int32).to(self.dev); seg = torch.zeros(n, dtype=torch.long).to(self.dev)
        m = self.m
        with torch.no_grad():
            x = m.embed(ids).float()
        with torch.no_grad():
            x0 = m.embed(torch.full_like(ids, 151643)).float()  # <|endoftext|> everywhere: a zero baseline breaks down because RMSNorm amplifies near-zero inputs

        def logit(e):
            return m.heads(m.hidden(ids, pos, cu, n, e), seg, 1)[1][0]

        grads = torch.zeros_like(x)
        with torch.enable_grad():
            for a in (torch.arange(steps, dtype=torch.float32) + 0.5) / steps:  # midpoint rule
                xa = (x0 + a * (x - x0)).clone().requires_grad_(True)
                grads += torch.autograd.grad(logit(xa), xa)[0] / steps
        tok_ig = ((x - x0) * grads).sum(-1).detach().cpu().numpy()
        with torch.no_grad():
            full, base = logit(x).item(), logit(x0).item()
            m.attn_store = []
            logit(x)
            A, m.attn_store = m.attn_store, None
            R = torch.eye(n)
            for w in A[-attn_layers:]:
                w = 0.5 * w.cpu() + 0.5 * torch.eye(n)
                R = w @ R
            tok_attn = R.mean(0).numpy()  # column mean: average share of token j in the final representations
        off = enc.offsets[:n]
        return dict(n_tokens=n, doc_logit=full, baseline_logit=base, ig_gap=float(tok_ig.sum() - (full - base)),
                    ig=token_to_word(text, off, tok_ig, "sum"), attn=token_to_word(text, off, tok_attn * n, "sum"))


def heatmap_html(text, words, doc_prob=None):
    """kept for compatibility: the redesigned views live in heatmap.py"""
    from heatmap import render_card
    return render_card(text, dict(doc_prob=doc_prob if doc_prob is not None else 0.5, words=words, n_tokens=len(words)))
