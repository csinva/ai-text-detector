#!/usr/bin/env python3
"""Zero-shot baselines on the held-out sets (comparison only; every checkpoint is Apache-2.0 or MIT):

  binoculars      Hans et al. 2024: observer Falcon-7B, performer Falcon-7B-Instruct, score = log-perplexity(performer) / cross-perplexity(observer, performer); lower = AI
  fast-detectgpt  Bao et al. 2024, analytic criterion, sampling model GPT-J-6B, scoring model GPT-Neo-2.7B; higher = AI
  detectgpt       Mitchell et al. 2023, scoring model GPT-Neo-2.7B, T5-large mask filling (15% of the words in spans of 2, N perturbations), z-score of the log-likelihood drop; higher = AI

Texts are cut to 512 tokens (200 words for DetectGPT, as in its paper). Scores are converted so that higher = AI.

  python 18_zeroshot_baselines.py --method binoculars --out runs/baselines_zs_binoculars.json [--sets a,b] [--cap 1500]
  python 18_zeroshot_baselines.py --method binoculars --humanizers --out runs/baselines_zs_binoculars_hum.json
  python 18_zeroshot_baselines.py --method binoculars --cpu-bench --out runs/cpu_bench_zs_binoculars.json
"""
import argparse, importlib.util, json, os, random, re, sys, time
import numpy as np, pandas as pd, torch
from sklearn.metrics import roc_auc_score
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__)); D = os.path.join(os.environ["V45_DATA_ROOT"], "data/aidetect")
sys.path.insert(0, HERE)
spec = importlib.util.spec_from_file_location("ev", f"{HERE}/04_eval.py"); ev = importlib.util.module_from_spec(spec); spec.loader.exec_module(ev)
ap = argparse.ArgumentParser()
ap.add_argument("--method", required=True, choices=["binoculars", "fast-detectgpt", "detectgpt"]); ap.add_argument("--out", required=True)
ap.add_argument("--sets", default=",".join(ev.DOC_SETS) + ",opai_doc"); ap.add_argument("--cap", type=int, default=1500); ap.add_argument("--npert", type=int, default=30)
ap.add_argument("--humanizers", action="store_true"); ap.add_argument("--cpu-bench", action="store_true"); ap.add_argument("--bs", type=int, default=8); ap.add_argument("--max-tokens", type=int, default=512)
a = ap.parse_args()
DEV = "cpu" if a.cpu_bench else "cuda"; DT = torch.bfloat16
if a.cpu_bench: torch.set_num_threads(8)
HUB = lambda r: r


def load_lm(repo, rev=None):
    tok = AutoTokenizer.from_pretrained(repo, revision=rev)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    m = AutoModelForCausalLM.from_pretrained(repo, revision=rev, torch_dtype=DT).to(DEV).eval()
    return tok, m


def batches(texts, bs):
    order = np.argsort([len(t) for t in texts])
    for i in range(0, len(texts), bs): yield order[i:i + bs]


# ------------------------------------------------------------------ Binoculars
class Binoculars:
    def __init__(self):
        self.tok, self.obs = load_lm("tiiuae/falcon-7b"); _, self.perf = load_lm("tiiuae/falcon-7b-instruct")

    @torch.no_grad()
    def score(self, texts):
        out = np.zeros(len(texts))
        for idx in batches(texts, a.bs):
            enc = self.tok([texts[j] for j in idx], return_tensors="pt", padding=True, truncation=True, max_length=a.max_tokens, return_token_type_ids=False).to(DEV)
            lo, lp = self.obs(**enc).logits[:, :-1].float(), self.perf(**enc).logits[:, :-1].float()
            lab = enc.input_ids[:, 1:]; mask = enc.attention_mask[:, 1:].float()
            ppl = -(torch.log_softmax(lp, -1).gather(-1, lab[..., None]).squeeze(-1) * mask).sum(-1) / mask.sum(-1).clamp(min=1)       # performer log-perplexity
            xppl = -((torch.softmax(lo, -1) * torch.log_softmax(lp, -1)).sum(-1) * mask).sum(-1) / mask.sum(-1).clamp(min=1)            # cross-perplexity
            out[idx] = -(ppl / xppl).cpu().numpy()                                                                                        # lower binoculars score = AI -> negate
        return out


# ------------------------------------------------------------------ Fast-DetectGPT
class FastDetectGPT:
    def __init__(self):
        self.tok, self.ref = load_lm("EleutherAI/gpt-j-6b", "float16"); _, self.sc = load_lm("EleutherAI/gpt-neo-2.7B")

    @torch.no_grad()
    def score(self, texts):
        out = np.zeros(len(texts))
        for idx in batches(texts, a.bs):
            enc = self.tok([texts[j] for j in idx], return_tensors="pt", padding=True, truncation=True, max_length=a.max_tokens).to(DEV)
            lr, ls = self.ref(input_ids=enc.input_ids, attention_mask=enc.attention_mask).logits[:, :-1].float(), self.sc(input_ids=enc.input_ids, attention_mask=enc.attention_mask).logits[:, :-1].float()
            v = min(lr.shape[-1], ls.shape[-1]); lr, ls = lr[..., :v], ls[..., :v]
            lab = enc.input_ids[:, 1:]; mask = enc.attention_mask[:, 1:].float()
            lps = torch.log_softmax(ls, -1); pr = torch.softmax(lr, -1)
            ll = (lps.gather(-1, lab[..., None]).squeeze(-1) * mask).sum(-1)
            mean_ref = ((pr * lps).sum(-1) * mask).sum(-1); var_ref = (((pr * lps.square()).sum(-1) - (pr * lps).sum(-1).square()) * mask).sum(-1)
            out[idx] = ((ll - mean_ref) / var_ref.clamp(min=1e-8).sqrt()).cpu().numpy()
        return out


# ------------------------------------------------------------------ DetectGPT
def tokenize_and_mask(text, rng, span=2, pct=0.15, buf=1):
    tokens = text.split(" "); n_spans = int(pct * len(tokens) / (span + buf * 2)); n = 0; tries = 0
    while n < n_spans and tries < 10 * n_spans + 10 and len(tokens) > span + 2:
        tries += 1; s = rng.randint(0, len(tokens) - span); e = s + span
        if "<<<mask>>>" not in tokens[max(0, s - buf):min(len(tokens), e + buf)]: tokens[s:e] = ["<<<mask>>>"]; n += 1
    k = 0
    for i, t in enumerate(tokens):
        if t == "<<<mask>>>": tokens[i] = f"<extra_id_{k}>"; k += 1
    return " ".join(tokens)


class DetectGPT:
    def __init__(self):
        self.tok, self.sc = load_lm("EleutherAI/gpt-neo-2.7B")
        self.mt = AutoTokenizer.from_pretrained("google-t5/t5-large"); self.mm = AutoModelForSeq2SeqLM.from_pretrained("google-t5/t5-large", torch_dtype=DT).to(DEV).eval(); self.rng = random.Random(0)

    @torch.no_grad()
    def perturb(self, text, n):
        masked = [tokenize_and_mask(text, self.rng) for _ in range(n)]; counts = [len(re.findall(r"<extra_id_\d+>", m)) for m in masked]
        enc = self.mt(masked, return_tensors="pt", padding=True, truncation=True, max_length=512).to(DEV)
        stop = self.mt.encode(f"<extra_id_{max(counts)}>")[0]
        outs = self.mm.generate(**enc, max_length=150, do_sample=True, top_p=1.0, num_return_sequences=1, eos_token_id=stop)
        dec = self.mt.batch_decode(outs, skip_special_tokens=False); res = []
        for m, d_, c in zip(masked, dec, counts):
            fills = [f.strip() for f in re.split(r"<extra_id_\d+>", d_.replace("<pad>", "").replace("</s>", "").strip())[1:-1]]
            toks = m.split(" ")
            if len(fills) < c: res.append(text); continue                                       # failed fill: keep the original (as the reference code skips it)
            for k in range(c): toks[toks.index(f"<extra_id_{k}>")] = fills[k]
            res.append(" ".join(toks))
        return res

    @torch.no_grad()
    def ll(self, texts):
        out = []
        for i in range(0, len(texts), 16):
            enc = self.tok(texts[i:i + 16], return_tensors="pt", padding=True, truncation=True, max_length=a.max_tokens).to(DEV)
            lg = self.sc(**enc).logits[:, :-1].float(); lab = enc.input_ids[:, 1:]; mask = enc.attention_mask[:, 1:].float()
            out += list(((torch.log_softmax(lg, -1).gather(-1, lab[..., None]).squeeze(-1) * mask).sum(-1) / mask.sum(-1).clamp(min=1)).cpu().numpy())
        return np.array(out)

    def score(self, texts):
        out = np.zeros(len(texts))
        for i, t in enumerate(texts):
            t = " ".join(t.split(" ")[:200]); pert = self.perturb(t, a.npert); lls = self.ll([t] + pert); sd = lls[1:].std()
            out[i] = (lls[0] - lls[1:].mean()) / max(sd, 1e-6)
        return out


M = {"binoculars": Binoculars, "fast-detectgpt": FastDetectGPT, "detectgpt": DetectGPT}[a.method]()
key = a.method

if a.cpu_bench:
    texts = pd.read_parquet(f"{D}/prep_lic3/eval/spans_test.parquet").text.tolist()[:3]
    texts = [M.tok.decode(M.tok(t, add_special_tokens=False)["input_ids"][:500]) for t in texts]
    M.score(texts[:1]) if False else None
    res = []
    for t in texts[:2]:
        t0 = time.time(); M.score([t]); res.append(time.time() - t0); print(key, "cpu s/doc", res[-1], flush=True)
    json.dump([dict(model=key, variant="bf16", threads=8, tokens=512, sec_per_doc=float(np.mean(res)), reps=res)], open(a.out, "w"), indent=1); sys.exit(0)

if a.humanizers:                                   # same protocol as 17_baselines_humanizers.py (subsampled for DetectGPT)
    H = []
    for n in ("wasit_2026", "textsight_2026", "gen_id"):
        df = pd.read_parquet(f"{D}/prep_lic3/eval/{n}.parquet"); df = df[df.label == 0]; H.append(df.sample(min(len(df), 1500), random_state=0).text.tolist())
    hum = sum(H, []); d = pd.read_parquet(f"{D}/licensed/humanizer.parquet")
    if key == "detectgpt":
        hum = random.Random(0).sample(hum, 500); d = pd.concat([d[d.kind == "output"].sample(600, random_state=0), d[d.kind == "input"].sample(150, random_state=0)])
    hp = M.score(hum); d = d.copy(); d["s"] = M.score(d.text.tolist())
    thr = {f: float(np.quantile(hp, 1 - f)) for f in (0.01, 0.05)}; o, i = d[d.kind == "output"], d[d.kind == "input"]
    row = lambda g: {**{f"recall@{int(f * 100)}": float((g.s > t).mean()) for f, t in thr.items()}, "auc": float(roc_auc_score(np.r_[np.zeros(len(hp)), np.ones(len(g))], np.r_[hp, g.s.values])), "n": len(g)}
    res = {key: {"thr": thr, "outputs": row(o), "inputs": row(i), "n_humans": len(hp)}}; print(key, res[key]["outputs"], res[key]["inputs"], flush=True)
    json.dump(res, open(a.out, "w"), indent=1); sys.exit(0)

res = {key: {}}
cap = a.cap
for name in a.sets.split(","):
    df = pd.read_parquet(f"{D}/prep_lic3/eval/{name}.parquet")
    if len(df) > cap:                                          # stratified cap: up to half of the cap from the smaller class (WasIt has only 6% human texts)
        h, m_ = df[df.label == 0], df[df.label == 1]; small, big = (h, m_) if len(h) <= len(m_) else (m_, h); ns = min(len(small), cap // 2)
        df = pd.concat([small.sample(ns, random_state=0), big.sample(cap - ns, random_state=0)])
    t0 = time.time(); p = M.score(df.text.tolist()); dt = time.time() - t0
    res[key][name] = dict(doc=ev.doc_metrics(df.label.values, p), seconds=dt, n_scored=len(df), **ev.breakdown(df, p, skip=name == "spans_test"))
    print(key, name, len(df), {k: round(v, 1) for k, v in res[key][name]["doc"].items()}, f"{dt:.0f}s", flush=True)
    json.dump(res, open(a.out, "w"), indent=1)
