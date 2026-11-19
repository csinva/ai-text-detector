#!/usr/bin/env python3
"""Round 5 data: explicit mixed human/AI documents (to teach the token head to separate), and layout-augmented pure documents (so line wrapping is not a cue).

Mixed documents are built from the human passages of the pool and the pure AI texts generated from them (same passage, a topic-matched AI text; or an AI text of another passage = cross-topic).
Units are swapped between the two texts with a persistence chain (the AI share per document is drawn from three buckets: AI-heavy 0.75-0.98, human-heavy 0.02-0.25, mid), at four grains:
  mix_sent   single sentences        mix_block  runs of 2-4 sentences   mix_para  paragraphs (or groups of 4 sentences)
  mix_word   runs of 3-14 words (a phrase-level splice)                    mix_cross sentences or blocks from an AI text of a different passage and generator
AI units optionally come from two different generators. Character spans of the AI units are exact. Typography of the AI units is straightened with probability 0.55 as in 01_prep.py.
Layout augmentation (length-preserving, so spans stay valid): 18% of mixed documents and all fmt_* documents are re-wrapped to a random width of 55 to 110 characters (as in text pasted from a PDF) or
flattened to one line per paragraph. fmt_human / fmt_ai are pure scientific and legal documents from the current training mixture with such layouts, one per class.
Held-out: documents whose passage uid is in the 8% evaluation hash split of 01_prep.py go only to the new evaluation sets.

  VER=6 python 20_build_mixed.py --n 250000     ->  data/aidetect/prep_lic6_r15_full/{train,dev}.parquet + eval/{mix_*,fmt_pure}.parquet
"""
import argparse, glob, hashlib, importlib.util, math, os, random, re, shutil, sys
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); D = os.path.join(os.environ["V45_DATA_ROOT"], "data/aidetect"); LIC = f"{D}/licensed"
SRC_DIR = f"{D}/prep_lic5_r15_full"
ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=250000); ap.add_argument("--n_fmt", type=int, default=30000); ap.add_argument("--n_eval", type=int, default=900); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out_name", default="prep_lic6_r15_full"); ap.add_argument("--frontier", action="store_true", help="add documents from the r5_* generation files (Claude, DeepSeek, gpt-oss-120b)"); a = ap.parse_args()
OUT = f"{D}/{a.out_name}"
argv, sys.argv = sys.argv, ["01_prep.py"]
spec = importlib.util.spec_from_file_location("prep", f"{HERE}/01_prep.py"); prep = importlib.util.module_from_spec(spec); spec.loader.exec_module(prep); sys.argv = argv
trim_end, straighten = prep.trim_end, prep.straighten
ev_uid = lambda u: int(hashlib.md5(u.encode()).hexdigest(), 16) % 100 < 8
UNIT = re.compile(r"(?s).*?[.!?]+[\"')\]”’]*(?:\s+|$)|.+$"); WORD = re.compile(r"\S+\s*|\s+")
SCI = {"gen_arxiv_full", "gen_pes2o", "gen_pubmed2", "gen_arxiv_abstract", "gen_regulations", "gen_caselaw"}


def units(t, kind):
    if kind == "word": return WORD.findall(t)
    if kind == "para":
        ps = [p for p in re.split(r"(\n\s*\n)", t)]; ps = ["".join(ps[i:i + 2]) for i in range(0, len(ps), 2)]
        if len(ps) >= 3: return ps
        s = [m.group(0) for m in UNIT.finditer(t)]; return ["".join(s[i:i + 4]) for i in range(0, len(s), 4)]
    return [m.group(0) for m in UNIT.finditer(t) if m.group(0)]


def assemble(hu, au, f, ks, target, rng):
    """alternate runs of human and AI units; `f` is the Bernoulli probability that a run is AI, the state persists with probability 0.55"""
    hi = ai = pos = 0; out, spans = [], []; state = rng.random() < f
    while pos < target:
        k = rng.choice(ks)
        if rng.random() > 0.55: state = rng.random() < f
        pool, i = (au, ai) if state else (hu, hi)
        if i >= len(pool):
            if state and hi < len(hu): state = False; pool, i = hu, hi
            elif not state and ai < len(au): state = True; pool, i = au, ai
            else: break
        seg = "".join(pool[i:i + k])
        if state: ai += k
        else: hi += k
        if out and not out[-1][-1:].isspace() and not seg[:1].isspace(): out[-1] += " "; pos += 1
        if state: spans.append([pos, pos + len(seg)])
        out.append(seg); pos += len(seg)
    text = "".join(out); merged = []
    for s, e in spans:
        if merged and s <= merged[-1][1]: merged[-1][1] = e
        else: merged.append([s, e])
    return text, merged


def rewrap(t, rng, mode=None):
    mode = mode or rng.choice(["wrap", "wrap", "flat"]); t = re.sub(r"(?<!\n)\n(?!\n)", " ", t)           # length-preserving: only spaces and single newlines trade places
    if mode == "flat": return t
    W = rng.randint(55, 110); c = list(t); col, last = 0, -1
    for i, ch in enumerate(c):
        if ch == "\n": col, last = 0, -1; continue
        col += 1
        if ch == " ": last = i
        if col > W and last >= 0: c[last] = "\n"; col = i - last; last = -1
    return "".join(c)


def main():
    rng = random.Random(a.seed)
    h = pd.read_parquet(f"{LIC}/human_cp.parquet"); h["uid"] = [f"{s}:{i}" for i, s in zip(h.index, h.src)]; h["text"] = h.text.map(trim_end); h = h[h.text.str.len() >= 300]
    hum = dict(zip(h.uid, h.text)); hsrc = dict(zip(h.uid, h.src))
    g = pd.concat([pd.read_parquet(f"{LIC}/gen/{f}", columns=["text", "spans", "src", "task", "gen_model", "uid"]) for f in sorted(os.listdir(f"{LIC}/gen")) if f.endswith(".parquet") and "smoke" not in f])
    pure = g[[len(sp) == 1 and sp[0][0] == 0 and sp[0][1] >= len(t) - 2 for sp, t in zip(g.spans, g.text)]]; pure = pure[pure.uid.isin(hum)]
    ai = {}; allai = []
    for u, t, m, s in zip(pure.uid, pure.text, pure.gen_model, pure.src):
        t = trim_end(t)
        if len(t) >= 300: ai.setdefault(u, []).append((t, m)); allai.append((u, t, m, s))
    uids = [u for u in ai]; tr_u = [u for u in uids if not ev_uid(u)]; ev_u = [u for u in uids if ev_uid(u)]
    trs, evs_ = set(tr_u), set(ev_u); bysrc = {"tr": {}, "ev": {}}; alls = {"tr": [x for x in allai if x[0] in trs], "ev": [x for x in allai if x[0] in evs_]}
    for x in allai: bysrc["tr" if x[0] in trs else "ev"].setdefault(x[3], []).append(x)
    hbysrc = {"tr": {}, "ev": {}}
    for u_ in uids: hbysrc["tr" if u_ in trs else "ev"].setdefault(hsrc[u_], []).append(hum[u_])
    print(f"human {len(hum):,}, passages with pure AI texts {len(uids):,} (train {len(tr_u):,}, eval {len(ev_u):,}), pure AI texts {len(allai):,}", flush=True)

    def make(u, gran, bucket, split, ai_d=None, by_d=None, all_d=None):
        f = {"heavy": rng.uniform(0.75, 0.98), "light": rng.uniform(0.02, 0.25), "mid": rng.uniform(0.25, 0.75)}[bucket]
        kind = {"sent": "sent", "block": "sent", "para": "para", "word": "word", "cross": "sent"}[gran]; ks = {"sent": [1], "block": [2, 3, 4], "para": [1], "word": list(range(3, 15)), "cross": [1, 1, 2, 3]}[gran]
        H = hum[u]; cands = (ai_d or ai)[u]
        if gran == "cross":
            pool = (by_d or bysrc)[split].get(hsrc[u]) or (all_d or alls)[split]; x = rng.choice(pool)
            if x[0] == u: return None
            _, A, m, _ = x
        else: A, m = rng.choice(cands)
        A = trim_end(A)
        if len(cands) > 1 and rng.random() < 0.35 and gran != "cross":                                       # two generators in one document
            A2, _ = rng.choice([c for c in cands if c[1] != m] or cands); A = A + " " + trim_end(A2) if A2 != A else A
        if rng.random() < 0.55: A = straighten(A)
        target = int(math.exp(rng.uniform(math.log(500), math.log(2400))))
        hu, au = units(H, kind), units(A, kind); need = 1.4 * target                                              # extend both pools with same-source text so the AI share follows its bucket
        for _ in range(3):
            if sum(map(len, hu)) < need: hu += units(rng.choice(hbysrc[split][hsrc[u]]), kind)
            if sum(map(len, au)) < need:
                A2 = trim_end(rng.choice((by_d or bysrc)[split].get(hsrc[u]) or (all_d or alls)[split])[1]); au += units(straighten(A2) if rng.random() < 0.55 else A2, kind)
        text, spans = assemble(hu, au, f, ks, target, rng)
        if len(text) < 250 or not spans: return None
        frac = sum(e - s for s, e in spans) / len(text)
        lo, hi_ = {"heavy": (0.75, 0.97), "light": (0.03, 0.25), "mid": (0.25, 0.75)}[bucket]               # buckets are defined by the realised AI share
        if not lo <= frac <= hi_: return None
        if rng.random() < 0.18: text = rewrap(text, rng)
        return text, spans, frac

    GRAN = ["sent"] * 35 + ["block"] * 30 + ["para"] * 10 + ["word"] * 15 + ["cross"] * 10; BUCKET = ["heavy"] * 30 + ["light"] * 30 + ["mid"] * 40
    def build(pool_u, n, tag, split):
        rows, tries = [], 0
        while len(rows) < n and tries < 60 * n:
            tries += 1; u = rng.choice(pool_u); gran, bucket = rng.choice(GRAN), rng.choice(BUCKET); r = make(u, gran, bucket, split)
            if r: rows.append(dict(text=r[0], spans=r[1], src=f"mix_{gran}", frac=r[2], bucket=bucket, uid=u))
            if len(rows) % 50000 == 0 and len(rows) and tries % 7 == 0: print(tag, len(rows), flush=True)
        return pd.DataFrame(rows)
    trn = build(tr_u, a.n, "train", "tr"); print("mixed train", len(trn), trn.src.value_counts().to_dict(), "mean AI share", round(trn.frac.mean(), 3), flush=True)

    # evaluation sets: balanced by bucket, held-out passages only
    evs = {}
    for name, bucket, grans in (("mix_ai_heavy", "heavy", ["sent", "block", "para"]), ("mix_human_heavy", "light", ["sent", "block", "para"]), ("mix_mid", "mid", ["sent", "block", "para"]), ("mix_word", None, ["word"]), ("mix_cross", None, ["cross"])):
        rows = []
        while len(rows) < a.n_eval:
            u = rng.choice(ev_u); gran = rng.choice(grans); r = make(u, gran, bucket or rng.choice(["heavy", "light", "mid"]), "ev")
            if r: rows.append(dict(text=r[0], label=1, grp=name, model="mixed", attack=gran, domain=hsrc[u], spans=r[1]))
        evs[name] = pd.DataFrame(rows)
    fr_new = []
    if a.frontier:
        fg = pd.concat([pd.read_parquet(x) for x in sorted(glob.glob(f"{LIC}/gen/r5_*.parquet"))]); fg = fg[[len(sp) == 1 and sp[0][0] == 0 and sp[0][1] >= len(t) - 2 for sp, t in zip(fg.spans, fg.text)]]
        fai = {"train": {}, "eval": {}}; fall = {"train": [], "eval": []}; fby = {"train": {}, "eval": {}}
        for r in fg.itertuples():
            sp_ = "eval" if r.split == "eval" else "train"; t = trim_end(r.text)
            if r.uid in hum and len(t) >= 200: fai[sp_].setdefault(r.uid, []).append((t, r.gen_model)); fall[sp_].append((r.uid, t, r.gen_model, r.src)); fby[sp_].setdefault(r.src, []).append((r.uid, t, r.gen_model, r.src))
        print("frontier texts: train", len(fall["train"]), "eval", len(fall["eval"]), fg.groupby("gen_model").size().to_dict(), flush=True)
        def fmake(u, split, rep):
            for _ in range(8):
                r = make(u, rng.choice(GRAN), rng.choice(BUCKET), "tr" if split == "train" else "ev", fai[split], {"tr": fby["train"], "ev": fby["eval"]}, {"tr": fall["train"], "ev": fall["eval"]})
                if r: return dict(text=r[0], spans=r[1], src="mix_sent" if rng.random() < 0.5 else "mix_block")
        for u, cands in fai["train"].items():
            reps = 8 if cands[0][1].startswith("claude") else 2
            for t, m in cands:
                t2 = straighten(t) if rng.random() < 0.3 else t; fr_new.append(dict(text=rewrap(t2, rng) if rng.random() < 0.15 else t2, spans=[[0, len(t2)]], src="fr_ai"))
            fr_new.append(dict(text=hum[u], spans=[], src="fr_human"))
            for _ in range(reps):
                r = fmake(u, "train", reps)
                if r: fr_new.append(r)
        fr_new = pd.DataFrame(fr_new); print("frontier training docs", fr_new.src.value_counts().to_dict(), flush=True)
        rows = [dict(text=hum[u], label=0, grp="frontier_ai", model="human", attack="human", domain=hsrc[u], spans=[]) for u in fai["eval"]]
        rows += [dict(text=t, label=1, grp="frontier_ai", model=m, attack="frontier", domain=hsrc[u], spans=[[0, len(t)]]) for u, cs in fai["eval"].items() for t, m in cs]
        evs["frontier_ai"] = pd.DataFrame(rows)
        mrows = []
        for _ in range(60 * len(fai["eval"])):
            if len(mrows) >= a.n_eval: break
            u = rng.choice(list(fai["eval"])); r = make(u, rng.choice(["sent", "block", "para"]), rng.choice(["heavy", "light", "mid"]), "ev", fai["eval"], {"ev": fby["eval"]}, {"ev": fall["eval"]})
            if r: mrows.append(dict(text=r[0], label=1, grp="frontier_mix", model="mixed", attack="mix", domain=hsrc[u], spans=r[1]))
        evs["frontier_mix"] = pd.DataFrame(mrows); print("frontier eval", {k: len(evs[k]) for k in ("frontier_ai", "frontier_mix")}, flush=True)
    # layout-augmented pure documents from the current training mixture (train) and from the held-out generated sets (eval)
    tr0 = pd.read_parquet(f"{SRC_DIR}/train.parquet"); sci = tr0[tr0.src.isin(SCI)]
    hh = sci[sci.spans.map(len) == 0].sample(a.n_fmt, random_state=1); aa = sci[[len(sp) == 1 and sp[0][1] >= len(t) - 2 for sp, t in zip(sci.spans, sci.text)]].sample(a.n_fmt, random_state=2)
    fmt = pd.concat([pd.DataFrame(dict(text=[rewrap(t, rng) for t in hh.text], spans=[[] for _ in range(len(hh))], src="fmt_human")), pd.DataFrame(dict(text=[rewrap(t, rng) for t in aa.text], spans=aa.spans.tolist(), src="fmt_ai"))])
    ev_gen = pd.concat([pd.read_parquet(f"{SRC_DIR}/eval/{n}.parquet") for n in ("gen_id", "gen_unseen_domain")]); ev_gen = ev_gen[ev_gen.domain.isin({s[4:] for s in SCI}) & ev_gen.attack.isin(["human", "continue", "rewrite", "write", "casual"])]
    ev_gen = ev_gen[[len(sp) == 0 or (len(sp) == 1 and sp[0][1] >= len(t) - 2) for sp, t in zip(ev_gen.spans, ev_gen.text)]].drop_duplicates("text")
    ev_gen = pd.concat([ev_gen[ev_gen.label == 0].head(1200), ev_gen[ev_gen.label == 1].head(1200)]).reset_index(drop=True); ev_gen["text"] = [rewrap(t, rng) for t in ev_gen.text]; ev_gen["grp"] = "fmt_pure"; evs["fmt_pure"] = ev_gen
    os.makedirs(f"{OUT}/eval", exist_ok=True)
    for f in os.listdir(f"{SRC_DIR}/eval"): shutil.copy(f"{SRC_DIR}/eval/{f}", f"{OUT}/eval/{f}")
    for k, v in evs.items(): v.to_parquet(f"{OUT}/eval/{k}.parquet")
    dev0 = pd.read_parquet(f"{SRC_DIR}/dev.parquet"); new = pd.concat([trn[["text", "spans", "src"]], fmt] + ([fr_new[["text", "spans", "src"]]] if a.frontier else []), ignore_index=True); dm = np.random.RandomState(0).rand(len(new)) < 0.01
    pd.concat([tr0, new[~dm]], ignore_index=True).to_parquet(f"{OUT}/train.parquet"); pd.concat([dev0, new[dm]], ignore_index=True).to_parquet(f"{OUT}/dev.parquet")
    print("train docs", len(tr0) + int((~dm).sum()), "dev", len(dev0) + int(dm.sum()), "| eval sets", {k: len(v) for k, v in evs.items()}, flush=True)


if __name__ == "__main__":
    main()
