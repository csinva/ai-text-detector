#!/usr/bin/env python3
"""Build the training mixture and eval sets from the commercially usable data tree (12_licensed_data.py), and nothing else.

Output under $V45_DATA_ROOT/data/aidetect/prep_lic[_full]/:
  train.parquet, dev.parquet   text, spans (AI char intervals [[s,e],...]; pure AI = [[0,len]]), src
  eval/<name>.parquet          text, label (0 human / 1 AI), grp, spans (word-level sets), extras

Held out of training for the out-of-distribution study (RAID): generators gpt4, mistral-chat, cohere; domain abstracts; attacks paraphrase,
homoglyph, article_deletion; plus 8% of RAID source documents (all their variants) for every model. `--full` also trains on the held-out
generators / domain / attacks. Human and AI texts are balanced inside every RAID domain and MAGE domain. Training texts that match an eval
text (hash of the normalized first 400 characters) are removed.
"""
import hashlib, json, os, random, re, sys
import numpy as np, pandas as pd

D = os.path.join(os.environ["V45_DATA_ROOT"], "data/aidetect")
FULL = "--full" in sys.argv
RATIO = float(next((a.split('=')[1] for a in sys.argv if a.startswith('--ratio=')), 1.3))  # AI documents per human document in the generated-text registers
RTAG = '' if RATIO == 1.3 else f'_r{int(round(RATIO * 10))}'
LIC, OUT = f"{D}/licensed", f"{D}/prep_lic{os.environ.get("VER", "3")}{RTAG}{'_full' if FULL else ''}"
os.makedirs(f"{OUT}/eval", exist_ok=True)
H_MODELS, H_DOMAINS, H_ATTACKS = {"gpt4", "mistral-chat", "cohere"}, {"abstracts"}, {"paraphrase", "homoglyph", "article_deletion"}
H_GENS, H_SRC = {"olmo2_7b", "phi4mini", "qwen3_30b"}, {"foodista", "uk_hansard"}  # generated-text study: generators and registers held out of the held-out model


def nh(t):
    return hashlib.md5(re.sub(r"[^a-z0-9]", "", str(t).lower())[:400].encode()).hexdigest()


def pure(df, label_col="y"):
    df = df.copy()
    df["spans"] = [[[0, len(t)]] if y else [] for t, y in zip(df.text, df[label_col])]
    return df


def balanced_ratio(df, key, ratio=1.3, seed=0):
    """per group: all human texts, and at most `ratio` times as many AI texts (so round-2 AI texts are not thrown away)."""
    out = []
    for _, g in df.groupby(key):
        h, a = g[g.y == 0], g[g.y == 1]
        if len(h) and len(a): out += [h, a.sample(min(len(a), int(ratio * len(h))), random_state=seed)]
    return pd.concat(out)


def balanced_by(df, key, seed=0):
    out = []
    for _, g in df.groupby(key):
        k = min((g.y == 0).sum(), (g.y == 1).sum())
        if k:
            out += [g[g.y == 0].sample(k, random_state=seed), g[g.y == 1].sample(k, random_state=seed)]
    return pd.concat(out)


SENT = re.compile(r"[^\n.!?]+[.!?]*\s*|\n+")


def sentences(t):
    return [m.group(0) for m in SENT.finditer(t) if m.group(0).strip()]


def splice(h_sents, a_sents, r):
    """Mix a human and an AI rendering of the same prompt: runs of sentences alternate between the two."""
    k = r.choice([1, 1, 2, 2, 3, 4, 6, 8])
    cuts = sorted(r.sample(range(1, 100), k))
    edges = [0] + [c / 100 for c in cuts] + [1]
    first_ai = r.random() < 0.5
    out, spans, pos = [], [], 0
    for i in range(len(edges) - 1):
        ai = (i % 2 == 0) == first_ai
        src = a_sents if ai else h_sents
        lo, hi = int(round(edges[i] * len(src))), int(round(edges[i + 1] * len(src)))
        seg = "".join(src[lo:hi])
        if not seg:
            continue
        if out and not out[-1][-1:].isspace() and not seg[:1].isspace():
            out[-1] += " "; pos += 1
        out.append(seg)
        if ai:
            if spans and spans[-1][1] == pos:
                spans[-1][1] = pos + len(seg)
            else:
                spans.append([pos, pos + len(seg)])
        pos += len(seg)
    return "".join(out), spans


def spans_of(df):
    return [[[x["start"], x["end"]] for x in l if x["label"] == "ai"] for l in df.spans]


CURLY = str.maketrans({"\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"', "\u2014": "-", "\u2013": "-"})
STRAIGHT_END = re.compile(r"[.!?][\"'\u201d\u2019)\]]*\s*$")


def straighten(t):
    """curly quotes and dashes to ASCII, one character for one character, so span offsets stay valid."""
    return t.translate(CURLY)


def curl(t):
    """ASCII quotes to curly quotes, one character for one character."""
    out, open_ = [], True
    for i, c in enumerate(t):
        if c == "'": out.append("\u2019" if i > 0 and t[i - 1].isalpha() else "\u2018")
        elif c == '"': out.append("\u201c" if open_ else "\u201d"); open_ = not open_
        else: out.append(c)
    return "".join(out)


def trim_end(t):
    """cut a trailing sentence fragment so human passages end like the generated texts do (a passage that stops mid-sentence is an easy giveaway)."""
    t = t.rstrip()
    if STRAIGHT_END.search(t): return t
    i = max(t.rfind(c) for c in ".!?")
    return t[:i + 1] if i > 0.5 * len(t) else t


def load_gen():
    """human passages (licensed Common Pile subsets) and the AI texts generated from them (14_generate.py); eval passages = 8% by hash."""
    h = pd.read_parquet(f"{LIC}/human_cp.parquet"); h["uid"] = [f"{s}:{i}" for i, s in zip(h.index, h.src)]
    h["text"] = h.text.map(trim_end); h = h[h.text.str.len() >= 300]
    g = pd.concat([pd.read_parquet(f"{LIC}/gen/{f}") for f in sorted(os.listdir(f"{LIC}/gen")) if f.endswith(".parquet") and "smoke" not in f])
    ev = lambda u: int(hashlib.md5(u.encode()).hexdigest(), 16) % 100 < 8
    h["evalg"] = h.uid.map(ev); g["evalg"] = g.uid.map(ev)
    return h, g


def main():
    ev, cnt = {}, {}
    raid = pd.read_parquet(f"{LIC}/raid.parquet").rename(columns={"generation": "text"})
    raid["y"] = (raid.model != "human").astype(int)
    raid["held_grp"] = raid.source_id.map(lambda s: int(hashlib.md5(s.encode()).hexdigest(), 16) % 100 < 8)
    # ---------------- eval sets ----------------
    e = raid[raid.held_grp]
    hum_row = e.y == 0
    seen_m = ~e.model.isin(H_MODELS); seen_d = ~e.domain.isin(H_DOMAINS); seen_a = ~e.attack.isin(H_ATTACKS)
    um = e.model.isin(H_MODELS)
    sets = {"raid_id": e[seen_d & seen_a & (seen_m | hum_row)], "raid_unseen_model": e[seen_d & seen_a & (um | hum_row)],
            "raid_unseen_domain": e[~seen_d & seen_a & (seen_m | hum_row)], "raid_unseen_attack": e[seen_d & ~seen_a & (seen_m | hum_row)],
            "raid_unseen_all": e[~seen_d & ~seen_a & (um | hum_row)]}
    for k, v in sets.items():
        ev[k] = pd.concat([g.pipe(lambda g: pd.concat([g[g.y == 0].sample(min(len(g[g.y == 0]), len(g[g.y == 1]), 1000), random_state=1),
                                                       g[g.y == 1].sample(min(len(g[g.y == 0]), len(g[g.y == 1]), 1000), random_state=1)])) for _, g in v.groupby("domain")]
                          ).assign(grp=k).rename(columns={"y": "label"})[["text", "label", "grp", "model", "domain", "attack"]]
    atk = e[seen_d & (seen_m | um | hum_row)]
    ev["raid_attacks"] = pd.concat([balanced_by(g, "domain", 2).sample(frac=1, random_state=2).head(600) for _, g in atk.groupby("attack")]).rename(columns={"y": "label"}).assign(grp="raid_attacks")[["text", "label", "model", "domain", "attack", "grp"]]
    hum = e[hum_row & (e.attack == "none")].set_index("source_id").text
    mac = e[(e.y == 1) & (e.attack == "none") & (um | e.model.isin(["chatgpt", "gpt3"]))]
    mac = mac[mac.source_id.isin(hum.index)]
    r = random.Random(5); pairs = []
    for row in mac.sample(min(3000, len(mac)), random_state=3).itertuples():
        t, sp = splice(sentences(hum[row.source_id]), sentences(row.text), r)
        if sp and len(t) > 200:
            pairs.append(dict(text=t, label=1, grp="raid_splice", spans=sp, model=row.model, domain=row.domain))
    ev["raid_splice"] = pd.DataFrame(pairs)

    gh, gg = load_gen()
    he, ge = gh[gh.evalg], gg[gg.evalg]

    def gen_set(name, hmask, gmask, mixed=False, cap=2000):
        hh, aa = he[hmask(he)], ge[gmask(ge)]
        if mixed: aa = aa[aa.task == "continue"]
        k = min(cap, len(hh), len(aa))
        hh, aa = hh.sample(k, random_state=1), aa.sample(k, random_state=1)
        df = pd.concat([pd.DataFrame(dict(text=hh.text, label=0, grp=name, model="human", attack="human", domain=hh.src, spans=[[] for _ in range(len(hh))])),
                        pd.DataFrame(dict(text=aa.text, label=1, grp=name, model=aa.gen_model, attack=aa.task, domain=aa.src, spans=aa.spans.tolist()))])
        ev[name] = df.reset_index(drop=True)
    R1 = lambda x: x.task.isin(["continue", "rewrite", "write", "casual"])
    gen_set("gen_id", lambda x: ~x.src.isin(H_SRC), lambda x: R1(x) & ~x.gen_model.isin(H_GENS) & ~x.src.isin(H_SRC))
    gen_set("gen_unseen_gen", lambda x: ~x.src.isin(H_SRC), lambda x: R1(x) & x.gen_model.isin(H_GENS) & ~x.src.isin(H_SRC))
    gen_set("gen_unseen_domain", lambda x: x.src.isin(H_SRC), lambda x: R1(x) & x.src.isin(H_SRC) & ~x.gen_model.isin(H_GENS))
    gen_set("gen_mixed", lambda x: ~x.src.isin(H_SRC), lambda x: ~x.src.isin(H_SRC), mixed=True, cap=1500)
    # round 2: humanizer-style rewrites, sentence-level edits, and the newest generators (the held-out model never trains on qwen3_30b)
    gen_set("gen_humanized", lambda x: ~x.src.isin(H_SRC), lambda x: x.task.str.startswith("humanize_") & ~x.src.isin(H_SRC), cap=1500)
    gen_set("gen_edit", lambda x: ~x.src.isin(H_SRC), lambda x: (x.task == "edit") & ~x.src.isin(H_SRC), cap=2000)
    gen_set("gen_newgen", lambda x: ~x.src.isin(H_SRC), lambda x: R1(x) & x.gen_model.isin(["gptoss20b", "qwen3_30b"]) & ~x.src.isin(H_SRC), cap=1500)

    m_te = pd.read_parquet(f"{LIC}/mage_squad_test.parquet"); m_te["label"] = 1 - m_te.label  # MAGE: 1 = human
    ev["mage_squad_test"] = m_te.assign(grp="mage_squad_test").rename(columns={"src": "model"})[["text", "label", "grp", "model"]]
    ts = []
    for f in sorted(os.listdir(f"{LIC}/textsight")):
        x = pd.read_json(f"{LIC}/textsight/{f}", lines=True)
        ts.append(pd.DataFrame(dict(text=x.text, label=(x.label == "ai").astype(int), grp="textsight_2026", model=x.source)))
    ev["textsight_2026"] = pd.concat(ts)
    sp = pd.read_parquet(f"{LIC}/spans_test.parquet")
    ev["spans_test"] = pd.DataFrame(dict(text=sp.text, label=(sp.kind != "human_control").astype(int), grp="spans_test", spans=spans_of(sp), model=sp.reported_model.fillna("human")))
    w = pd.read_parquet(f"{LIC}/wasit.parquet")  # human / ai / humanized; every licence CC-BY-4.0 or MIT
    ev["wasit_2026"] = pd.DataFrame(dict(text=w.text, label=(w.label != "human").astype(int), grp="wasit_2026",
                                         model=[("human" if l == "human" else g) for l, g in zip(w.label, w.group)], kind=w.label.values,
                                         zerogpt=w.zerogpt_ai_pct.values, pangram=w.pangram_ai_score.values, gptzero=w.gptzero_label.values, tropa3=w.tropa3_score.values))
    ev_hash = set()
    for k, v in ev.items():
        v = v.reset_index(drop=True); v["text"] = v.text.astype(str)
        v.to_parquet(f"{OUT}/eval/{k}.parquet"); cnt[k] = len(v)
        ev_hash |= {nh(t) for t in v.text}
    print("eval sets", cnt, flush=True)

    # ---------------- training mixture ----------------
    parts = []
    tr = raid[~raid.held_grp] if FULL else raid[~raid.held_grp & ~raid.model.isin(H_MODELS) & ~raid.domain.isin(H_DOMAINS) & ~raid.attack.isin(H_ATTACKS)]
    parts.append(pure(balanced_by(tr, "domain", 3)).assign(src="raid"))
    hum_t = tr[(tr.y == 0) & (tr.attack == "none")].set_index("source_id").text
    mac_t = tr[(tr.y == 1) & (tr.attack == "none")]
    mac_t = mac_t[mac_t.source_id.isin(hum_t.index)]
    r = random.Random(11); rows = []
    for row in mac_t.sample(min(100000, len(mac_t)), random_state=4).itertuples():
        t, spn = splice(sentences(hum_t[row.source_id]), sentences(row.text), r)
        if spn and len(t) > 150:
            rows.append(dict(text=t, spans=spn, src="raid_splice"))
    parts.append(pd.DataFrame(rows))
    gt = gg[~gg.evalg]; ht = gh[~gh.evalg]
    if not FULL:
        gt, ht = gt[~gt.gen_model.isin(H_GENS) & ~gt.src.isin(H_SRC)], ht[~ht.src.isin(H_SRC)]
    def trim_ai(t, sp):  # generations cut at the token limit end mid-sentence: trim them like the human passages
        t2 = trim_end(t)
        return t2, [[a, min(b, len(t2))] for a, b in sp if a < len(t2) - 20]
    tr_ai = [trim_ai(t, sp) for t, sp in zip(gt.text, gt.spans)]
    gtr = pd.concat([pd.DataFrame(dict(text=ht.text, spans=[[] for _ in range(len(ht))], src="gen_" + ht.src, y=0)),
                     pd.DataFrame(dict(text=[t for t, _ in tr_ai], spans=[sp for _, sp in tr_ai], src="gen_" + gt.src, y=1))])
    gtr = gtr[(gtr.y == 0) | (gtr.spans.map(len) > 0)]
    gtr = balanced_ratio(gtr, "src", RATIO, 7); ra = random.Random(21)
    gtr["text"] = [straighten(t) if (y == 1 and ra.random() < 0.55) else t for t, y in zip(gtr.text, gtr.y)]  # typography is not a label
    parts.append(gtr.drop(columns=["y"]))
    mg = pd.concat([pd.read_parquet(f"{LIC}/mage_squad_{s}.parquet") for s in ("train", "valid")]); mg["y"] = 1 - mg.label
    mg["dom"] = "squad"
    parts.append(pure(balanced_by(mg, "dom", 5)).assign(src="mage_squad"))
    s = pd.concat([pd.read_parquet(f"{LIC}/spans_{x}.parquet") for x in ("train", "validation")])
    parts.append(pd.DataFrame(dict(text=s.text, spans=spans_of(s), src="spans")))
    tr_all = pd.concat([p[["text", "spans", "src"]] for p in parts])
    tr_all["text"] = tr_all.text.astype(str)
    n0 = len(tr_all)
    tr_all = tr_all[~tr_all.text.map(nh).isin(ev_hash) & (tr_all.text.str.len() >= 20)].drop_duplicates("text")
    print("decontaminate + dedupe", n0, "->", len(tr_all), flush=True)
    tr_all["spans"] = tr_all.spans.map(lambda l: [list(map(int, x)) for x in l])
    dev_mask = np.random.RandomState(0).rand(len(tr_all)) < 0.01
    tr_all[dev_mask].reset_index(drop=True).to_parquet(f"{OUT}/dev.parquet")
    tr_all[~dev_mask].reset_index(drop=True).to_parquet(f"{OUT}/train.parquet")
    af = tr_all.assign(ai=[sum(e_ - s_ for s_, e_ in sp_) / max(1, len(t)) for t, sp_ in zip(tr_all.text, tr_all.spans)], nchar=tr_all.text.str.len())
    print(af.groupby("src").agg(docs=("ai", "size"), ai_frac=("ai", "mean"), chars=("nchar", "mean")).to_string())
    json.dump(dict(eval=cnt, train=len(tr_all)), open(f"{OUT}/prep_stats.json", "w"))


if __name__ == "__main__":
    main()
