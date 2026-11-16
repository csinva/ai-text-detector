#!/usr/bin/env python3
"""Build the AI-text-detector training mixture and eval sets from the raw open datasets.

Output (all under $V45_DATA_ROOT/data/aidetect/prep/):
  train.parquet, dev.parquet   text, spans (AI char intervals [[s,e],...]; pure AI = [[0,len]]), src
  eval/<name>.parquet          text, label (0 human / 1 AI, doc level), grp, spans (word-level sets), extra

Held out of training for out-of-distribution evals (RAID): generators gpt4, mistral-chat, cohere; domains recipes,
reviews; attacks paraphrase, homoglyph, article_deletion; plus 8% of source documents (all their variants).
All training texts are decontaminated against every eval text (hash of normalized first 400 chars).
"""
import hashlib, json, os, random, re, sys
import numpy as np, pandas as pd

D = os.path.join(os.environ["V45_DATA_ROOT"], "data/aidetect")
FULL = "--full" in sys.argv  # final model: also train on the held-out RAID generators/domains/attacks (benchmarks stay held out)
BAL = "--bal" in sys.argv  # balance human / AI inside every source domain (MAGE domain, M4GT sub-source) so the domain cannot predict the label
RAW, OUT = f"{D}/raw", f"{D}/prep{'_full' if FULL else ''}{'_bal' if BAL else ''}"
os.makedirs(f"{OUT}/eval", exist_ok=True)
H_MODELS, H_DOMAINS, H_ATTACKS = {"gpt4", "mistral-chat", "cohere"}, {"recipes", "reviews"}, {"paraphrase", "homoglyph", "article_deletion"}


def nh(t):
    return hashlib.md5(re.sub(r"[^a-z0-9]", "", t.lower())[:400].encode()).hexdigest()


def pure(df, label_col="y"):
    df = df.copy()
    df["spans"] = [[[0, len(t)]] if y else [] for t, y in zip(df.text, df[label_col])]
    return df


def balanced_by(df, key, seed=0):
    """per group: as many human as AI texts (the smaller class decides)."""
    out = []
    for _, g in df.groupby(key):
        k = min((g.y == 0).sum(), (g.y == 1).sum())
        if k:
            out += [g[g.y == 0].sample(k, random_state=seed), g[g.y == 1].sample(k, random_state=seed)]
    return pd.concat(out)


def balanced(df, n, seed=0):
    h, m = df[df.y == 0], df[df.y == 1]
    k = min(n // 2, len(h), len(m))
    return pd.concat([h.sample(k, random_state=seed), m.sample(k, random_state=seed)])


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


def main():
    ev, cnt = {}, {}
    raid = pd.read_parquet(f"{D}/raid_train.parquet").rename(columns={"generation": "text"})
    raid["y"] = (raid.model != "human").astype(int)
    raid["held_grp"] = raid.source_id.map(lambda s: int(hashlib.md5(s.encode()).hexdigest(), 16) % 100 < 8)
    # ---------------- eval sets ----------------
    e = raid[raid.held_grp]
    hum_row = e.y == 0
    seen_m = ~e.model.isin(H_MODELS); seen_d = ~e.domain.isin(H_DOMAINS); seen_a = ~e.attack.isin(H_ATTACKS)
    um = e.model.isin(H_MODELS)
    sets = {
        "raid_id": e[seen_d & seen_a & (seen_m | hum_row)],
        "raid_unseen_model": e[seen_d & seen_a & (um | hum_row)],
        "raid_unseen_domain": e[~seen_d & seen_a & (seen_m | hum_row)],
        "raid_unseen_attack": e[seen_d & ~seen_a & (seen_m | hum_row)],
        "raid_unseen_all": e[~seen_d & ~seen_a & (um | hum_row)],
    }
    for k, v in sets.items():
        ev[k] = balanced(v, 6000, 1).assign(grp=k).rename(columns={"y": "label"})[["text", "label", "grp", "model", "domain", "attack"]]
    atk = e[seen_d & (seen_m | um | hum_row)]
    ev["raid_attacks"] = pd.concat([balanced(g, 600, 2) for _, g in atk.groupby("attack")]).rename(columns={"y": "label"}).assign(grp="raid_attacks")[["text", "label", "grp", "model", "domain", "attack"]]
    hum = e[hum_row & (e.attack == "none")].set_index("source_id").text
    mac = e[(e.y == 1) & (e.attack == "none") & (um | e.model.isin(["chatgpt", "gpt3"]))]
    mac = mac[mac.source_id.isin(hum.index)]
    r = random.Random(5); pairs = []
    for row in mac.sample(min(3000, len(mac)), random_state=3).itertuples():
        t, sp = splice(sentences(hum[row.source_id]), sentences(row.text), r)
        if sp and len(t) > 200:
            pairs.append(dict(text=t, label=1, grp="raid_splice", spans=sp, model=row.model, domain=row.domain))
    ev["raid_splice"] = pd.DataFrame(pairs)

    mage_te = pd.read_csv(f"{RAW}/mage/test.csv"); mage_te["label"] = 1 - mage_te.label  # MAGE: 1 = human
    ev["mage_test"] = mage_te.sample(12000, random_state=0).assign(grp="mage_test").rename(columns={"src": "model"})[["text", "label", "grp", "model"]]
    for f, n in [("test_ood_set_gpt", "mage_ood_gpt"), ("test_ood_set_gpt_para", "mage_ood_gpt_para")]:
        x = pd.read_csv(f"{RAW}/mage/{f}.csv"); x["label"] = 1 - x.label
        ev[n] = x.assign(grp=n)[["text", "label", "grp"]]
    sm = pd.read_parquet(f"{RAW}/semeval8/subtaskA_monolingual/test-00000-of-00001.parquet")
    ev["semeval_a_test"] = sm.sample(12000, random_state=0).assign(grp="semeval_a_test")[["text", "label", "grp", "model", "domain"]]
    hc = pd.concat([pd.read_json(f"{RAW}/hc3/{f}.jsonl", lines=True) for f in ["finance", "medicine", "open_qa", "reddit_eli5", "wiki_csai"]])
    rows = [(a, 0) for l in hc.human_answers for a in l] + [(a, 1) for l in hc.chatgpt_answers for a in l]
    hcd = pd.DataFrame(rows, columns=["text", "y"]); hcd = hcd[hcd.text.str.len() > 40]
    ev["hc3"] = balanced(hcd, 8000, 0).rename(columns={"y": "label"}).assign(grp="hc3")
    ts = []
    for f in sorted(os.listdir(f"{RAW}/textsight/data")):
        x = pd.read_json(f"{RAW}/textsight/data/{f}", lines=True)
        ts.append(pd.DataFrame(dict(text=x.text, label=(x.label == "ai").astype(int), grp="textsight_2026", model=x.source)))
    ev["textsight_2026"] = pd.concat(ts)
    sp = pd.read_parquet(f"{RAW}/spans/data/all/test.parquet")
    ev["spans_test"] = pd.DataFrame(dict(text=sp.text, label=(sp.kind != "human_control").astype(int), grp="spans_test",
                                          spans=spans_of(sp), model=sp.reported_model.fillna("human")))
    for sp_name in ["dev", "test"]:
        c = pd.read_parquet(f"{RAW}/semeval8/subtaskC/{sp_name}-00000-of-00001.parquet").rename(columns={"label": "boundary"})
        c["label"] = 1; spans = []
        for t, b in zip(c.text, c.boundary):
            ws = list(re.finditer(r"\S+", t)); spans.append([[ws[b].start(), len(t)]] if b < len(ws) else [])
        c["spans"] = spans; c["grp"] = f"semeval_c_{sp_name}"
        ev[f"semeval_c_{sp_name}"] = c[["text", "label", "grp", "boundary", "spans"]]
    ro = pd.read_csv(f"{RAW}/roft/roft.csv").query("model != 'human'").drop_duplicates(["prompt_body", "gen_body"])
    rr = []
    for x in ro.itertuples():
        ps = [s for s in str(x.prompt_body).split("_SEP_") if s]; gs = [s for s in str(x.gen_body).split("_SEP_") if s]
        t = " ".join(ps + gs)[:6000]; b = len(" ".join(ps)) + 1
        if b < len(t):
            rr.append(dict(text=t, label=1, grp="roft", spans=[[b, len(t)]], model=x.model))
    ev["roft"] = pd.DataFrame(rr).sample(4000, random_state=0)
    ev_hash = set()
    for k, v in ev.items():
        v = v.reset_index(drop=True); v["text"] = v.text.astype(str)
        v.to_parquet(f"{OUT}/eval/{k}.parquet"); cnt[k] = len(v)
        ev_hash |= {nh(t) for t in v.text}
    print("eval sets", cnt, flush=True)

    # ---------------- training mixture ----------------
    parts = []
    tr = raid[~raid.held_grp] if FULL else raid[~raid.held_grp & ~raid.model.isin(H_MODELS) & ~raid.domain.isin(H_DOMAINS) & ~raid.attack.isin(H_ATTACKS)]
    parts.append(pure(balanced(tr, 2 * int((tr.y == 0).sum()), 3)).assign(src="raid"))
    hum_t = tr[(tr.y == 0) & (tr.attack == "none")].set_index("source_id").text
    mac_t = tr[(tr.y == 1) & (tr.attack == "none")]
    mac_t = mac_t[mac_t.source_id.isin(hum_t.index)]
    r = random.Random(11); rows = []
    for row in mac_t.sample(min(150000, len(mac_t)), random_state=4).itertuples():
        t, spn = splice(sentences(hum_t[row.source_id]), sentences(row.text), r)
        if spn and len(t) > 150:
            rows.append(dict(text=t, spans=spn, src="raid_splice"))
    parts.append(pd.DataFrame(rows))
    mg = pd.read_csv(f"{RAW}/mage/train.csv"); mg["y"] = 1 - mg.label
    mg["dom"] = mg.src.str.replace("_human", "").str.replace(r"_machine.*", "", regex=True)
    parts.append(pure(balanced_by(mg, "dom", 5) if BAL else balanced(mg, 2 * int((mg.y == 0).sum()), 5)).assign(src="mage"))
    col = pd.concat([pd.read_parquet(f"{RAW}/coling_en/data/{f}") for f in ["dev-00000-of-00001.parquet", "train-00000-of-00002.parquet", "train-00001-of-00002.parquet"]])
    col = col[col.source == "m4gt"].assign(y=lambda x: (x.model != "human").astype(int))
    parts.append(pure(balanced_by(col, "sub_source", 6) if BAL else balanced(col, 2 * int((col.y == 0).sum()), 6)).assign(src="m4gt"))
    s = pd.read_parquet(f"{RAW}/spans/data/all/train.parquet")
    parts.append(pd.DataFrame(dict(text=s.text, spans=spans_of(s), src="spans")))
    c = pd.read_parquet(f"{RAW}/semeval8/subtaskC/train-00000-of-00001.parquet"); rows = []
    for t, b in zip(c.text, c.label):
        ws = list(re.finditer(r"\S+", t))
        if b < len(ws): rows.append(dict(text=t, spans=[[ws[b].start(), len(t)]], src="semeval_c"))
    parts.append(pd.DataFrame(rows))
    sv = pd.read_parquet(f"{RAW}/spans/data/all/validation.parquet")  # spans validation also trains (test stays held out)
    parts.append(pd.DataFrame(dict(text=sv.text, spans=spans_of(sv), src="spans")))
    tr_all = pd.concat([p[["text", "spans", "src"]] for p in parts])
    tr_all["text"] = tr_all.text.astype(str)
    n0 = len(tr_all)
    tr_all = tr_all[~tr_all.text.map(nh).isin(ev_hash) & (tr_all.text.str.len() >= 20)].drop_duplicates("text")
    print("decontaminate + dedupe", n0, "->", len(tr_all), flush=True)
    tr_all["spans"] = tr_all.spans.map(lambda l: [list(map(int, x)) for x in l])
    dev_mask = np.random.RandomState(0).rand(len(tr_all)) < 0.01
    tr_all[dev_mask].reset_index(drop=True).to_parquet(f"{OUT}/dev.parquet")
    tr_all[~dev_mask].reset_index(drop=True).to_parquet(f"{OUT}/train.parquet")
    af = tr_all.assign(ai=[sum(e_ - s_ for s_, e_ in sp_) / max(1, len(t)) for t, sp_ in zip(tr_all.text, tr_all.spans)],
                       nchar=tr_all.text.str.len())
    print(af.groupby("src").agg(docs=("ai", "size"), ai_frac=("ai", "mean"), chars=("nchar", "mean")).to_string())
    json.dump(dict(eval=cnt, train=len(tr_all)), open(f"{OUT}/prep_stats.json", "w"))


if __name__ == "__main__":
    main()
