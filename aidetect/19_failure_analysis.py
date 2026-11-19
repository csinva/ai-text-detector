#!/usr/bin/env python3
"""Failure analysis of the detector (round-2 and round-4 models):
  1. old arXiv papers (before 2017) as users paste them: PDF text extracted with pypdf, scored as extracted and after unwrapping the lines
  2. TextSight 2026: recall per generator (Claude Opus 5, Haiku 4.5, GPT-4.1, gpt-oss-120b, Qwen) at the 0.5 threshold and at the threshold of 1% FPR on its human texts, and what the misses look like
  3. false positive rate by human source on held-out passages of the training pools
python 19_failure_analysis.py --out <json>"""
import argparse, glob, json, os, random, re, sys
import numpy as np, pandas as pd
from pypdf import PdfReader

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from infer import Predictor

D = os.path.join(os.environ["V45_DATA_ROOT"], "data/aidetect"); R = f"{D}/runs"
ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--models", default="round2=lic3_nano_r10,round4=lic5_nano_r15", help="name=run,name=run (runs under data/aidetect/runs)"); ap.add_argument("--prep", default="prep_lic5_r15_full"); a = ap.parse_args()
MODELS = {kv.split("=")[0]: f"{R}/{kv.split('=')[1]}/model.pt" for kv in a.models.split(",")}; REF = list(MODELS)[-1]
rng = random.Random(0)

# ---- 1. old arXiv papers
passages = []
for f in sorted(glob.glob(f"{D}/arxiv_old/*.pdf")):
    try: txt = "\n".join((p.extract_text() or "") for p in PdfReader(f).pages)
    except Exception as e: print("skip", f, e); continue
    lines = txt.split("\n"); chunks, cur = [], ""
    for ln in lines:
        cur += ln + "\n"
        if len(cur) > 1400: chunks.append(cur); cur = ""
    chunks = [c for c in chunks[1:] if sum(ch.isalpha() for ch in c) / max(len(c), 1) > 0.55]      # drop the title block and symbol-heavy chunks (tables, equation pages)
    for c in rng.sample(chunks, min(30, len(chunks))): passages.append(dict(paper=os.path.basename(f)[:-4], raw=c.strip(), flat=re.sub(r"\s+", " ", re.sub(r"-\n(?=[a-z])", "", c)).strip()))
arx = pd.DataFrame(passages); print("arXiv passages", len(arx), arx.paper.nunique(), "papers", flush=True)

# ---- 2. TextSight
ts = pd.read_parquet(f"{D}/prep_lic3/eval/textsight_2026.parquet")
# ---- 3. held-out human passages by source (prep_lic5 eval sets built from the pools)
hold = pd.concat([pd.read_parquet(f"{D}/{a.prep}/eval/{n}.parquet") for n in ("gen_id", "gen_unseen_domain", "gen_edit")]); hold = hold[hold.label == 0].drop_duplicates("text")
hold = pd.concat([g.sample(min(len(g), 150), random_state=0) for _, g in hold.groupby("domain")])

out = {}
for name, path in MODELS.items():
    P = Predictor(path, device="cuda"); sc = lambda texts: np.array([x["doc_prob"] for x in P.predict(texts, with_words=False)])
    r = {}
    arx[f"p_raw_{name}"] = sc(arx.raw.tolist()); arx[f"p_flat_{name}"] = sc(arx.flat.tolist())
    r["arxiv"] = dict(n=len(arx), raw_gt05=float((arx[f"p_raw_{name}"] > .5).mean()), flat_gt05=float((arx[f"p_flat_{name}"] > .5).mean()), raw_mean=float(arx[f"p_raw_{name}"].mean()), flat_mean=float(arx[f"p_flat_{name}"].mean()),
                      by_paper=arx.groupby("paper")[[f"p_raw_{name}", f"p_flat_{name}"]].mean().round(3).to_dict("index"))
    ts[f"p_{name}"] = sc(ts.text.tolist()); h = ts[ts.label == 0]; thr1 = float(np.quantile(h[f"p_{name}"], .99))
    r["textsight"] = dict(thr_1pct=thr1, by_model={m: dict(n=len(g), mean=float(g[f"p_{name}"].mean()), recall05=float((g[f"p_{name}"] > .5).mean()), recall1=float((g[f"p_{name}"] > thr1).mean())) for m, g in ts[ts.label == 1].groupby("model")},
                          human_fpr05={m: float((g[f"p_{name}"] > .5).mean()) for m, g in ts[ts.label == 0].groupby("model")})
    hold[f"p_{name}"] = sc(hold.text.tolist()); r["heldout_human_fpr05"] = {d_: dict(n=len(g), fpr05=float((g[f"p_{name}"] > .5).mean()), mean=float(g[f"p_{name}"].mean())) for d_, g in hold.groupby("domain")}
    out[name] = r; print(name, json.dumps({k: v for k, v in r["arxiv"].items() if k != "by_paper"}), flush=True)
    print(name, "textsight", {m: (round(v["mean"], 2), round(v["recall05"], 2), round(v["recall1"], 2)) for m, v in r["textsight"]["by_model"].items()}, flush=True)
    print(name, "held-out human FPR@0.5 by source", {k: round(v["fpr05"], 3) for k, v in r["heldout_human_fpr05"].items()}, flush=True)
# examples (public papers and the CC BY TextSight set, safe to print)
top = arx.sort_values(f"p_raw_{REF}", ascending=False).head(8)
out["arxiv_top_flagged"] = [dict(paper=x.paper, p=float(getattr(x, f"p_raw_{REF}")), p_flat=float(getattr(x, f"p_flat_{REF}")), snippet=x.raw[:260].replace("\n", " | ")) for x in top.itertuples()]
miss = ts[(ts.label == 1) & (ts.model.str.contains("opus|haiku"))].sort_values(f"p_{REF}").head(6)
out["textsight_missed_claude"] = [dict(model=x.model, p=float(getattr(x, f"p_{REF}")), snippet=x.text[:360].replace("\n", " | ")) for x in miss.itertuples()]
# simple surface statistics: do the misses differ from the caught AI texts?
feat = lambda t: dict(chars=len(t), emdash=t.count("—"), endash=t.count("–"), colon=t.count(":"), bullets=len(re.findall(r"^\s*(?:[-*•]|\d+\.)\s", t, flags=re.M)), md=len(re.findall(r"\*\*|^#+ ", t, flags=re.M)), nl=t.count("\n"), curly=len(re.findall("[‘’“”]", t)))
f = pd.DataFrame([feat(t) for t in ts.text]); f["label"] = ts.label.values; f["model"] = ts.model.values; f["caught"] = (ts[f"p_{REF}"] > ts[f"p_{REF}"][ts.label == 0].quantile(.99)).values
out["textsight_features"] = {k: f[f.label == 1].groupby(["model"])[["chars", "emdash", "colon", "bullets", "md", "nl", "curly"]].mean().round(2).to_dict("index") for k in ("AI",)}
out["textsight_features"]["human"] = f[f.label == 0][["chars", "emdash", "colon", "bullets", "md", "nl", "curly"]].mean().round(2).to_dict()
json.dump(out, open(a.out, "w"), indent=1, default=float); arx.drop(columns=["raw", "flat"]).to_csv(a.out.replace(".json", "_arxiv_scores.csv"), index=False)
for x in out["arxiv_top_flagged"][:5]: print("ARXIV FLAGGED", round(x["p"], 2), round(x["p_flat"], 2), x["paper"], x["snippet"][:200])
for x in out["textsight_missed_claude"][:4]: print("MISSED", x["model"], round(x["p"], 2), x["snippet"][:240])
