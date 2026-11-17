"""Round 3: score commercial-humanizer text (licensed/humanizer.parquet) with a detector.
Humans = pooled human texts of the external sets (WasIt, TextSight) and the Common Pile eval split; thresholds at 1% / 5% FPR on that pool.
Reports per bench / humanizer: recall of humanizer outputs at those thresholds, AUROC vs pooled humans, recall on the raw LLM inputs, and the vendor detectors' detection rate (human-ness < 0.5).
python 16_eval_humanizers.py --model runs/<name>/model.pt --out runs/<name>/humanizers.json [--score doc|tok|avg]"""
import argparse, json, os
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

D = os.path.join(os.environ["V45_DATA_ROOT"], "data/aidetect")
ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True); ap.add_argument("--out"); ap.add_argument("--n_layers", type=int, default=12)
ap.add_argument("--score", default="doc"); ap.add_argument("--device", default="cuda"); ap.add_argument("--prep", default="prep_lic3")
a = ap.parse_args()
sys_path = os.path.dirname(os.path.abspath(__file__))
import sys; sys.path.insert(0, sys_path)
from infer import Predictor

P = Predictor(a.model, device=a.device, n_layers=a.n_layers)
sc = {"doc": lambda x: x["doc_prob"], "tok": lambda x: x["tok_prob_mean"], "avg": lambda x: 0.5 * (x["doc_prob"] + x["tok_prob_mean"])}[a.score]
score = lambda texts: np.array([sc(x) for x in P.predict(texts, with_words=False)])

H = []
for n in ("wasit_2026", "textsight_2026", "gen_id"):
    df = pd.read_parquet(f"{D}/{a.prep}/eval/{n}.parquet"); df = df[df.label == 0]
    H.append(df.sample(min(len(df), 1500), random_state=0).text.tolist())
hs = [score(h) for h in H]; hp = np.concatenate(hs)
thr = {f: float(np.quantile(hp, 1 - f)) for f in (0.01, 0.05)}
print("pooled humans", len(hp), "thresholds", thr, "per-source mean", [round(float(x.mean()), 3) for x in hs], flush=True)

d = pd.read_parquet(f"{D}/licensed/humanizer.parquet")
d["s"] = score(d.text.tolist())
vend = [c for c in d.columns if c.startswith("v_")]
res = {"thr": thr, "n_human": len(hp)}
def row(g):
    r = {"n": len(g), **{f"recall@{int(f*100)}": float((g.s > t).mean()) for f, t in thr.items()}, "auc": float(roc_auc_score(np.r_[np.zeros(len(hp)), np.ones(len(g))], np.r_[hp, g.s.values]))}
    for c in vend:
        v = g[c].dropna()
        if len(v): r[c] = float((v < 0.5).mean())
    return r
for bench in ("hb", "ahb", "all"):
    b = d if bench == "all" else d[d.bench == bench]
    o, i = b[b.kind == "output"], b[b.kind == "input"]
    res[bench] = {"outputs": row(o), "inputs": row(i), "by_humanizer": {h: row(g) for h, g in o.groupby("humanizer")}, "by_cycle": {h: row(g) for h, g in o.groupby("cycle")},
                  "by_category": {h: row(g) for h, g in o.groupby("category")}}
    print(bench, {k: round(v, 3) for k, v in res[bench]["outputs"].items()}, "| inputs", {k: round(v, 3) for k, v in res[bench]["inputs"].items() if k.startswith(("recall", "auc"))}, flush=True)
for h, r in sorted(res["all"]["by_humanizer"].items(), key=lambda x: x[1]["recall@1"]): print(f"{h:22s} n={r['n']:4d} r@1={r['recall@1']:.2f} r@5={r['recall@5']:.2f} auc={r['auc']:.3f}")
if a.out: json.dump(res, open(a.out, "w"), indent=1)
