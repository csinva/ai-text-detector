#!/usr/bin/env python3
"""Humanizer benchmark for the comparison detectors, same protocol as 16_eval_humanizers.py: thresholds at 1% / 5% FPR on 2,216 pooled human texts, recall on humanizer outputs and raw LLM inputs.
python 17_baselines_humanizers.py --out <json>"""
import argparse, importlib.util, json, os
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__)); D = os.path.join(os.environ["V45_DATA_ROOT"], "data/aidetect")
spec = importlib.util.spec_from_file_location("bl", f"{HERE}/05_baselines.py"); mod = importlib.util.module_from_spec(spec)
exec(compile(open(f"{HERE}/05_baselines.py").read(), "bl", "exec"), mod.__dict__)
ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); a = ap.parse_args()
H = []
for n in ("wasit_2026", "textsight_2026", "gen_id"):
    df = pd.read_parquet(f"{D}/prep_lic3/eval/{n}.parquet"); df = df[df.label == 0]; H.append(df.sample(min(len(df), 1500), random_state=0).text.tolist())
hum = sum(H, []); d = pd.read_parquet(f"{D}/licensed/humanizer.parquet"); res = {}
for key in mod.REPOS:
    tok, m, post = mod.load(key); hp = mod.score(tok, m, post, hum); d["s"] = mod.score(tok, m, post, d.text.tolist())
    thr = {f: float(np.quantile(hp, 1 - f)) for f in (0.01, 0.05)}; o, i = d[d.kind == "output"], d[d.kind == "input"]
    row = lambda g: {**{f"recall@{int(f * 100)}": float((g.s > t).mean()) for f, t in thr.items()}, "auc": float(roc_auc_score(np.r_[np.zeros(len(hp)), np.ones(len(g))], np.r_[hp, g.s.values])), "n": len(g)}
    res[key] = {"thr": thr, "outputs": row(o), "inputs": row(i), "by_humanizer": {h: row(g) for h, g in o.groupby("humanizer")}}
    print(key, {k: round(v, 3) for k, v in res[key]["outputs"].items()}, "inputs", {k: round(v, 3) for k, v in res[key]["inputs"].items()}, flush=True)
json.dump(res, open(a.out, "w"), indent=1)
