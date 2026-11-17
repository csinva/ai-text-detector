#!/usr/bin/env python3
"""CPU latency of the detector (one document at a time, document + word probabilities) for several token lengths and threads.

  python 06_cpu_bench.py --model <run>/model.pt [--n_layers 12] [--threads 1,4,8,16] [--out file.json]
"""
import argparse, json, os, sys, time
import numpy as np, pandas as pd, torch

sys.path.insert(0, os.path.dirname(__file__))
from infer import Predictor

D = os.path.join(os.environ["V45_DATA_ROOT"], "data/aidetect")
ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True); ap.add_argument("--n_layers", type=int, default=12)
ap.add_argument("--threads", default="1,4,8,16"); ap.add_argument("--out"); ap.add_argument("--reps", type=int, default=6)
a = ap.parse_args()
df = pd.read_parquet(f"{D}/prep_lic2/eval/spans_test.parquet")
texts = df.text.tolist()
res = []
for variant in ["fp32", "bf16", "int8"]:
    P = Predictor(a.model, device="cpu", n_layers=a.n_layers, dtype=torch.bfloat16 if variant == "bf16" else torch.float32, quant=2 if variant == "int8" else 0)
    for th in map(int, a.threads.split(",")):
        torch.set_num_threads(th)
        for ntok in [128, 512, 1024]:
            docs = []
            for t in texts[:a.reps + 1]:
                ids = P.tk.encode(t).ids[:ntok]; docs.append(P.tk.decode(ids))
            P.predict(docs[:1])  # warmup
            t0 = time.time()
            for d in docs[1:]:
                P.predict([d])
            dt = (time.time() - t0) / a.reps
            res.append(dict(variant=variant, threads=th, tokens=ntok, sec_per_doc=dt, tok_per_s=ntok / dt))
            print(res[-1], flush=True)
if a.out: json.dump(res, open(a.out, "w"), indent=1)
