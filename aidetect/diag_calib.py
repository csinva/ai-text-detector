"""Calibration diagnosis on CPU: mean P(AI) and false positive rate by (set, group, label) for a sample of each eval set.
python diag_calib.py <model.pt> <set> [group_col] [n]"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from infer import Predictor

D = os.path.join(os.environ["V45_DATA_ROOT"], "data/aidetect")
model, name = sys.argv[1], sys.argv[2]
col = sys.argv[3] if len(sys.argv) > 3 else "model"
n = int(sys.argv[4]) if len(sys.argv) > 4 else 600
df = pd.read_parquet(f"{D}/prep_lic3/eval/{name}.parquet").sample(n, random_state=0)
P = Predictor(model, device="cpu")
pr = P.predict(df.text.tolist(), with_words=False)
df["p"] = [x["doc_prob"] for x in pr]; df["ntok"] = [x["n_tokens"] for x in pr]; df["logit"] = [x["doc_logit"] for x in pr]
df["fp_or_tp"] = df.p > 0.5
g = df.groupby([col, "label"]).agg(n=("p", "size"), mean_p=("p", "mean"), frac_pos=("fp_or_tp", "mean"), med_logit=("logit", "median"), ntok=("ntok", "median"))
print(g.round(3).to_string())
