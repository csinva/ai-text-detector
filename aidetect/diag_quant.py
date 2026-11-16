"""Which linear layers can be int8-quantized? Compare logits and AUROC against the fp32 CPU model on a sample.
python diag_quant.py <model.pt> [set] [n] [threads]"""
import os, sys, time
import numpy as np, pandas as pd, torch
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.dirname(__file__))
from infer import Predictor

D = os.path.join(os.environ["V45_DATA_ROOT"], "data/aidetect")
model = sys.argv[1]; name = sys.argv[2] if len(sys.argv) > 2 else "raid_id"; n = int(sys.argv[3]) if len(sys.argv) > 3 else 300
torch.set_num_threads(int(sys.argv[4]) if len(sys.argv) > 4 else 8)
df = pd.read_parquet(f"{D}/prep/eval/{name}.parquet").sample(n, random_state=1)
res = {}
for q in [0, 1, 3, 2]:
    P = Predictor(model, device="cpu", quant=q)
    t0 = time.time(); pr = P.predict(df.text.tolist(), with_words=False); dt = time.time() - t0
    lg = np.array([x["doc_logit"] for x in pr]); res[q] = lg
    msg = f"quant={q}: AUROC {100 * roc_auc_score(df.label, lg):.2f}  FPR@0.5 {100 * (lg[df.label.values == 0] > 0).mean():.1f}  recall@0.5 {100 * (lg[df.label.values == 1] > 0).mean():.1f}  {dt / n:.3f} s/doc"
    if q: msg += f"  logit diff vs fp32: mean {np.mean(lg - res[0]):+.2f}, mean abs {np.mean(np.abs(lg - res[0])):.2f}, corr {np.corrcoef(lg, res[0])[0, 1]:.4f}"
    print(msg, flush=True)
