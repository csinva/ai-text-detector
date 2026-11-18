#!/usr/bin/env python3
"""Compare the int8 ONNX model (windowing and aggregation as in the JS app) with the fp32 torch Predictor on evaluation texts.
python app/tools/verify_onnx.py --model <run>/model.pt [--onnx app/model/detector.onnx] [--n 60]"""
import argparse, os, sys, time
import numpy as np, pandas as pd, onnxruntime as ort
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "..", "aidetect"))
from infer import Predictor, token_to_word, windows  # noqa: E402

D = os.path.join(os.environ["V45_DATA_ROOT"], "data/aidetect")
ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--onnx", default=os.path.join(HERE, "..", "model", "detector.onnx")); ap.add_argument("--n", type=int, default=60); a = ap.parse_args()
P = Predictor(a.model, device="cpu", dtype=None, quant=0)
sess = ort.InferenceSession(a.onnx, providers=["CPUExecutionProvider"])


def run_onnx(text):
    enc = P.tk.encode(text); ids = np.array(enc.ids, dtype=np.int64); n = len(ids)
    tok_sum = np.zeros(n); tok_cnt = np.zeros(n); dl = 0.0; dw = 0
    for s, e in windows(n, 1024):
        t, d = sess.run(None, {"ids": ids[s:e]}); tok_sum[s:e] += 1 / (1 + np.exp(-t)); tok_cnt[s:e] += 1; dl += float(d) * (e - s); dw += e - s
    tp = tok_sum / np.maximum(tok_cnt, 1)
    return dict(doc_prob=1 / (1 + np.exp(-dl / dw)), doc_logit=dl / dw, tok_prob_mean=float(tp.mean()), words=token_to_word(text, enc.offsets, tp))


rows = []
for name in ("textsight_2026", "wasit_2026", "spans_test", "gen_mixed"):
    df = pd.read_parquet(f"{D}/prep_lic3/eval/{name}.parquet"); df = df.sample(min(a.n, len(df)), random_state=0)
    t0 = time.time(); ref = P.predict(df.text.tolist(), with_words=True); t1 = time.time(); got = [run_onnx(t) for t in df.text]; t2 = time.time()
    dl = np.array([r["doc_logit"] for r in ref]); dg = np.array([r["doc_logit"] for r in got])
    wr = np.concatenate([[w[2] for w in r["words"]] for r in ref]); wg = np.concatenate([[w[2] for w in r["words"]] for r in got])
    out = dict(set=name, n=len(df), doc_logit_maxabs=float(np.abs(dl - dg).max()), doc_logit_mae=float(np.abs(dl - dg).mean()), doc_corr=float(np.corrcoef(dl, dg)[0, 1]), word_mae=float(np.abs(wr - wg).mean()), word_corr=float(np.corrcoef(wr, wg)[0, 1]),
               torch_s=t1 - t0, onnx_s=t2 - t1)
    if df.label.nunique() > 1: out.update(auroc_torch=100 * roc_auc_score(df.label, [r["doc_prob"] for r in ref]), auroc_onnx=100 * roc_auc_score(df.label, [r["doc_prob"] for r in got]))
    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in out.items()}, flush=True)
