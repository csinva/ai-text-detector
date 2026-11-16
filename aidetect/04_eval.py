#!/usr/bin/env python3
"""Evaluate a detector on the held-out open benchmarks built by 01_prep.py.

Document level (all sets with pure labels): AUROC, TPR at 1% / 5% FPR (threshold set on the human texts of the same set),
accuracy and F1 at the fixed threshold 0.5 (no tuning on test data).
Word level (spans_test, raid_splice, semeval_c_*, roft): AUROC and F1 of the word probabilities against the span labels,
and for the "human then AI" sets (SemEval-2024 task 8 subtask C, RoFT) the mean absolute error of the boundary word index
obtained by maximum likelihood on the word probabilities.

  python 04_eval.py --model <run>/model.pt --out <run>/eval.json [--device cuda] [--sets a,b] [--max_docs N]
"""
import argparse, json, os, sys, time
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, f1_score

sys.path.insert(0, os.path.dirname(__file__))
D = os.path.join(os.environ["V45_DATA_ROOT"], "data/aidetect")
DOC_SETS = ["mage_test", "mage_ood_gpt", "mage_ood_gpt_para", "raid_id", "raid_unseen_model", "raid_unseen_domain", "raid_unseen_attack",
            "raid_unseen_all", "raid_attacks", "semeval_a_test", "hc3", "textsight_2026", "spans_test"]
WORD_SETS = ["spans_test", "raid_splice", "semeval_c_dev", "semeval_c_test", "roft"]


def tpr_at_fpr(y, s, fpr):
    h = np.sort(s[y == 0])
    if len(h) == 0 or (y == 1).sum() == 0:
        return float("nan")
    thr = h[min(len(h) - 1, int(np.ceil((1 - fpr) * len(h))))] if fpr > 0 else h[-1]
    return float((s[y == 1] > thr).mean())


def doc_metrics(y, p):
    y = np.asarray(y); p = np.asarray(p)
    r = dict(n=int(len(y)), n_ai=int(y.sum()))
    if 0 < y.sum() < len(y):
        r.update(auroc=100 * roc_auc_score(y, p), tpr1=100 * tpr_at_fpr(y, p, 0.01), tpr5=100 * tpr_at_fpr(y, p, 0.05))
        r["f1"] = 100 * f1_score(y, p > 0.5)
    r["acc"] = 100 * float(((p > 0.5) == (y == 1)).mean())
    if (y == 0).any(): r["fpr"] = 100 * float((p[y == 0] > 0.5).mean())
    if (y == 1).any(): r["recall"] = 100 * float((p[y == 1] > 0.5).mean())
    return r


def word_truth(text, words, spans):
    """1 if more than half of the word's characters lie in an AI span."""
    m = np.zeros(len(text) + 1, dtype=np.int8)
    for s, e in spans:
        m[s:e] = 1
    cs = np.concatenate([[0], np.cumsum(m)])
    return np.array([(cs[e] - cs[s]) / max(e - s, 1) > 0.5 for s, e, _ in words], dtype=np.int8)


def best_boundary(p):
    """k maximizing sum_{i<k} log(1-p_i) + sum_{i>=k} log p_i (words before k human, from k on AI)."""
    p = np.clip(np.asarray(p), 1e-4, 1 - 1e-4)
    a = np.concatenate([[0], np.cumsum(np.log(1 - p))]); b = np.concatenate([np.cumsum(np.log(p)[::-1])[::-1], [0]])
    return int(np.argmax(a + b))


def word_metrics(df, preds):
    yw, pw, maes, ious, mixed_auc = [], [], [], [], []
    for t, sp, r in zip(df.text, df.spans, preds):
        w = r["words"]
        if not w: continue
        y = word_truth(t, w, [tuple(x) for x in sp]); p = np.array([x[2] for x in w])
        yw.append(y); pw.append(p)
        k = best_boundary(p); ktrue = int(np.argmax(y)) if y.any() else len(y)
        maes.append(abs(k - ktrue))
        pred = p > 0.5; inter = (pred & (y == 1)).sum(); uni = (pred | (y == 1)).sum()
        ious.append(inter / uni if uni else 1.0)
        if 0 < y.sum() < len(y): mixed_auc.append(roc_auc_score(y, p))
    y, p = np.concatenate(yw), np.concatenate(pw)
    r = dict(words=int(len(y)), word_auroc=100 * roc_auc_score(y, p) if 0 < y.sum() < len(y) else float("nan"),
             word_f1=100 * f1_score(y, p > 0.5), mean_doc_iou=100 * float(np.mean(ious)),
             mean_doc_word_auroc=100 * float(np.mean(mixed_auc)) if mixed_auc else float("nan"),
             boundary_mae_words=float(np.mean(maes)), boundary_median_ae=float(np.median(maes)),
             boundary_within5=100 * float((np.array(maes) <= 5).mean()))
    return r


def run(model, out, device, sets, max_docs, window, n_layers=12, causal=False, quant=0, dtype=None):
    from infer import Predictor
    P = Predictor(model, device=device, window=window, n_layers=n_layers, causal=causal, quant=quant, dtype=dtype)
    res, t_all = {}, time.time()
    for name in sets:
        df = pd.read_parquet(f"{D}/prep/eval/{name}.parquet")
        if max_docs and len(df) > max_docs: df = df.sample(max_docs, random_state=0)
        need_words = name in WORD_SETS
        t0 = time.time()
        preds = P.predict(df.text.tolist(), with_words=need_words)
        dt = time.time() - t0
        r = {}
        if name in DOC_SETS:
            p = np.array([x["doc_prob"] for x in df.assign(_=0).index.map(lambda i: preds[df.index.get_loc(i)])])
            r["doc"] = doc_metrics(df.label.values, p)
            for col in ["model", "attack", "domain"]:
                if col in df and df[col].nunique() > 1 and name != "spans_test":
                    sub = {}
                    h = df.label.values == 0
                    for g, idx in df.groupby(col).indices.items():
                        mask = np.zeros(len(df), bool); mask[idx] = True
                        if (df.label.values[mask] == 1).any() and col != "model" or col == "model" and g != "human":
                            # machine rows of this group against all human rows of the set (shared threshold)
                            m2 = mask & (df.label.values == 1) | h
                            sub[g] = doc_metrics(df.label.values[m2], p[m2])
                    r[f"by_{col}"] = {k: {kk: vv for kk, vv in v.items() if kk in ("auroc", "tpr1", "recall", "n_ai")} for k, v in sub.items()}
        if need_words:
            r["word"] = word_metrics(df, preds)
            if name == "spans_test":  # fraction-of-AI-words estimate versus truth, for mixed docs
                fr = [np.mean([x[2] for x in pr["words"]]) for pr in preds]
                tr = [sum(e - s for s, e in sp) / max(len(t), 1) for t, sp in zip(df.text, df.spans)]
                r["word"]["ai_fraction_corr"] = float(np.corrcoef(fr, tr)[0, 1])
        r["seconds"] = dt; r["docs_per_s"] = len(df) / dt
        res[name] = r
        print(name, json.dumps({k: (v if k in ("seconds", "docs_per_s") else {kk: round(vv, 2) if isinstance(vv, float) else vv for kk, vv in v.items()} if k in ("doc", "word") else "...") for k, v in r.items()}), flush=True)
    res["_total_seconds"] = time.time() - t_all
    if out: json.dump(res, open(out, "w"), indent=1)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--out"); ap.add_argument("--device", default="cuda")
    ap.add_argument("--sets", default=",".join(dict.fromkeys(DOC_SETS + WORD_SETS)))
    ap.add_argument("--max_docs", type=int, default=0); ap.add_argument("--window", type=int, default=1024)
    ap.add_argument("--n_layers", type=int, default=12); ap.add_argument("--causal", type=int, default=0)
    ap.add_argument("--quant", type=int, default=0)
    a = ap.parse_args()
    run(a.model, a.out, a.device, a.sets.split(","), a.max_docs, a.window, a.n_layers, bool(a.causal), int(a.quant))
