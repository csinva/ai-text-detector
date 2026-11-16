"""Explanations: IG completeness gap and sanity of attention rollout. python test_explain.py <model.pt>"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from infer import Predictor

P = Predictor(sys.argv[1], device="cpu", quant=0)
texts = ["I walked home in the rain yesterday and honestly my shoes are still soaked. Mum says I should buy boots.",
         "Photosynthesis is the biological process by which plants convert light energy into chemical energy. It is important to note that this process plays a vital role in the global carbon cycle."]
for steps in (32, 96):
    for t in texts:
        r = P.explain(t, steps=steps)
        ig = np.array([w[2] for w in r["ig"]]); at = np.array([w[2] for w in r["attn"]])
        print(f"steps {steps}: logit {r['doc_logit']:.2f} baseline {r['baseline_logit']:.2f} sum(IG) {ig.sum():.2f} gap {r['ig_gap']:+.2f} | IG range {ig.min():.2f}..{ig.max():.2f} | attn mean {at.mean():.2f} max {at.max():.2f}")
