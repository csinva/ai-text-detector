#!/usr/bin/env python3
"""Classify text as AI-generated and write a word-level heatmap page.

  python predict.py --model <run>/model.pt --text "..." [--file doc.txt] [--device cpu] [--html out.html] [--json out.json] [--explain]

Prints P(AI-generated) for the document. --html writes a page with the AI-probability view; with --explain it also contains the
feature-importance (integrated gradients of the document logit) and attention-rollout views. --explain needs the fp32 model (no --quant),
costs about `steps` forward+backward passes, and explains the first 1,024 tokens.
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(__file__))
from infer import Predictor
from heatmap import render_page

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True); ap.add_argument("--text"); ap.add_argument("--file")
ap.add_argument("--device", default="cpu"); ap.add_argument("--html"); ap.add_argument("--json")
ap.add_argument("--quant", type=int, default=0, help="CPU int8: 1 all linear, 2 all but MLP down + heads (recommended), 3 all but heads")
ap.add_argument("--explain", action="store_true", help="add feature-importance (integrated gradients) and attention-rollout views")
ap.add_argument("--steps", type=int, default=32, help="integrated gradients steps")
a = ap.parse_args()
if a.explain and a.quant:
    sys.exit("--explain needs the fp32 model: drop --quant")
text = a.text if a.text is not None else open(a.file).read()
P = Predictor(a.model, device=a.device, quant=int(a.quant))
t0 = time.time(); r = P.predict([text])[0]; dt = time.time() - t0
print(f"P(AI-generated) = {r['doc_prob']:.3f}   tokens = {r['n_tokens']}   {dt:.2f}s on {a.device}")
ex = None
if a.explain:
    t0 = time.time(); ex = P.explain(text, steps=a.steps); print(f"explained in {time.time() - t0:.1f}s (IG completeness gap {ex['ig_gap']:+.2f} logits)")
if a.html: open(a.html, "w").write(render_page(text, r, ex))
if a.json: json.dump(dict(prediction=r, explanation=ex), open(a.json, "w"))
