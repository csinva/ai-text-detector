#!/usr/bin/env python3
"""Classify text as AI-generated and write a word-level heatmap.

  python predict.py --model <run>/model.pt --text "..." [--file doc.txt] [--device cpu] [--html out.html] [--json out.json]

Prints P(AI-generated) for the document and the most suspicious words; --html writes the heatmap (red = AI-like).
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(__file__))
from infer import Predictor, heatmap_html

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True); ap.add_argument("--text"); ap.add_argument("--file")
ap.add_argument("--device", default="cpu"); ap.add_argument("--html"); ap.add_argument("--json"); ap.add_argument("--quant", type=int, default=0, help="CPU int8: 1 all linear, 2 all but MLP down + heads (recommended), 3 all but heads")
a = ap.parse_args()
text = a.text if a.text is not None else open(a.file).read()
P = Predictor(a.model, device=a.device, quant=int(a.quant))
t0 = time.time(); r = P.predict([text])[0]; dt = time.time() - t0
print(f"P(AI-generated) = {r['doc_prob']:.3f}   tokens = {r['n_tokens']}   {dt:.2f}s on {a.device}")
if a.html: open(a.html, "w").write(heatmap_html(text, r["words"], r["doc_prob"]))
if a.json: json.dump(r, open(a.json, "w"))
