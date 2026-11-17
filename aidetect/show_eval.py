"""Print the headline numbers of eval.json files. python show_eval.py run [run ...]"""
import json, os, sys
R = os.path.join(os.environ["V45_DATA_ROOT"], "data/aidetect/runs")
for run in sys.argv[1:]:
    r = json.load(open(f"{R}/{run}/eval.json")); print("==", run)
    for k, v in r.items():
        if k.startswith("_"): continue
        if "doc" in v: print(f"  {k:20s} AUROC {v['doc']['auroc']:.1f}  TPR@1%FPR {v['doc']['tpr1']:.1f}  FPR@0.5 {v['doc'].get('fpr', 0):.1f}  recall@0.5 {v['doc'].get('recall', 0):.1f}  n {v['doc']['n']}")
        if "word" in v: print(f"  {k:20s} word AUROC {v['word']['word_auroc']:.1f}  IoU {v['word']['mean_doc_iou']:.1f}")
    if "wasit_2026" in r: print("  wasit groups:", {g: round(x["auroc"], 1) for g, x in r["wasit_2026"].get("by_model", {}).items()})
