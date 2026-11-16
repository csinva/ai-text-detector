#!/usr/bin/env python3
"""Compact comparison table of eval.json files:  python 08_compare.py run1 run2 ...  (run dirs or names under $V45_DATA_ROOT/data/aidetect/runs)"""
import json, os, sys
import numpy as np

D = os.path.join(os.environ["V45_DATA_ROOT"], "data/aidetect/runs")
DOC = [("mage_test", "auroc"), ("mage_ood_gpt", "auroc"), ("mage_ood_gpt_para", "auroc"), ("raid_id", "auroc"), ("raid_unseen_model", "auroc"),
       ("raid_unseen_domain", "auroc"), ("raid_unseen_attack", "auroc"), ("raid_unseen_all", "auroc"), ("semeval_a_test", "auroc"), ("hc3", "auroc"),
       ("textsight_2026", "auroc"), ("spans_test", "auroc")]
WORD = [("spans_test", "word_auroc"), ("spans_test", "mean_doc_iou"), ("raid_splice", "word_auroc"), ("semeval_c_test", "boundary_mae_words"), ("roft", "word_auroc")]
cols = [f"{s}:{m}" for s, m in DOC]
names = sys.argv[1:]
rows = []
for n in names:
    r = json.load(open(n if n.endswith(".json") else f"{D}/{n}/eval.json"))
    v = [r[s]["doc"][m] if s in r and "doc" in r[s] else float("nan") for s, m in DOC] + [r[s]["word"][m] if s in r and "word" in r[s] else float("nan") for s, m in WORD]
    rows.append(v)
hdr = [f"{s}" for s, _ in DOC] + [f"w:{s}:{m}" for s, m in WORD]
print("metric".ljust(34) + "".join(n[:14].rjust(15) for n in names))
for j, h in enumerate(hdr):
    print(h.ljust(34) + "".join(f"{rows[i][j]:15.2f}" for i in range(len(names))))
dm = [np.mean([rows[i][j] for j in range(len(DOC))]) for i in range(len(names))]
print("mean doc AUROC".ljust(34) + "".join(f"{x:15.2f}" for x in dm))
