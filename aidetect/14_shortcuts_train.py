"""Surface features of the gen_* training texts by label after the prep fixes. python 14_shortcuts_train.py"""
import os, warnings
import pandas as pd
warnings.filterwarnings("ignore")
D = os.path.join(os.environ["V45_DATA_ROOT"], "data/aidetect")
t = pd.read_parquet(f"{D}/prep_lic2_full/train.parquet")
t = t[t.src.str.startswith("gen_")].copy(); t["ai"] = t.spans.map(len) > 0
feats = {"curly quote": "[‘’“”]", "em dash": "—", "ends with . ! ?": r"[.!?][\"'”’)\]]*\s*$", "bullet list": r"(?:^|\n)\s*(?:[-*]|\d+\.) "}
rows = []
for (src, ai), x in t.groupby(["src", "ai"]):
    r = {"src": src, "AI": ai, "n": len(x), "chars": int(x.text.str.len().mean())}
    for f, pat in feats.items(): r[f] = round(100 * x.text.str.contains(pat, regex=True).mean())
    rows.append(r)
print(pd.DataFrame(rows).to_string(index=False))
