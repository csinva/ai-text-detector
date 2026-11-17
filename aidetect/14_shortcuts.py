"""Surface-artefact check: how often do human passages and generated texts show markdown, lists, curly quotes, em dashes, length? python 14_shortcuts.py"""
import os, re
import pandas as pd

D = os.path.join(os.environ["V45_DATA_ROOT"], "data/aidetect/licensed")
h = pd.read_parquet(f"{D}/human_cp.parquet")
g = pd.concat([pd.read_parquet(f"{D}/gen/{f}") for f in sorted(os.listdir(f"{D}/gen")) if f.endswith(".parquet")])
feats = {"markdown **": r"\*\*", "heading #": r"(^|\n)#{1,4} ", "bullet list": r"(^|\n)\s*([-*]|\d+\.) ", "curly quote": "[‘’“”]", "em dash": "—", "starts uppercase": r"^[A-Z]", "ends with period": r"[.!?\"']$"}
rows = []
for name, df in (("human", h), ("AI", g)):
    for src, x in df.groupby("src"):
        r = {"who": name, "src": src, "n": len(x), "mean_chars": int(x.text.str.len().mean())}
        for f, pat in feats.items(): r[f] = round(100 * x.text.str.contains(pat, regex=True).mean())
        rows.append(r)
print(pd.DataFrame(rows).sort_values(["src", "who"]).to_string(index=False))
