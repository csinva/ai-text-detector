#!/usr/bin/env python3
"""Append more Stack Exchange passages (informal Q&A, 2017 or earlier, per-document CC BY-SA) to licensed/human_cp.parquet.
Existing rows keep their index (generation ids are src:index), new rows get new indices; duplicates of existing texts are dropped.
python 15_human_extra.py [quota] [keep_p] [seed]"""
import json, os, random, re, sys
import pandas as pd
from datasets import load_dataset

D = os.path.join(os.environ["V45_DATA_ROOT"], "data/aidetect"); P = f"{D}/licensed/human_cp.parquet"
quota = int(sys.argv[1]) if len(sys.argv) > 1 else 18000
keep_p = float(sys.argv[2]) if len(sys.argv) > 2 else 0.02
rng = random.Random(int(sys.argv[3]) if len(sys.argv) > 3 else 1)
OKLIC = re.compile(r"attribution(?! non)", re.I); BADLIC = re.compile(r"non-?commercial|noderiv|unknown", re.I)
old = pd.read_parquet(P); seen = set(old.text)


def clean(t):
    t = re.sub(r"\r", "", t); t = re.sub(r"[ \t]+\n", "\n", t); return re.sub(r"\n{3,}", "\n\n", t).strip()


def chunk(text, lo=500, hi=2200):
    paras = [p for p in clean(text).split("\n\n") if p.strip()]
    if not paras: return None
    for _ in range(4):
        i = rng.randrange(0, len(paras)); out = ""
        while i < len(paras) and len(out) < hi:
            nxt = (out + "\n\n" + paras[i]).strip()
            if len(nxt) > hi and len(out) >= lo: break
            out = nxt if len(nxt) <= hi * 1.4 else paras[i][:hi]; i += 1
        if lo <= len(out) <= hi * 1.4 and sum(c.isalpha() for c in out) / len(out) > 0.7: return out
    return None


rows, n = [], 0
for r in load_dataset("common-pile/stackexchange_filtered", split="train", streaming=True):
    n += 1
    if n > 6_000_000 or len(rows) >= quota: break
    if rng.random() > keep_p: continue
    m = r.get("metadata"); m = json.loads(m) if isinstance(m, str) else m
    lic = str((m or {}).get("license", ""))
    if not OKLIC.search(lic) or BADLIC.search(lic): continue
    y = re.search(r"(20[01]\d)", str(r.get("created") or ""))
    if not y or int(y.group(1)) > 2017: continue
    t = chunk(r["text"])
    if t and t not in seen:
        seen.add(t); rows.append(dict(text=t, src="stackexchange", license=lic[:60], year=int(y.group(1)), uid=r.get("id")))
    if n % 200000 == 0: print(f"scanned {n}, kept {len(rows)}", flush=True)
new = pd.DataFrame(rows)
out = pd.concat([old, new], ignore_index=True)  # old rows keep their index
out.to_parquet(P)
print("added", len(new), "stackexchange passages; corpus now", len(out), out.groupby("src").size().to_dict())
