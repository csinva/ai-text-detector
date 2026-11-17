#!/usr/bin/env python3
"""Round 3: append informal human prose to licensed/human_cp.parquet (existing rows keep their index):
 - Global Voices and other CC BY news/blog posts of Common Pile news_filtered with a url year <= 2017 (Global Voices) (blogger-style reporting), English only
 - US GPO documents (public domain; government text, no AI exposure)
python 16_human_news.py [news_quota] [gpo_quota] [seed]"""
import json, os, random, re, sys
import pandas as pd
from datasets import load_dataset

D = os.path.join(os.environ["V45_DATA_ROOT"], "data/aidetect"); P = f"{D}/licensed/human_cp.parquet"
nq = int(sys.argv[1]) if len(sys.argv) > 1 else 20000; gq = int(sys.argv[2]) if len(sys.argv) > 2 else 4000
rng = random.Random(int(sys.argv[3]) if len(sys.argv) > 3 else 1)
old = pd.read_parquet(P); seen = set(old.text)
EN = re.compile(r"\b(the|and|of|to|is|that)\b", re.I)


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
        if lo <= len(out) <= hi * 1.4 and sum(c.isalpha() for c in out) / len(out) > 0.7 and len(EN.findall(out)) > len(out) / 120: return out
    return None


rows = []
def run(sub, src, quota, keep_p, cap, yearf):
    n = k = 0
    for r in load_dataset(f"common-pile/{sub}", split="train", streaming=True):
        n += 1
        if n > cap or k >= quota: break
        if rng.random() > keep_p: continue
        m = r.get("metadata"); m = json.loads(m) if isinstance(m, str) else m
        lic = str((m or {}).get("license", ""))
        if re.search(r"non-?commercial|noderiv|unknown", lic, re.I): continue
        y = yearf(r)
        if y is False: continue
        t = chunk(r["text"])
        if t and t not in seen:
            seen.add(t); k += 1; rows.append(dict(text=t, src=src, license=lic[:60], year=y or 0, uid=str(m.get("url", ""))[:200]))
    print(sub, "scanned", n, "kept", k, flush=True)


def news_year(r):
    md = r.get("metadata"); md = json.loads(md) if isinstance(md, str) else md
    m = re.search(r"globalvoices\.org/(\d{4})/", str((md or {}).get("url", "")))  # Global Voices urls carry the year; other hosts are skipped
    return int(m.group(1)) if m and int(m.group(1)) <= 2017 else False


run("news_filtered", "news_blog", nq, 0.6, 1_000_000, news_year)
if gq: run("usgpo_filtered", "usgpo", gq, 0.5, 100_000, lambda r: None)
new = pd.DataFrame(rows); out = pd.concat([old, new], ignore_index=True); out.to_parquet(P)
print("added", len(new), "corpus now", len(out), out.groupby("src").size().to_dict())
