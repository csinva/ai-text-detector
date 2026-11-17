#!/usr/bin/env python3
"""Sample human-written passages (400 to 2,400 characters) from openly licensed Common Pile subsets into
$V45_DATA_ROOT/data/aidetect/licensed/human_cp.parquet.

Only documents whose own metadata licence is public domain, CC0, CC BY, CC BY-SA or the Open Parliament Licence are used (no NonCommercial, no
unknown), and where a date exists it must be 2017 or earlier (before widespread AI text). Gutenberg and pre-1929 books are old by construction.
Columns: text, src (register), license, year, uid.   python 14_human_corpus.py [quota scale]
"""
import itertools, json, os, random, re, sys
import pandas as pd
from datasets import load_dataset

D = os.path.join(os.environ["V45_DATA_ROOT"], "data/aidetect"); OUT = f"{D}/licensed/human_cp.parquet"
SCALE = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
OKLIC = re.compile(r"public domain|creative commons zero|attribution(?! non)|open parliament", re.I)
BADLIC = re.compile(r"non-?commercial|noderiv|no derivatives|unknown|all rights", re.I)
rng = random.Random(0)


def licence_ok(meta):
    lic = str((meta or {}).get("license", ""))
    return bool(OKLIC.search(lic)) and not BADLIC.search(lic), lic


def year_of(s):
    m = re.search(r"(1[89]\d\d|20[0-2]\d)", str(s or ""))
    return int(m.group(1)) if m else None


def clean(t):
    t = re.sub(r"\r", "", t); t = re.sub(r"[ \t]+\n", "\n", t); t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def chunk(text, lo=500, hi=2200):
    """one paragraph-aligned chunk of lo..hi characters from a random position (skipping the first and last 8% of long documents)."""
    paras = [p for p in clean(text).split("\n\n") if p.strip()]
    if not paras: return None
    n = len(paras); a = int(n * 0.08) if n > 30 else 0; b = n - a
    for _ in range(6):
        i = rng.randrange(a, max(a + 1, b)); out = ""
        while i < b and len(out) < hi:
            nxt = (out + "\n\n" + paras[i]).strip()
            if len(nxt) > hi and len(out) >= lo: break
            out = nxt if len(nxt) <= hi * 1.4 else paras[i][:hi]; i += 1
        if lo <= len(out) <= hi * 1.4 and sum(c.isalpha() for c in out) / len(out) > 0.7:
            return out
    return None


def stream(sub):
    return load_dataset(f"common-pile/{sub}", split="train", streaming=True)


def collect(name, sub, quota, per_doc=1, date_of=None, max_year=2017, whole=False, scan_limit=2_500_000, keep_p=1.0):
    rows, seen = [], 0
    for r in stream(sub):
        seen += 1
        if seen > scan_limit or len(rows) >= quota: break
        meta = r.get("metadata"); meta = json.loads(meta) if isinstance(meta, str) else meta
        if keep_p < 1 and rng.random() > keep_p: continue  # random subsample so the passages span the whole date range of a chronologically ordered stream
        ok, lic = licence_ok(meta)
        if not ok: continue
        y = year_of(date_of(r, meta)) if date_of else None
        if date_of and (y is None or y > max_year): continue
        text = r["text"]
        if whole:
            t = clean(text)
            if 500 <= len(t) <= 2400: rows.append(dict(text=t, src=name, license=lic[:60], year=y, uid=r.get("id")))
            continue
        for _ in range(per_doc):
            t = chunk(text)
            if t: rows.append(dict(text=t, src=name, license=lic[:60], year=y, uid=r.get("id")))
        if seen % 20000 == 0: print(f"  {name}: scanned {seen}, kept {len(rows)}", flush=True)
    print(f"{name}: {len(rows)} passages from {seen} documents", flush=True)
    return rows


q = lambda n: int(n * SCALE)
rows = []
rows += collect("stackexchange", "stackexchange_filtered", q(30000), per_doc=1, date_of=lambda r, m: r.get("created"), keep_p=0.01)
rows += collect("arxiv_abstract", "arxiv_abstracts_filtered", q(15000), date_of=lambda r, m: r.get("created"), whole=True, keep_p=0.02)
rows += collect("pubmed", "pubmed_filtered", q(15000), per_doc=1, date_of=lambda r, m: r.get("created"), keep_p=0.03)
rows += collect("gutenberg", "project_gutenberg_filtered", q(20000), per_doc=2, keep_p=0.5)
rows += collect("pre1929_books", "pre_1929_books_filtered", q(8000), per_doc=1, keep_p=0.3)
rows += collect("uk_hansard", "uk_hansard_filtered", q(8000), per_doc=1, date_of=lambda r, m: (m or {}).get("year"), keep_p=0.1)
rows += collect("foodista", "foodista_filtered", q(4000), date_of=lambda r, m: r.get("created"), whole=True)
df = pd.DataFrame(rows).drop_duplicates("text").reset_index(drop=True)
df.to_parquet(OUT)
print(df.groupby("src").agg(n=("text", "size"), chars=("text", lambda s: int(s.str.len().mean())), min_year=("year", "min"), max_year=("year", "max")).to_string())
print(df.groupby("license").size().to_dict())
