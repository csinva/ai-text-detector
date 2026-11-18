#!/usr/bin/env python3
"""Round 4: much more licensed human text. One source per process, written to licensed/human_r4_<src>.parquet; `--merge` appends all of them to human_cp.parquet
(existing rows keep their index, new rows get src names news_blog / se2 / pes2o / arxiv_full / caselaw / regulations / usgpo2 / pubmed2).
Filters: per-document licence must not be NC/ND/unknown; dated sources are kept only for 2017 or earlier (peS2o year, arXiv id < 1801, Global Voices url year, Stack Exchange creation date);
caselaw and government text are public domain. Passages of 500-2200 characters, English only, deduplicated against the pool; Global Voices skips urls already used.
python 17_human_more.py <src> [seed]      python 17_human_more.py --merge"""
import json, os, random, re, sys
import pandas as pd
from datasets import load_dataset

D = os.path.join(os.environ["V45_DATA_ROOT"], "data/aidetect"); P = f"{D}/licensed/human_cp.parquet"
# src: (common-pile subset, quota, keep probability, scan cap)
SRC = {"se2": ("stackexchange_filtered", 60000, 0.06, 8_000_000), "pes2o": ("peS2o_filtered", 40000, 0.004, 40_000_000), "arxiv_full": ("arxiv_papers_filtered", 15000, 0.5, 400_000),
       "caselaw": ("caselaw_access_project_filtered", 25000, 0.02, 3_000_000), "gv2": ("news_filtered", 20000, 0.9, 200_000), "regulations": ("regulations_filtered", 8000, 0.3, 200_000),
       "usgpo2": ("usgpo_filtered", 10000, 0.1, 500_000), "pubmed2": ("pubmed_filtered", 20000, 0.01, 4_000_000)}
EN = re.compile(r"\b(the|and|of|to|is|that)\b", re.I); BAD = re.compile(r"non-?commercial|noderiv|unknown|\bnc\b", re.I)


def clean(t):
    t = re.sub(r"\r", "", t); t = re.sub(r"[ \t]+\n", "\n", t); t = re.sub(r"[ \t]{2,}", " ", t); return re.sub(r"\n{3,}", "\n\n", t).strip()


def chunk(rng, text, lo=500, hi=2200):
    paras = [p.strip() for p in clean(text).split("\n\n") if p.strip()]
    if not paras: return None
    for _ in range(4):
        i = rng.randrange(0, len(paras)); out = ""
        while i < len(paras) and len(out) < hi:
            nxt = (out + "\n\n" + paras[i]).strip()
            if len(nxt) > hi and len(out) >= lo: break
            out = nxt if len(nxt) <= hi * 1.4 else paras[i][:hi]; i += 1
        if lo <= len(out) <= hi * 1.4 and sum(c.isalpha() for c in out) / len(out) > 0.7 and len(EN.findall(out)) > len(out) / 120: return out
    return None


def year_ok(src, r, m):
    """returns the year (or 0 when the source is undated public-domain text) or None to skip the document"""
    if src == "se2":
        y = re.search(r"(20[01]\d)", str(r.get("created") or "")); return int(y.group(1)) if y and int(y.group(1)) <= 2017 else None
    if src == "pes2o":
        y = str(m.get("year", "")); return int(y) if y.isdigit() and 1990 <= int(y) <= 2017 and str(m.get("oa_license", "")).upper().startswith(("CCBY", "CC0", "CC-BY", "PUBLICDOMAIN")) and "NC" not in str(m.get("oa_license", "")).upper() else None
    if src == "arxiv_full":
        a = re.search(r"abs/(\d{2})(\d{2})\.\d+$|abs/[a-z\-]+/(\d{2})\d+", str(m.get("url", "")))
        return 2000 + int(a.group(1)) if a and a.group(1) and int(a.group(1)) < 18 and int(a.group(1)) >= 7 else None
    if src == "gv2":
        a = re.search(r"globalvoices\.org/(\d{4})/", str(m.get("url", ""))); return int(a.group(1)) if a and int(a.group(1)) <= 2017 else None
    return 0


def run(src, seed):
    sub, quota, keep_p, cap = SRC[src]; rng = random.Random(seed)
    old = pd.read_parquet(P); seen = set(old.text); used = set(old.uid.astype(str)) if src == "gv2" else set()
    rows, n = [], 0
    for r in load_dataset(f"common-pile/{sub}", split="train", streaming=True):
        n += 1
        if n > cap or len(rows) >= quota: break
        if rng.random() > keep_p: continue
        m = r.get("metadata"); m = json.loads(m) if isinstance(m, str) else (m or {})
        lic = str(m.get("license", m.get("oa_license", "")))
        if BAD.search(lic): continue
        if src == "se2" and not re.search(r"attribution(?! non)", lic, re.I): continue
        y = year_ok(src, r, m)
        if y is None: continue
        if src == "gv2" and str(m.get("url", ""))[:200] in used: continue
        t = chunk(rng, r["text"])
        if t and t not in seen:
            seen.add(t); used.add(str(m.get("url", ""))[:200]); rows.append(dict(text=t, src=src, license=lic[:60], year=y, uid=str(m.get("url", r.get("id", "")))[:200]))
        if n % 200000 == 0: print(src, f"scanned {n} kept {len(rows)}", flush=True)
    pd.DataFrame(rows).to_parquet(f"{D}/licensed/human_r4_{src}.parquet"); print(src, "scanned", n, "kept", len(rows), flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "--merge":
        old = pd.read_parquet(P); seen = set(old.text); new = []
        for s in SRC:
            f = f"{D}/licensed/human_r4_{s}.parquet"
            if os.path.exists(f):
                x = pd.read_parquet(f); x = x[~x.text.isin(seen)]; seen |= set(x.text); new.append(x.assign(src="news_blog" if s == "gv2" else s))
        out = pd.concat([old] + new, ignore_index=True); out.to_parquet(P); print("corpus", len(out), out.groupby("src").size().to_dict())
    else: run(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 1)
