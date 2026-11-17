"""Peek at candidate datasets: files, README start and the first rows of one small data file. python 10_inspect_datasets.py repo [repo ...]"""
import sys, os, json
import pandas as pd
from huggingface_hub import HfApi, hf_hub_download

a = HfApi()
for repo in sys.argv[1:]:
    print("=" * 100); print(repo)
    try:
        i = a.dataset_info(repo, files_metadata=True)
    except Exception as e:
        print("  ERR", str(e)[:100]); continue
    fs = [(s.rfilename, (s.size or 0) / 1e6) for s in i.siblings]
    print("  license:", [t for t in i.tags if t.startswith("license:")], "files:", [(n, round(sz, 1)) for n, sz in fs if not n.startswith((".git", "eda/", "assets/"))][:14])
    try:
        t = open(hf_hub_download(repo, "README.md", repo_type="dataset")).read()
        body = t.split("---", 2)[-1].strip().replace("\n\n", "\n")
        print("  README:", body[:900].replace("\n", " | "))
    except Exception as e:
        print("  no README")
    cand = [n for n, sz in fs if n.endswith((".parquet", ".csv", ".jsonl", ".json")) and sz < 150 and not n.startswith(("eda/", "audits/", "scores/"))]
    for n in cand[:1]:
        try:
            p = hf_hub_download(repo, n, repo_type="dataset")
            d = pd.read_parquet(p) if n.endswith(".parquet") else pd.read_csv(p) if n.endswith(".csv") else pd.read_json(p, lines=n.endswith(".jsonl"))
            print(f"  {n}: {d.shape} cols={list(d.columns)[:12]}")
            for c in d.columns:
                if d[c].dtype == object and d[c].nunique() < 12:
                    print("   ", c, d[c].value_counts().head(8).to_dict())
            tc = max((c for c in d.columns if d[c].dtype == object), key=lambda c: d[c].astype(str).str.len().mean())
            print("   text col", tc, "mean chars", int(d[tc].astype(str).str.len().mean()))
        except Exception as e:
            print("  peek failed:", str(e)[:120])
