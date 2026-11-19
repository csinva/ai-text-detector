#!/usr/bin/env python3
"""Collect the Claude outputs written by the subagents (frontier/claude_<model>_out_*.jsonl) into licensed/gen/r5_claude_<model>.parquet, same schema as 21_generate_frontier.py.
Invalid lines, empty or very short texts, refusals and meta commentary are dropped and counted.  python 21_ingest_claude.py"""
import glob, json, os, re
import pandas as pd

D = os.path.join(os.environ["V45_DATA_ROOT"], "data/aidetect"); P = pd.read_parquet(f"{D}/frontier/prompts.parquet").set_index("pid")
for grp in ("claude_opus", "claude_sonnet", "claude_haiku", "claude_fable"):
    rows, bad = [], 0
    for f in sorted(glob.glob(f"{D}/frontier/{grp}_out_*.jsonl")):
        for ln in open(f):
            ln = ln.strip()
            if not ln: continue
            try: o = json.loads(ln); pid, t = o["id"], o["text"].strip()
            except Exception: bad += 1; continue
            if pid not in P.index or len(t) < 100 or re.match(r"^(i can't|i cannot|i'm sorry|sorry,)", t, re.I): bad += 1; continue
            r = P.loc[pid]
            if r.task == "continue" and r.prefix:
                sep = " " if not r.prefix.endswith("\n") else ""; text = r.prefix + sep + t; spans = [[len(r.prefix) + len(sep), len(text)]]
            else: text, spans = t, [[0, len(t)]]
            rows.append(dict(text=text, spans=spans, src=r.src, task=r.task, gen_model=grp, uid=r.uid, split=r.split, quirks=r.quirks, pid=pid))
    if rows: pd.DataFrame(rows).to_parquet(f"{D}/licensed/gen/r5_{grp}.parquet")
    print(grp, "kept", len(rows), "dropped", bad, "| mean chars", int(pd.DataFrame(rows).text.str.len().mean()) if rows else 0, flush=True)
