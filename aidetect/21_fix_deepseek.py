#!/usr/bin/env python3
"""vLLM returned the DeepSeek-R1-0528-Qwen3-8B generations as byte-level BPE strings (U+0120 for a space, U+010A for a newline, ...). Decode the generated part back to UTF-8 with the inverse of the GPT-2 byte map
(for 'continue' rows only the part after the human prefix) and recompute the AI span. Rows that cannot be decoded cleanly are dropped.  python 21_fix_deepseek.py"""
import os
import pandas as pd

P = os.path.join(os.environ["V45_DATA_ROOT"], "data/aidetect/licensed/gen/r5_deepseek_r1_0528_qwen3_8b.parquet")
bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256)); cs = bs[:]; n = 0
for b in range(256):
    if b not in bs: bs.append(b); cs.append(256 + n); n += 1
u2b = {chr(c): b for b, c in zip(bs, cs)}
g = pd.read_parquet(P); rows = []; dropped = 0
for r in g.to_dict("records"):
    start = r["spans"][0][0] if r["task"] == "continue" else 0; gen = r["text"][start:]
    if not all(c in u2b for c in gen): rows.append(r) if "Ġ" not in gen and "Ċ" not in gen else None; dropped += int("Ġ" in gen or "Ċ" in gen); continue    # already clean text is kept as is
    fixed = bytes(u2b[c] for c in gen).decode("utf-8", errors="replace").strip()
    if "�" in fixed or len(fixed) < 120: dropped += 1; continue
    text = r["text"][:start] + fixed; r["text"] = text; r["spans"] = [[start, len(text)]]; rows.append(r)
out = pd.DataFrame(rows); out.to_parquet(P); print("kept", len(out), "dropped", dropped, "| artefacts left:", int(out.text.str.contains("Ġ|Ċ").sum()), "| mean chars", int(out.text.str.len().mean()))
print(repr(out[out.task == "write"].iloc[0].text[:200])); print(repr(out[out.task == "continue"].iloc[0].text[-160:]))
