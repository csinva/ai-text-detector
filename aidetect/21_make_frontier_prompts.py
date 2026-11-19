#!/usr/bin/env python3
"""Prompts for the 'frontier' generators (Claude models through Claude Code subagents, DeepSeek-R1-0528-Qwen3-8B and gpt-oss-120b through vLLM).
Each prompt is built from a human passage of the pool (same task wording as 14_generate.py: continue, rewrite, write, casual), so the AI text is topic-matched to a human passage; about half of the prompts add
1 to 3 'quirk instructions' from 21_frontier_quirks.json (tells of recent models, brainstormed with Fable) that suit the task. Passages come from all registers (equal weight per source).
Train prompts use passages outside the 8% evaluation hash split, eval prompts use passages inside it.
  python 21_make_frontier_prompts.py   ->  data/aidetect/frontier/prompts.parquet and claude_<model>_prompts.jsonl"""
import hashlib, importlib.util, json, os, random, sys
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); D = os.path.join(os.environ["V45_DATA_ROOT"], "data/aidetect"); OUT = f"{D}/frontier"; os.makedirs(OUT, exist_ok=True)
spec = importlib.util.spec_from_file_location("g1", f"{HERE}/14_generate.py"); g1 = importlib.util.module_from_spec(spec); spec.loader.exec_module(g1)
argv, sys.argv = sys.argv, ["01_prep.py"]; sp2 = importlib.util.spec_from_file_location("prep", f"{HERE}/01_prep.py"); prep = importlib.util.module_from_spec(sp2); sp2.loader.exec_module(prep); sys.argv = argv
Q = json.load(open(f"{HERE}/21_frontier_quirks.json"))["quirk_instructions"]
PROSE, STRUCT = [0, 1, 2, 3, 4, 10, 11, 13], [5, 6, 7, 8, 9, 12]               # indices of quirks that fit running prose / that need headings, bullets or an assistant reply
ev_uid = lambda u: int(hashlib.md5(u.encode()).hexdigest(), 16) % 100 < 8
GROUPS = {"claude_opus": 48, "claude_sonnet": 48, "claude_haiku": 48, "claude_fable": 48, "deepseek": 5000, "gptoss120b": 5000}
rng = random.Random(5)
h = pd.read_parquet(f"{D}/licensed/human_cp.parquet"); h["uid"] = [f"{s}:{i}" for i, s in zip(h.index, h.src)]; h["text"] = h.text.map(prep.trim_end); h = h[h.text.str.len() >= 300]
def clean_start(t):                                                              # start at the first real line, so the topic seed is a sentence and not a heading or a number
    ls = t.split("\n")
    for i, l in enumerate(ls):
        if len(l.split()) >= 10: return "\n".join(ls[i:]).strip()
    return ""
h["text"] = h.text.map(clean_start); h = h[(h.text.str.len() >= 300) & (h.text.str.count(r"\$") < 4)]
h["ev"] = h.uid.map(ev_uid); bysrc = {s: g for s, g in h.groupby("src")}; srcs = sorted(bysrc); used = set(); rows = []
for grp, n in GROUPS.items():
    n_eval = max(1, round(0.12 * n)); plan = [("eval", n_eval)] + [("train", n - n_eval)]
    for split, k in plan:
        for j in range(k):
            for _ in range(50):                                                   # equal weight per source, no passage reused across groups
                s = srcs[rng.randrange(len(srcs))]; g = bysrc[s]; g = g[g.ev == (split == "eval")]; row = g.iloc[rng.randrange(len(g))]
                if row.uid not in used: used.add(row.uid); break
            task = g1.task_of(rng.random()); prefix, prompt, nw = g1.build(row, task, rng)
            if prefix is None and task == "continue": task = "write"; prefix, prompt, nw = g1.build(row, task, rng)
            ids = []
            if rng.random() < 0.5 and task != "casual":
                pool = PROSE if task in ("continue", "rewrite") else PROSE + STRUCT; ids = rng.sample(pool, rng.choice([1, 2, 3]))
                prompt += "\n\nStyle requirements (follow all of them): " + " ".join(Q[i] for i in ids)
            rows.append(dict(pid=f"{grp}-{split}-{j}", group=grp, uid=row.uid, src=row.src, split=split, task=task, quirks=json.dumps(ids), prompt=prompt, prefix=prefix, n=nw))
df = pd.DataFrame(rows); df.to_parquet(f"{OUT}/prompts.parquet")
for grp in GROUPS:
    if grp.startswith("claude"): open(f"{OUT}/{grp}_prompts.jsonl", "w").write("\n".join(json.dumps(dict(id=r.pid, prompt=r.prompt), ensure_ascii=False) for r in df[df.group == grp].itertuples()) + "\n")
print(df.groupby(["group", "split"]).size().unstack().to_dict("index")); print(df.task.value_counts().to_dict(), "with quirks:", round((df.quirks != "[]").mean(), 2), "| prompt words (claude):", int(df[df.group.str.startswith("claude")].prompt.str.split().str.len().mean()))
