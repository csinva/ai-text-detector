#!/usr/bin/env python3
"""Generate AI text for the frontier prompts (21_make_frontier_prompts.py) with open-weight models through vLLM (MIT or Apache-2.0 licences):
  deepseek    deepseek-ai/DeepSeek-R1-0528-Qwen3-8B (MIT; R1-0528 distilled into Qwen3-8B, so its prose is DeepSeek-R1-like; the reasoning before </think> is dropped)
  gptoss120b  openai/gpt-oss-120b (Apache-2.0; reasoning effort low, the final channel is kept)
Output: licensed/gen/r5_<tag>.parquet (text, spans, src, task, gen_model, uid, split, quirks, pid); 'continue' rows are the human prefix plus the AI continuation, with the AI span marked.
  python 21_generate_frontier.py --model deepseek-ai/DeepSeek-R1-0528-Qwen3-8B --tag deepseek_r1_0528_qwen3_8b --group deepseek [--limit N]"""
import argparse, os, re
import pandas as pd

ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--tag", required=True); ap.add_argument("--group", required=True); ap.add_argument("--limit", type=int, default=0); a = ap.parse_args()
D = os.path.join(os.environ["V45_DATA_ROOT"], "data/aidetect"); df = pd.read_parquet(f"{D}/frontier/prompts.parquet"); df = df[df.group == a.group]
if a.limit: df = pd.concat([df[df.split == "eval"].head(a.limit // 2), df[df.split == "train"].head(a.limit - a.limit // 2)])
gptoss = "gpt-oss" in a.model; think = "R1" in a.model
from vllm import LLM, SamplingParams
llm = LLM(model=a.model, dtype="auto" if gptoss else "bfloat16", max_model_len=8192 if think else 4096, gpu_memory_utilization=0.92, enable_prefix_caching=False, seed=5)
pre = [{"role": "system", "content": "Reasoning: low"}] if gptoss else []
msgs = [pre + [{"role": "user", "content": p}] for p in df.prompt]
sps = [SamplingParams(temperature=0.8, top_p=0.95, max_tokens=int(n * 2.2) + (3500 if think else 700 if gptoss else 0), skip_special_tokens=not gptoss) for n in df.n]
outs = llm.chat(msgs, sps, use_tqdm=False)


def final(t):
    if gptoss:
        if "<|channel|>final<|message|>" not in t: return None
        t = re.sub(r"<\|[^|]+\|>", "", t.split("<|channel|>final<|message|>")[-1])
    if think:
        if "</think>" not in t: return None
        t = t.split("</think>")[-1]
    t = re.sub(r"^(sure|here is|here's|certainly|okay|of course|below is)[^\n]{0,120}\n+", "", t.strip(), flags=re.I).strip() if not gptoss else t.strip()
    return t if t and "<think>" not in t and "As an AI" not in t else None


rows = []
for r, o in zip(df.itertuples(), outs):
    t = final(o.outputs[0].text)
    if t is None or len(t) < 120: continue
    if r.task == "continue" and r.prefix:
        sep = " " if not r.prefix.endswith("\n") else ""; text = r.prefix + sep + t; spans = [[len(r.prefix) + len(sep), len(text)]]
    else: text, spans = t, [[0, len(t)]]
    rows.append(dict(text=text, spans=spans, src=r.src, task=r.task, gen_model=a.tag, uid=r.uid, split=r.split, quirks=r.quirks, pid=r.pid))
out = pd.DataFrame(rows); os.makedirs(f"{D}/licensed/gen", exist_ok=True); out.to_parquet(f"{D}/licensed/gen/r5_{a.tag}.parquet")
print("kept", len(out), "of", len(df), out.task.value_counts().to_dict(), "mean chars", int(out.text.str.len().mean()), flush=True)
