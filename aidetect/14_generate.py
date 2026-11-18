#!/usr/bin/env python3
"""Generate AI text for the human passages with one permissively licensed open model (Apache-2.0 / MIT), using vLLM.

Every human passage in licensed/human_cp.parquet is assigned (by hash) one generator and one task:
  continue  human prefix (35-65% of the sentences) + AI continuation   -> mixed text, exact AI span
  rewrite   AI paraphrase of the whole passage                          -> pure AI
  write     AI text written from the passage's first sentence as the idea -> pure AI
  casual    like write, in informal typing style (humanizer-like)        -> pure AI
This script handles the passages assigned to --model. Output: licensed/gen/<tag>.parquet with text, spans, src, task, gen_model, uid.
  python 14_generate.py --model Qwen/Qwen3-8B --tag qwen3_8b [--limit N]
"""
import argparse, hashlib, os, re, random
import pandas as pd

GENS = {"Qwen/Qwen3-8B": "qwen3_8b", "Qwen/Qwen3-4B-Instruct-2507": "qwen3_4b", "mistralai/Mistral-7B-Instruct-v0.3": "mistral7b", "microsoft/Phi-4-mini-instruct": "phi4mini",
        "allenai/OLMo-2-1124-7B-Instruct": "olmo2_7b", "HuggingFaceTB/SmolLM3-3B": "smollm3"}
TASKS = [("continue", 0.40), ("rewrite", 0.25), ("write", 0.25), ("casual", 0.10)]
REG = {"stackexchange": "a helpful post on a Stack Exchange site", "arxiv_abstract": "a scientific paper abstract", "pubmed": "a paragraph from a biomedical research article",
       "gutenberg": "a passage from a novel or non-fiction book", "pre1929_books": "a passage from an old book", "uk_hansard": "a speech or written answer in the UK Parliament",
       "foodista": "a food blog or recipe article", "news_blog": "a blog post or news report from a citizen-media site", "usgpo": "a US government document", "se2": "a helpful post on a Stack Exchange site", "pes2o": "a paragraph from an open-access research paper", "arxiv_full": "a passage from an arXiv paper",
       "caselaw": "a passage from a US court opinion", "regulations": "a public comment or notice in a US federal regulation docket", "pubmed2": "a paragraph from a biomedical research article", "usgpo2": "a US government document"}
SENT = re.compile(r"(?<=[.!?])\s+|\n+")
PRE = re.compile(r"^(sure|here is|here's|certainly|okay|of course|below is)[^\n]{0,120}\n+", re.I)


def pick(uid, n):
    h = int(hashlib.md5(str(uid).encode()).hexdigest(), 16)
    return h % n, ((h >> 20) % 10000) / 10000


def task_of(u):
    c = 0
    for t, p in TASKS:
        c += p
        if u < c: return t
    return TASKS[-1][0]


def build(row, task, rng):
    text, reg = row.text, REG.get(row.src, "a passage of text")
    sents = [s for s in SENT.split(text) if s.strip()]
    words = len(text.split())
    if task == "continue" and len(sents) >= 4:
        k = max(1, min(len(sents) - 1, int(round(len(sents) * rng.uniform(0.35, 0.65)))))
        # prefix = the text up to the end of sentence k (character-exact slice of the original)
        end = 0
        for s in sents[:k]: end = text.index(s, end) + len(s)
        prefix = text[:end].rstrip(); rest_words = max(40, min(350, len(text[end:].split())))
        return prefix, f"Continue the following text, matching its style, tone and formatting. Write about {rest_words} more words. Output only the continuation, with no introduction or commentary.\n\n{prefix}", rest_words
    first = sents[0].strip()[:300] if sents else text[:200]
    n = max(60, min(350, words))
    if task == "rewrite": return None, f"Rewrite the following text in your own words, keeping the meaning, style and approximate length. Output only the rewritten text.\n\n{text}", n
    if task == "casual": return None, f"Write {reg} (about {n} words) about this: \"{first}\". Write like a real person typing quickly: informal, contractions, a couple of small typos, no lists or headings. Output only the text.", n
    return None, f"Write {reg} (about {n} words) that starts from this idea: \"{first}\". Output only the text itself, with no title or commentary.", n


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--tag", required=True); ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    D = os.path.join(os.environ["V45_DATA_ROOT"], "data/aidetect"); os.makedirs(f"{D}/licensed/gen", exist_ok=True)
    h = pd.read_parquet(f"{D}/licensed/human_cp.parquet")
    gi = list(GENS).index(a.model); ng = len(GENS)
    h["uid2"] = [f"{s}:{i}" for i, s in zip(h.index, h.src)]
    if os.environ.get("GEN_SRCS"): h = h[h.src.isin(os.environ["GEN_SRCS"].split(","))]  # round 3: only the new sources
    if os.environ.get("GEN_AFTER"): h = h[h.index >= int(os.environ["GEN_AFTER"])]  # round 4: only rows added after this index
    mine = h[[pick(u, ng)[0] == gi for u in h.uid2]]
    if a.limit: mine = mine.head(a.limit)
    rng = random.Random(gi); reqs = []
    for row in mine.itertuples():
        task = task_of(pick(row.uid2, 1000)[1])
        prefix, prompt, n = build(row, task, rng)
        if prefix is None and task == "continue": task = "write"
        reqs.append((row, task if prefix is not None or task != "continue" else "write", prefix, prompt, n))
    print(a.model, "requests", len(reqs), pd.Series([r[1] for r in reqs]).value_counts().to_dict(), flush=True)
    from vllm import LLM, SamplingParams
    llm = LLM(model=a.model, dtype="bfloat16", max_model_len=4096, gpu_memory_utilization=0.90, enable_prefix_caching=False, seed=gi)
    msgs = [[{"role": "user", "content": r[3]}] for r in reqs]
    sps = [SamplingParams(temperature=rng.choice([0.6, 0.8, 1.0]), top_p=rng.choice([0.9, 0.95, 1.0]), max_tokens=min(900, int(r[4] * 1.8) + 40)) for r in reqs]
    kw = {"enable_thinking": False} if ("Qwen3-8B" in a.model or "SmolLM3" in a.model) else {}
    outs = llm.chat(msgs, sps, chat_template_kwargs=kw, use_tqdm=False)
    rows = []
    for (row, task, prefix, prompt, n), o in zip(reqs, outs):
        g = o.outputs[0].text.strip()
        g = re.sub(r"<think>.*?</think>", "", g, flags=re.S).strip()
        g = PRE.sub("", g).strip()
        if "<think>" in g or "As an AI" in g or len(g) < 120 or o.outputs[0].finish_reason == "length" and len(g) < 200: continue
        if prefix is not None:
            sep = " " if not prefix.endswith("\n") else ""
            text = prefix + sep + g; spans = [[len(prefix) + len(sep), len(text)]]
        else:
            text, spans = g, [[0, len(g)]]
        rows.append(dict(text=text, spans=spans, src=row.src, task=task, gen_model=GENS[a.model], uid=row.uid2))
    df = pd.DataFrame(rows); df.to_parquet(f"{D}/licensed/gen/{a.tag}.parquet")
    print("saved", len(df), "of", len(reqs), df.groupby("task").size().to_dict(), "mean chars", int(df.text.str.len().mean()), flush=True)


if __name__ == "__main__":
    main()
