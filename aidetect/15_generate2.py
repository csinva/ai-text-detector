#!/usr/bin/env python3
"""Second generation round (vLLM, Apache-2.0 / MIT models only). One model per process; it handles the requests hashed to it:

  std       (new generators only) the round-1 tasks (continue / rewrite / write / casual) on a 40% share of the human passages
  humanize  humanizer-style rewrite of a round-1 AI text (35% of them) by a different model: pure AI, or human prefix + humanized AI continuation
  edit      sentence-level AI edits inside a human passage (40% of passages; about 30% of sentences, 1 to 6): mixed text with exact spans

Output: licensed/gen/r2_<tag>_<mode>.parquet with text, spans, src, task, gen_model, uid (same schema as round 1).
  python 15_generate2.py --model openai/gpt-oss-20b --tag gptoss20b [--modes std,humanize,edit] [--limit N]
"""
import argparse, hashlib, importlib.util, os, random, re
import pandas as pd

spec = importlib.util.spec_from_file_location("g1", os.path.join(os.path.dirname(os.path.abspath(__file__)), "14_generate.py"))
g1 = importlib.util.module_from_spec(spec); spec.loader.exec_module(g1)
GENS = dict(g1.GENS); GENS.update({"openai/gpt-oss-20b": "gptoss20b", "Qwen/Qwen3-30B-A3B-Instruct-2507": "qwen3_30b"})
ORDER = list(GENS); NEW = ["openai/gpt-oss-20b", "Qwen/Qwen3-30B-A3B-Instruct-2507"]
HUM = ["Rewrite the following text so that it reads as if a person wrote it: vary the sentence length, use natural wording and contractions where they fit, drop stock phrases and formulaic structure. Keep the meaning and about the same length. Output only the rewritten text.\n\n{t}",
       "Paraphrase the text below the way a human writer would, with a distinctive voice, uneven rhythm and no generic transitions. Keep every fact. Output only the new text.\n\n{t}",
       "Edit this text to sound less like AI and more like an ordinary person writing quickly: plain words, some short sentences, a few casual phrases. Do not add new facts. Output only the edited text.\n\n{t}"]
PIECE = re.compile(r"[^\n.!?]+[.!?]*\s*|\n+")


def hv(u, salt): return int(hashlib.md5(f"{salt}:{u}".encode()).hexdigest(), 16)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--tag", required=True)
    ap.add_argument("--modes", default="std,humanize,edit"); ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args(); modes = a.modes.split(","); mi = ORDER.index(a.model)
    D = os.path.join(os.environ["V45_DATA_ROOT"], "data/aidetect"); G = f"{D}/licensed/gen"
    h = pd.read_parquet(f"{D}/licensed/human_cp.parquet"); h["uid2"] = [f"{s}:{i}" for i, s in zip(h.index, h.src)]
    if os.environ.get("GEN_SRCS"): h = h[h.src.isin(os.environ["GEN_SRCS"].split(","))]  # round 3: only the new sources
    if os.environ.get("GEN_AFTER"): h = h[h.index >= int(os.environ["GEN_AFTER"])]  # round 4: only rows added after this index
    g = pd.concat([pd.read_parquet(f"{G}/{f}") for f in sorted(os.listdir(G)) if f.endswith(".parquet") and not f.startswith(("smoke", "r2_", "r3_", "r4_"))])
    rng = random.Random(100 + mi); reqs = []  # (mode, task, row dict, prompt, extra, max_new_tokens)

    if "std" in modes and a.model in NEW:
        sub = h[[hv(u, "std") % 100 < 40 and hv(u, "stdgen") % 2 == NEW.index(a.model) for u in h.uid2]]
        for row in sub.head(a.limit or len(sub)).itertuples():
            task = g1.task_of(g1.pick(row.uid2, 1000)[1]); prefix, prompt, n = g1.build(row, task, rng)
            if prefix is None and task == "continue": task = "write"
            reqs.append(("std", task, dict(src=row.src, uid=row.uid2), prompt, prefix, int(n * 1.8) + 40))
    if "humanize" in modes:
        sub = g[[hv(u + t, "hum") % 100 < 35 and (hv(u, "humgen") % len(ORDER) if hv(u, "humgen") % len(ORDER) != ORDER.index({v: k for k, v in GENS.items()}[m]) else (hv(u, "humgen") + 1) % len(ORDER)) == mi
                 for u, t, m in zip(g.uid, g.task, g.gen_model)]]
        for row in sub.head(a.limit or len(sub)).itertuples():
            s = row.spans[0][0] if row.task == "continue" else 0
            prefix = row.text[:s] if row.task == "continue" else None; body = row.text[s:]
            reqs.append(("humanize", "humanize_" + row.task, dict(src=row.src, uid=row.uid), rng.choice(HUM).format(t=body), prefix, min(900, int(len(body.split()) * 1.8) + 60)))
    edit_meta = {}
    if "edit" in modes:
        sub = h[[hv(u, "edit") % 100 < 40 and hv(u, "editgen") % len(ORDER) == mi for u in h.uid2]]
        for row in sub.head(a.limit or len(sub)).itertuples():
            pieces = PIECE.findall(row.text); cand = [i for i, p in enumerate(pieces) if len(p.strip()) > 40]
            if len(cand) < 3: continue
            k = max(1, min(6, round(0.3 * len(pieces)))); pick = sorted(rng.sample(cand, min(k, len(cand))))
            edit_meta[row.uid2] = (row.src, pieces, pick)
            for i in pick:
                ctx = pieces[i - 1].strip() if i > 0 else ""
                reqs.append(("edit", "edit", dict(src=row.src, uid=row.uid2, i=i),
                             "Rewrite the sentence below in different words, keeping its meaning and a similar length. Output only the rewritten sentence, on one line.\n\n" + (f"Previous sentence (context only): {ctx}\n\n" if ctx else "") + f"Sentence: {pieces[i].strip()}",
                             None, 90))
    block_meta = {}
    if "block" in modes:  # round 4: OpAI-style operations on sentence blocks at 10-100% coverage (polish, paraphrase, style, compress, expand)
        OPS = {"polish": "Polish the following passage: fix awkward phrasing and improve flow and clarity while keeping the structure and every fact. Output only the revised passage.\n\n{t}",
               "paraphrase": "Paraphrase the following passage in your own words, keeping all facts and the length about the same. Output only the paraphrase.\n\n{t}",
               "style": "Rewrite the following passage in a {sty} style, keeping all facts. Output only the rewritten passage.\n\n{t}",
               "compress": "Condense the following passage to about two thirds of its length, keeping the key facts. Output only the condensed passage.\n\n{t}",
               "expand": "Expand the following passage with a little more detail and elaboration, up to about one and a half times its length, without changing the facts. Output only the expanded passage.\n\n{t}"}
        STY = ["more formal and academic", "more conversational and fluent", "concise and professional", "polished journalistic"]
        sub = h[[hv(u, "block") % 100 < 50 and hv(u, "blockgen") % len(ORDER) == mi for u in h.uid2]]
        for row in sub.head(a.limit or len(sub)).itertuples():
            pieces = PIECE.findall(row.text)
            if len(pieces) < 4: continue
            blocks, i = [], 0
            while i < len(pieces):
                k = rng.randint(1, 4); blocks.append((i, min(len(pieces), i + k))); i += k
            cov = rng.choice([0.1, 0.2, 0.3, 0.45, 0.6, 0.8, 1.0]); op = rng.choice(list(OPS)); tot = sum(len(x) for x in pieces); pick, acc = [], 0
            for b in rng.sample(range(len(blocks)), len(blocks)):
                if acc >= cov * tot: break
                pick.append(b); acc += sum(len(x) for x in pieces[blocks[b][0]:blocks[b][1]])
            block_meta[row.uid2] = (row.src, pieces, blocks, sorted(pick), op)
            for b in pick:
                seg = "".join(pieces[blocks[b][0]:blocks[b][1]]).strip()
                if len(seg) < 60: continue
                reqs.append(("block", "block_" + op, dict(src=row.src, uid=row.uid2, i=b), OPS[op].format(t=seg, sty=rng.choice(STY)), None, min(900, int(len(seg.split()) * 2.2) + 60)))
    print(a.model, "requests", pd.Series([r[0] for r in reqs]).value_counts().to_dict(), flush=True)
    if not reqs: return
    from vllm import LLM, SamplingParams
    gptoss = "gpt-oss" in a.model
    llm = LLM(model=a.model, dtype="bfloat16", max_model_len=4096, gpu_memory_utilization=0.90, enable_prefix_caching=False, seed=mi)
    pre = [{"role": "system", "content": "Reasoning: low"}] if gptoss else []
    msgs = [pre + [{"role": "user", "content": r[3]}] for r in reqs]
    sps = [SamplingParams(temperature=rng.choice([0.6, 0.8, 1.0]), top_p=rng.choice([0.9, 0.95, 1.0]), max_tokens=r[5] + (600 if gptoss else 0), skip_special_tokens=not gptoss) for r in reqs]
    kw = {"enable_thinking": False} if ("Qwen3-8B" in a.model or "SmolLM3" in a.model) else {}
    outs = llm.chat(msgs, sps, chat_template_kwargs=kw, use_tqdm=False)

    def clean(o):
        t = o.outputs[0].text
        if gptoss:
            if "<|channel|>final<|message|>" not in t: return None
            t = re.sub(r"<\|[^|]+\|>", "", t.split("<|channel|>final<|message|>")[-1])
        t = re.sub(r"<think>.*?</think>", "", t, flags=re.S).strip(); t = g1.PRE.sub("", t).strip()
        return None if ("<think>" in t or "As an AI" in t or not t) else t

    rows, edits, blocks_out = [], {}, {}
    for (mode, task, meta, prompt, prefix, _), o in zip(reqs, outs):
        t = clean(o)
        if t is None: continue
        if mode == "edit":
            t = " ".join(t.split())
            if 15 <= len(t) <= 400: edits.setdefault(meta["uid"], {})[meta["i"]] = t
            continue
        if mode == "block":
            if len(t) >= 30: blocks_out.setdefault(meta["uid"], {})[meta["i"]] = t.strip()
            continue
        if len(t) < 120: continue
        if prefix is not None:
            sep = " " if not prefix.endswith("\n") else ""; text = prefix + sep + t; spans = [[len(prefix) + len(sep), len(text)]]
        else:
            text, spans = t, [[0, len(t)]]
        rows.append(dict(text=text, spans=spans, src=meta["src"], task=task, gen_model=GENS[a.model], uid=meta["uid"], mode=mode))
    for uid, rep in edits.items():
        src, pieces, pick = edit_meta[uid]; out, spans, pos = [], [], 0
        for i, p in enumerate(pieces):
            if i in rep:
                tail = p[len(p.rstrip()):]; new = rep[i] + tail
                spans.append([pos, pos + len(rep[i])]); out.append(new); pos += len(new)
            else:
                out.append(p); pos += len(p)
        if spans: rows.append(dict(text="".join(out), spans=spans, src=src, task="edit", gen_model=GENS[a.model], uid=uid, mode="edit"))
    for uid, rep in blocks_out.items():
        src, pieces, blocks, pick, op = block_meta[uid]; out, spans, pos = [], [], 0
        for b, (x0, x1) in enumerate(blocks):
            seg = "".join(pieces[x0:x1])
            if b in rep:
                tail = seg[len(seg.rstrip()):]; new = rep[b] + tail
                spans.append([pos, pos + len(rep[b])]); out.append(new); pos += len(new)
            else:
                out.append(seg); pos += len(seg)
        if spans: rows.append(dict(text="".join(out), spans=spans, src=src, task="block_" + op, gen_model=GENS[a.model], uid=uid, mode="block"))
    df = pd.DataFrame(rows)
    for mode, x in df.groupby("mode"): x.to_parquet(f"{G}/r2_{a.tag}_{mode}.parquet")
    print("saved", df.groupby("mode").size().to_dict(), flush=True)


if __name__ == "__main__":
    main()
