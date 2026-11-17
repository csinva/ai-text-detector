#!/usr/bin/env python3
"""Tokenize prep/{train,dev}.parquet with the voyage-4-nano tokenizer into flat memory-mappable arrays.

ids uint32 (first MAXTOK tokens of each doc), lab uint8 = 255 * fraction of the token's characters inside an AI span,
off int64 doc starts (len n+1), src uint8 codes (srcs.json).
"""
import json, os, sys
import numpy as np, pandas as pd
from multiprocessing import Pool
from tokenizers import Tokenizer

D = os.path.join(os.environ["V45_DATA_ROOT"], "data/aidetect")
MAXTOK, MAXCHAR = 3072, 20000
TOK = "/data/new/prod_models/voyage-4-nano/tokenizer.json"


def work(args):
    texts, spans = args
    tk = Tokenizer.from_file(TOK)
    encs = tk.encode_batch([t[:MAXCHAR] for t in texts])
    ids, labs, lens = [], [], []
    for t, sp, e in zip(texts, spans, encs):
        n = min(len(e.ids), MAXTOK)
        m = np.zeros(min(len(t), MAXCHAR) + 1, dtype=np.int32)
        for s, en in sp:
            m[s:en] = 1
        cs = np.concatenate([[0], np.cumsum(m)])
        off = np.asarray(e.offsets[:n], dtype=np.int64).reshape(-1, 2)
        L = len(cs) - 1
        a, b = np.minimum(off[:, 0], L), np.minimum(off[:, 1], L)
        frac = (cs[b] - cs[a]) / np.maximum(b - a, 1)
        ids.append(np.asarray(e.ids[:n], dtype=np.uint32)); labs.append(np.round(frac * 255).astype(np.uint8)); lens.append(n)
    return np.concatenate(ids), np.concatenate(labs), np.asarray(lens)


def run(name, prep="prep", tok="tok"):
    df = pd.read_parquet(f"{D}/{prep}/{name}.parquet")
    srcs = sorted(df.src.unique()); code = {s: i for i, s in enumerate(srcs)}
    texts, spans = df.text.tolist(), [[tuple(x) for x in s] for s in df.spans]
    ch = 2000
    jobs = [(texts[i:i + ch], spans[i:i + ch]) for i in range(0, len(texts), ch)]
    with Pool(48) as p:
        res = p.map(work, jobs, chunksize=1)
    ids = np.concatenate([r[0] for r in res]); lab = np.concatenate([r[1] for r in res]); lens = np.concatenate([r[2] for r in res])
    off = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)
    os.makedirs(f"{D}/{tok}", exist_ok=True)
    np.save(f"{D}/{tok}/{name}_ids.npy", ids); np.save(f"{D}/{tok}/{name}_lab.npy", lab); np.save(f"{D}/{tok}/{name}_off.npy", off)
    np.save(f"{D}/{tok}/{name}_src.npy", df.src.map(code).to_numpy(np.uint8))
    json.dump(srcs, open(f"{D}/{tok}/srcs.json", "w"))
    print(name, "docs", len(df), "tokens", len(ids), "mean len", lens.mean(), "frac>1024", (lens > 1024).mean(), flush=True)


if __name__ == "__main__":
    rt = next((a.split("=")[1] for a in sys.argv if a.startswith("--ratio=")), "1.3")
    suf = "_lic" + os.environ.get("VER", "3") + ("" if rt == "1.3" else f"_r{int(round(float(rt) * 10))}") + ("_full" if "--full" in sys.argv else "")
    for n in [a for a in sys.argv[1:] if not a.startswith("--")] or ["dev", "train"]:
        run(n, "prep" + suf, "tok" + suf)
