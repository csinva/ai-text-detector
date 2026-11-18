#!/usr/bin/env python3
"""CPU latency of the comparison detectors (document score only, one document at a time, 512 tokens), fp32 and int8 (torch dynamic quantization of all Linear layers), same
documents and protocol as 06_cpu_bench.py.   python 17_cpu_bench_baselines.py --out <json> [--threads 1,8] [--reps 6]"""
import argparse, importlib.util, json, os, time
import pandas as pd, torch

HERE = os.path.dirname(os.path.abspath(__file__)); D = os.path.join(os.environ["V45_DATA_ROOT"], "data/aidetect")
spec = importlib.util.spec_from_file_location("bl", f"{HERE}/05_baselines.py")
src = open(f"{HERE}/05_baselines.py").read().replace(".cuda().eval().half()", ".eval()")  # CPU fp32
mod = importlib.util.module_from_spec(spec); exec(compile(src, "bl", "exec"), mod.__dict__)
ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--threads", default="1,8"); ap.add_argument("--reps", type=int, default=6); a = ap.parse_args()
texts = pd.read_parquet(f"{D}/prep_lic3/eval/spans_test.parquet").text.tolist(); res = []
for key in mod.REPOS:
    tok, m, post = mod.load(key)
    for variant in ("fp32", "int8"):
        mm = torch.quantization.quantize_dynamic(m, {torch.nn.Linear}, dtype=torch.qint8) if variant == "int8" else m
        for th in map(int, a.threads.split(",")):
            torch.set_num_threads(th)
            docs = [tok.decode(tok(t, add_special_tokens=False)["input_ids"][:510]) for t in texts[:a.reps + 1]]
            def one(d):
                enc = tok(d, truncation=True, max_length=512, return_tensors="pt")
                with torch.no_grad(): out = mm(enc["input_ids"], enc["attention_mask"]) if key == "desklib" else mm(**enc)
                return post(out)
            one(docs[0]); t0 = time.time()
            for d in docs[1:]: one(d)
            dt = (time.time() - t0) / a.reps; res.append(dict(model=key, variant=variant, threads=th, tokens=512, sec_per_doc=dt)); print(res[-1], flush=True)
json.dump(res, open(a.out, "w"), indent=1)
