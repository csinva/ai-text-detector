"""Round 3: build licensed/humanizer.parquet from HumanizerBench (CC BY 4.0, 6 monthly cycles) and AI Humanizer Benchmark (CC BY 4.0, 3 cycles).
One row per text: bench, kind (input = raw LLM text given to the tools, output = a commercial humanizer's rewrite), cycle, humanizer, category,
source_model, text, sample_id, test_id and the vendor detectors' human-ness scores for outputs (v_<detector>, 1 = judged human).
python 16_build_humanizer_sets.py"""
import os
import pandas as pd
from huggingface_hub import hf_hub_download as dl

D = os.path.join(os.environ["V45_DATA_ROOT"], "data/aidetect"); OUT = f"{D}/licensed/humanizer.parquet"
f = lambda r, n: pd.read_json(dl(r, f"flat/{n}.jsonl", repo_type="dataset"), lines=True)
rows = []
# ---- HumanizerBench
r = "HumanizerBench/humanizerbench"
t, s, d = f(r, "tests"), f(r, "samples"), f(r, "detector_scores")
t = t[t.status == "complete"].copy(); s = s.set_index(["cycle", "id"])
v = d.pivot_table(index="test_id", columns="detector_slug", values="raw_score", aggfunc="mean").add_prefix("v_")
t = t.join(v, on="id")
for x in t.itertuples():
    sm = s.loc[(x.cycle, x.sample_id)] if (x.cycle, x.sample_id) in s.index else None
    if sm is None or not isinstance(x.output_text, str) or len(x.output_text) < 200 or x.output_text.strip() == str(x.input_text).strip(): continue
    rows.append(dict(bench="hb", kind="output", cycle=x.cycle, humanizer=x.humanizer_slug, category=sm.category, source_model=sm.source_model_slug, text=x.output_text, sample_id=x.sample_id, test_id=x.id,
                     **{c: getattr(x, c) for c in v.columns}))
for (cy, sid), sm in s.iterrows():
    rows.append(dict(bench="hb", kind="input", cycle=cy, humanizer="none", category=sm.category, source_model=sm.source_model_slug, text=sm.input_text, sample_id=sid, test_id=f"in-{cy}-{sid}"))
# ---- AI Humanizer Benchmark
r = "ai-humanizer-benchmark/ai-humanizer-benchmark"
t2 = f(r, "tests"); t2 = t2[t2.status == "ok"]
dcols = [c for c in t2.columns if c.startswith("detector_")]
d2 = f(r, "detector_scores").pivot_table(index="test_id", columns="detector", values="human_prob", aggfunc="mean").add_prefix("v_")
t2 = t2.join(d2, on="test_id")
vcols = [c for c in t2.columns if c.startswith("v_")]
seen_in = set()
for x in t2.itertuples():
    if not isinstance(x.output, str) or len(x.output) < 200 or x.output.strip() == str(x.input).strip(): continue
    rows.append(dict(bench="ahb", kind="output", cycle=x.cycle, humanizer=x.humanizer, category=x.category, source_model=x.source_model, text=x.output, sample_id=x.sample_id, test_id=x.test_id, **{c: getattr(x, c) for c in vcols}))
    if (x.cycle, x.sample_id) not in seen_in:
        seen_in.add((x.cycle, x.sample_id))
        rows.append(dict(bench="ahb", kind="input", cycle=x.cycle, humanizer="none", category=x.category, source_model=x.source_model, text=x.input, sample_id=x.sample_id, test_id=f"in-{x.cycle}-{x.sample_id}"))
df = pd.DataFrame(rows); df["text"] = df.text.astype(str)
df = df.drop_duplicates(["bench", "kind", "text"]).reset_index(drop=True)
df.to_parquet(OUT)
print(df.groupby(["bench", "kind"]).size().to_dict())
print(df[df.kind == "output"].groupby(["bench", "humanizer"]).size().to_string())
print("cycles", df.groupby(["bench", "cycle"]).size().to_dict())
