"""Round 4: OpAI-Bench test split (Apache-2.0 declared; human source provenance not stated, so EVALUATION ONLY, never trained on) -> eval sets in prep_lic3/eval and prep_lic4/eval.
 opai_doc:   v0 (human seed, label 0) vs v1..v8 (progressively AI-edited, label 1); model = generator, attack = version, domain = domain.
 opai_spans: mixed versions v2..v6 with character spans of the AI-edited text (word-level eval).
python 17_build_opai_eval.py"""
import ast, glob, os
import pandas as pd

D = os.path.join(os.environ["V45_DATA_ROOT"], "data/aidetect")
d = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f"{D}/licensed/opai/default/test/*.csv"))], ignore_index=True)
d = d.drop_duplicates(["text"]); d["spans"] = d.ai_spans_char.map(lambda s: [list(x) for x in ast.literal_eval(s)] if isinstance(s, str) else [])
keep = lambda x: pd.DataFrame(dict(text=x.text, label=(x.version != "v0").astype(int), grp="opai_doc", model=x.generator, attack=x.version, domain=x.domain, spans=x.spans))
h = d[d.version == "v0"].drop_duplicates(["id", "domain"]).pipe(lambda x: x.sample(min(len(x), 2500), random_state=0))
a = d[d.version != "v0"]; a = pd.concat([g.sample(min(len(g), 600), random_state=0) for _, g in a.groupby("version")])
doc = pd.concat([keep(h), keep(a)], ignore_index=True)
sp = d[d.version.isin(["v2", "v3", "v4", "v5", "v6"]) & (d.spans.map(len) > 0)].pipe(lambda x: x.sample(min(len(x), 2500), random_state=1))
sp = pd.DataFrame(dict(text=sp.text, label=1, grp="opai_spans", model=sp.generator, attack=sp.version, domain=sp.domain, spans=sp.spans))
for v in ("prep_lic3", "prep_lic4_r10_full", "prep_lic5_r10_full", "prep_lic5_r15_full", "prep_lic5_r20_full"):
    o = f"{D}/{v}/eval"
    if os.path.isdir(o): doc.to_parquet(f"{o}/opai_doc.parquet"); sp.to_parquet(f"{o}/opai_spans.parquet")
print(len(doc), doc.label.mean(), len(sp)); print(doc.groupby("attack").size().to_dict())
