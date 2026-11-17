#!/usr/bin/env python3
"""Build the commercially usable data tree $V45_DATA_ROOT/data/aidetect/licensed/ and the licence audit (license_audit.json).

Rule (applied to every dataset, model and sub-source used in this project):
  KEEP  text whose rights are traceable to an explicit open licence that allows commercial use (public domain, CC0, CC BY, CC BY-SA,
        MIT, Apache-2.0, OGL, ODC-By), or a per-row licence of that kind.
  DROP  non-commercial / research-only / "other" terms, no declared licence, and scraped third-party text (news, reviews, forum posts,
        poems, recipes, web pages) even when the aggregate dataset card says MIT or Apache, because a card cannot grant rights to text
        its author does not hold.
  Machine-written text is dropped when its generator's weights are released under non-commercial or research-only terms (LLaMA-1, OPT,
  GLM-130B, MPT-30B-Chat). Text from commercial APIs and Apache/MIT/Llama-2/RAIL models is kept.
This is an engineering audit from the published terms, not legal advice.
"""
import hashlib, json, os, re, sys
import pandas as pd
from huggingface_hub import hf_hub_download

D = os.path.join(os.environ["V45_DATA_ROOT"], "data/aidetect"); RAW, OUT = f"{D}/raw", f"{D}/licensed"
os.makedirs(OUT, exist_ok=True)
nh = lambda t: hashlib.md5(re.sub(r"[^a-z0-9]", "", str(t).lower())[:400].encode()).hexdigest()
audit = []


def add(name, kind, declared, decision, why, url="", kept=None, total=None):
    audit.append(dict(name=name, kind=kind, declared=declared, decision=decision, why=why, url=url, kept=kept, total=total))


# ---------------------------------------------------------------- RAID
RAID_KEEP = {"books": "CMU Book Summary Corpus (plot summaries from Wikipedia), CC BY-SA", "wiki": "Wikipedia introductions (GPT-wiki-intro), CC BY-SA",
             "abstracts": "arXiv paper abstracts (arXiv metadata dedicated CC0)"}
RAID_DROP = {"news": "BBC News dataset (Greene and Cunningham 2006), BBC copyright, non-commercial research terms", "poetry": "poems scraped from poemhunter.com, copyrighted",
             "recipes": "RecipeNLG, non-commercial research licence", "reddit": "Reddit TL;DR posts, user-owned third-party text", "reviews": "IMDb movie reviews (Maas et al. 2011), research use"}
if os.path.exists(f"{D}/raid_train.parquet"):
    r = pd.read_parquet(f"{D}/raid_train.parquet")
else:  # raw RAID csv (multi-line cells)
    import pyarrow.csv as pc
    cols = ["id", "source_id", "model", "decoding", "attack", "domain", "generation"]
    r = pc.read_csv(f"{RAW}/raid/train.csv", convert_options=pc.ConvertOptions(include_columns=cols, column_types={c: "string" for c in cols}),
                    read_options=pc.ReadOptions(block_size=1 << 28), parse_options=pc.ParseOptions(newlines_in_values=True)).to_pandas()
n0 = len(r)
raid_domain_counts = r.domain.value_counts().to_dict()
r = r[r.domain.isin(RAID_KEEP) & (r.model != "mpt-chat")]
r.to_parquet(f"{OUT}/raid.parquet", compression="zstd")
for dom, why in RAID_KEEP.items():
    add(f"RAID / {dom}", "dataset domain", "MIT (card) over upstream " + why, "keep", why, "https://huggingface.co/datasets/liamdugan/raid", int((r.domain == dom).sum()), int(int(raid_domain_counts.get(dom, 0))))
for dom, why in RAID_DROP.items():
    add(f"RAID / {dom}", "dataset domain", "MIT (card), upstream differs", "drop", why, "https://arxiv.org/abs/2405.07940", 0)
add("RAID / generator mpt-chat", "machine text", "MPT-30B-Chat weights: CC BY-NC-SA 4.0", "drop", "non-commercial generator; its texts are removed", "https://huggingface.co/mosaicml/mpt-30b-chat", 0)
add("RAID / other generators (gpt2, mpt, mistral, mistral-chat, llama-chat, cohere, cohere-chat, gpt3, chatgpt, gpt4)", "machine text", "MIT, Apache-2.0, Llama 2 Community, vendor API terms", "keep",
    "permissive weights or API output terms that allow commercial use of outputs (OpenAI and Llama 2 terms forbid using outputs to train competing generative LLMs, not classifiers)", "", int(len(r)), n0)
print("raid kept", len(r), "of", n0, flush=True)

# ---------------------------------------------------------------- MAGE (SQuAD domain only, permissive generators only)
BAD_GEN = re.compile(r"machine_continuation_(?:opt|7B|13B|30B|65B|GLM130B)", re.I)  # OPT / OPT-IML (research), LLaMA-1 sizes (research), GLM-130B (research)
parts = {}
for sp in ("train", "valid", "test"):
    m = pd.read_csv(f"{RAW}/mage/{sp}.csv")
    m["dom"] = m.src.str.replace("_human", "").str.replace(r"_machine.*", "", regex=True)
    keep = (m.dom == "squad") & ~m.src.str.contains(BAD_GEN)
    parts[sp] = (int(keep.sum()), len(m)); m[keep].drop(columns=["dom"]).to_parquet(f"{OUT}/mage_squad_{sp}.parquet")
add("MAGE / squad domain", "dataset domain", "Apache-2.0 (card) over SQuAD (CC BY-SA 4.0, Wikipedia paragraphs)", "keep", "SQuAD contexts are CC BY-SA 4.0 Wikipedia text", "https://huggingface.co/datasets/yaful/MAGE", sum(v[0] for v in parts.values()), sum(v[1] for v in parts.values()))
for dom, why in {"cmv": "Reddit ChangeMyView posts, third-party user text", "eli5": "Reddit ELI5 answers, third-party user text", "hswag": "HellaSwag contexts include WikiHow (CC BY-NC-SA) and ActivityNet captions",
                 "roct": "ROCStories, research-only licence", "sci_gen": "SciGen table descriptions from papers, no clear licence", "tldr": "Reddit TL;DR posts, third-party user text", "wp": "Reddit WritingPrompts stories, third-party user text",
                 "xsum": "BBC articles (XSum), BBC copyright", "yelp": "Yelp reviews, academic-use-only dataset terms"}.items():
    add(f"MAGE / {dom}", "dataset domain", "Apache-2.0 (card), upstream differs", "drop", why, "https://arxiv.org/abs/2305.13242", 0)
add("MAGE / generators OPT, OPT-IML, LLaMA-1 (7B to 65B), GLM-130B", "machine text", "research-only or non-commercial model terms", "drop", "texts from non-commercial generators removed", "", 0)
add("MAGE OOD test sets (GPT, GPT paraphrased)", "dataset", "Apache-2.0 (card)", "drop", "no domain column, so their human text cannot be traced to a licensed source", "", 0)
print("mage squad", parts, flush=True)

# ---------------------------------------------------------------- Heterogeneous AI Spans (no JMLR)
tot = kept = 0
for sp in ("train", "validation", "test"):
    s = pd.read_parquet(f"{RAW}/spans/data/all/{sp}.parquet"); tot += len(s)
    s = s[s.source_dataset != "jmlr_pre2015"]; kept += len(s); s.to_parquet(f"{OUT}/spans_{sp}.parquet")
add("Heterogeneous AI Spans / Gutenberg, Standard Ebooks, Beige Book, WikiText (Wikipedia), Hansard", "dataset sources", "other: mixed-source-terms", "keep",
    "public domain (Gutenberg, Standard Ebooks CC0 edition, Federal Reserve), CC BY-SA/GFDL (Wikipedia), Open Parliament Licence; AI spans written with Claude and Codex", "https://huggingface.co/datasets/open-text-detector/heterogeneous-ai-spans", kept, tot)
add("Heterogeneous AI Spans / JMLR excerpts", "dataset source", "paper-specific licence unconfirmed", "drop", "the dataset's own LICENSE.md says the licence of each historical JMLR paper was not established", "", 0)
print("spans", kept, "of", tot, flush=True)

# ---------------------------------------------------------------- TextSight 2026 (eval)
import shutil
shutil.copytree(f"{RAW}/textsight/data", f"{OUT}/textsight", dirs_exist_ok=True)
add("TextSight 2026 (eval)", "dataset", "CC BY-SA 4.0", "keep", "components: Wikipedia CC BY-SA, PMC open access CC BY 4.0, public-domain literature; AI passages generated for the benchmark", "https://huggingface.co/datasets/textsightai/ai-text-detection-benchmark-2026", 700, 700)

# ---------------------------------------------------------------- WasItAIGenerated 2026-09 (eval)
w = pd.read_parquet(hf_hub_download("wasitaigeneratedcom/ai-detector-benchmark-2026-09", "data/benchmark.parquet", repo_type="dataset"))
ok = w.text_available.fillna(False) & w.text_license.isin(["CC-BY-4.0", "MIT"]) & (w.lang == "en") & w.label.isin(["human", "ai", "humanized"])
w2 = w[ok].copy(); w2["group"] = w2.group.astype(str)
w2.to_parquet(f"{OUT}/wasit.parquet")
add("WasItAIGenerated AI-detector benchmark 2026-09 (eval)", "dataset", "per-row licence (CC-BY-4.0, MIT, CC-BY-NC-SA, not redistributed)", "keep (rows)", "kept English rows whose own licence is CC-BY-4.0 or MIT; CC-BY-NC-SA and non-redistributable rows dropped", "https://huggingface.co/datasets/wasitaigeneratedcom/ai-detector-benchmark-2026-09", int(ok.sum()), len(w))
add("WasItAIGenerated / CC-BY-NC-SA rows", "dataset rows", "CC BY-NC-SA 4.0", "drop", "non-commercial", "", 0, int((w.text_license == "CC-BY-NC-SA-4.0").sum()))

# ---------------------------------------------------------------- VeriGuard humanized (eval)
v = pd.read_csv(hf_hub_download("Shivaay200005/veriguard-ai-text-detection", "data/veriguard_humanized.csv", repo_type="dataset"))
v = v[v.licence.isin(["CC-BY-4.0", "MIT"])]
ov = set(map(nh, v.text)) & set(map(nh, w2.text)); print("veriguard rows", len(v), "text overlap with wasit", len(ov), flush=True)
add("VeriGuard humanized table (eval)", "dataset", "per-row CC-BY-4.0 / MIT", "drop", f"{len(ov)} of its {len(v)} licensed rows are copies of WasItAIGenerated rows already used, so it adds nothing as a separate set", "https://huggingface.co/datasets/Shivaay200005/veriguard-ai-text-detection", 0, 1228)

# ---------------------------------------------------------------- dropped wholesale
for name, declared, why, url in [
    ("M4GT-Bench via COLING-2025 MGT (en)", "none declared", "no licence on the card or the M4 repository; includes WikiHow (CC BY-NC-SA), Reddit, PeerRead and OUTFOX/PERSUADE (CC BY-NC-SA) text", "https://huggingface.co/datasets/Jinyan1/COLING_2025_MGT_en"),
    ("SemEval-2024 task 8 (A and C, d0rj copy)", "Apache-2.0 (uploader's copy)", "derived from M4, which declares no licence; subtask A test is OUTFOX/PERSUADE essays (CC BY-NC-SA) and subtask C is PeerRead", "https://huggingface.co/datasets/d0rj/SemEval2024-task8"),
    ("HC3", "CC BY-SA 4.0", "the card says stricter source licences apply; sources include WikiQA (Microsoft research-only), MedDialog and FiQA", "https://huggingface.co/datasets/Hello-SimpleAI/HC3"),
    ("RoFT", "MIT", "human text is RecipeNLG (non-commercial), Reddit stories and New York Times articles, plus presidential speeches of unconfirmed terms", "https://huggingface.co/datasets/liamdugan/roft"),
    ("johnson/AI-writing-detector-dataset", "Apache-2.0", "sentences of arXiv full-text papers 2007 to 2017 with per-paper author licences (mostly the arXiv non-exclusive licence) that the card does not trace", "https://huggingface.co/datasets/johnson/AI-writing-detector-dataset"),
    ("author-ai-text-detect/longlamp-ai-detection-gpt5mini", "other", "built on LongLaMP (Amazon reviews, Reddit, arXiv); the card tells users to check upstream terms", "https://huggingface.co/datasets/author-ai-text-detect/longlamp-ai-detection-gpt5mini"),
    ("delvin-ays/ai-text-detection-training", "none declared", "no licence", "https://huggingface.co/datasets/delvin-ays/ai-text-detection-training"),
    ("HumanizerBench, jialinyyzz/humanizer-data, gohumanize, Lelonthecodeur/humanizer-5m (inspected, never used)", "mixed / none", "not used", "")]:
    add(name, "dataset", declared, "drop", why, url, 0)

# ---------------------------------------------------------------- models
for name, lic, dec, why, url in [
    ("voyage-4-nano (backbone)", "Apache-2.0 (LICENSE file next to the weights)", "keep", "", ""),
    ("Qwen3-Embedding-0.6B (baseline backbone)", "Apache-2.0", "keep", "", "https://huggingface.co/Qwen/Qwen3-Embedding-0.6B"),
    ("Qwen3-0.6B-Base (baseline backbone)", "Apache-2.0", "keep", "", "https://huggingface.co/Qwen/Qwen3-0.6B-Base"),
    ("desklib/ai-text-detector-v1.01 (comparison only)", "MIT", "keep", "run as a black box for comparison; its training data is undisclosed", "https://huggingface.co/desklib/ai-text-detector-v1.01"),
    ("fakespot-ai/roberta-base-ai-text-detection-v1 (comparison only)", "Apache-2.0", "keep", "run as a black box for comparison", "https://huggingface.co/fakespot-ai/roberta-base-ai-text-detection-v1"),
    ("openai-community/roberta-large-openai-detector (comparison only)", "MIT", "keep", "run as a black box for comparison", "https://huggingface.co/openai-community/roberta-large-openai-detector"),
    ("Hello-SimpleAI/chatgpt-detector-roberta (comparison only)", "none declared", "drop", "no licence on the model card", "https://huggingface.co/Hello-SimpleAI/chatgpt-detector-roberta")]:
    add(name, "model", lic, dec, why, url)
json.dump(audit, open(f"{OUT}/license_audit.json", "w"), indent=1)
if "--purge-raw" in sys.argv:  # remove the raw copies that still contain unlicensed domains / generators
    import shutil
    for k in ("raid", "mage", "spans", "textsight"):
        shutil.rmtree(f"{RAW}/{k}", ignore_errors=True)
    print("purged raw copies")
print("audit entries", len(audit), "| kept:", sum(a["decision"].startswith("keep") for a in audit), "dropped:", sum(a["decision"] == "drop" for a in audit))
