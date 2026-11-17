"""Append the Common Pile subsets and the generator models to licensed/license_audit.json (idempotent). python 14_audit_extra.py"""
import json, os
import pandas as pd

D = os.path.join(os.environ["V45_DATA_ROOT"], "data/aidetect"); P = f"{D}/licensed/license_audit.json"
audit = [a for a in json.load(open(P)) if not a.get("extra")]
h = pd.read_parquet(f"{D}/licensed/human_cp.parquet"); cnt = h.groupby("src").size().to_dict()
g = pd.concat([pd.read_parquet(f"{D}/licensed/gen/{f}") for f in sorted(os.listdir(f"{D}/licensed/gen")) if f.endswith(".parquet") and "smoke" not in f])
gcnt = g.groupby("gen_model").size().to_dict()
CP = "https://huggingface.co/common-pile"
add = lambda **k: audit.append(dict(extra=True, url=k.pop("url", ""), total=None, **k))
for src, sub, lic, why in [
    ("stackexchange", "stackexchange_filtered", "per-document CC BY-SA 3.0 / 4.0", "Stack Exchange posts created in 2017 or earlier; the licence of every document is in its metadata"),
    ("arxiv_abstract", "arxiv_abstracts_filtered", "CC0 (arXiv metadata)", "abstracts of papers submitted in 2017 or earlier"),
    ("pubmed", "pubmed_filtered", "per-document CC BY (NonCommercial variants excluded)", "PubMed Central open-access articles from 2017 or earlier, one random 500 to 2,200 character passage"),
    ("gutenberg", "project_gutenberg_filtered", "public domain", "Project Gutenberg books (public domain in the USA), random passages"),
    ("pre1929_books", "pre_1929_books_filtered", "public domain", "books published before 1929 (HathiTrust / Internet Archive), random passages"),
    ("uk_hansard", "uk_hansard_filtered", "Open Parliament Licence", "UK Parliament debates and written answers from 2017 or earlier"),
    ("foodista", "foodista_filtered", "CC BY 3.0", "Foodista recipe and food-blog pages from 2017 or earlier")]:
    add(name=f"Common Pile / {sub}", kind="human text (negatives)", declared=lic, decision="keep", why=why, url=f"{CP}/{sub}", kept=int(cnt.get(src, 0)))
add(name="Common Pile / youtube, news, ubuntu_irc, oercommons, libretexts, pressbooks, doab, wikimedia, caselaw, usgpo, regulations, uspto, peS2o, github", kind="human text", declared="per-document open licences",
    decision="not used", why="inspected; left out for time or because their documents are mostly dated after 2017 (so AI-contamination cannot be excluded) or are formal registers already covered", url=CP, kept=0)
for model, tag, lic in [("Qwen/Qwen3-8B", "qwen3_8b", "Apache-2.0"), ("Qwen/Qwen3-4B-Instruct-2507", "qwen3_4b", "Apache-2.0"), ("mistralai/Mistral-7B-Instruct-v0.3", "mistral7b", "Apache-2.0"),
                        ("microsoft/Phi-4-mini-instruct", "phi4mini", "MIT"), ("allenai/OLMo-2-1124-7B-Instruct", "olmo2_7b", "Apache-2.0"), ("HuggingFaceTB/SmolLM3-3B", "smollm3", "Apache-2.0"), ("openai/gpt-oss-20b", "gptoss20b", "Apache-2.0"), ("Qwen/Qwen3-30B-A3B-Instruct-2507", "qwen3_30b", "Apache-2.0")]:
    add(name=f"{model} (generator)", kind="model / machine text", declared=lic, decision="keep", why="open weights under a commercial-use licence; its generated passages are used as AI text", url=f"https://huggingface.co/{model}", kept=int(gcnt.get(tag, 0)))
for name, declared, why, url in [
    ("Jerry999/AIWritingEffect (second search, Oct 2026)", "CC BY 4.0 (card)", "human side is ICLR / NeurIPS paper text: authors keep the copyright and the venues only hold a non-exclusive licence, and the card traces no per-paper CC licence for the human paragraphs", "https://huggingface.co/datasets/Jerry999/AIWritingEffect"),
    ("avabbas/when-ai-reviews-itself", "CC BY 4.0", "study of AI-written peer reviews, not a detection dataset", "https://huggingface.co/datasets/avabbas/when-ai-reviews-itself"),
    ("JaberKhan011/AI_vs_Human_Text, shamimhasan8 copies, kazakh sets", "MIT (cards)", "provenance of the human text is not given (Kaggle-derived), so its rights cannot be traced", "https://huggingface.co/datasets/JaberKhan011/AI_vs_Human_Text"),
    ("alea-institute/kl3m, LoC PD books, Chronicling America, Wikisource", "CC BY 4.0 / CC0 / CC BY-SA", "openly licensed human text found in the second search; not used (formal or OCR-noisy registers that the corpus already covers, and OCR noise would be a shortcut)", "https://huggingface.co/alea-institute")]:
    add(name=name, kind="dataset", declared=declared, decision="drop" if "AI" in name or "avabbas" in name or "JaberKhan" in name else "not used", why=why, url=url, kept=0)
add(name="vLLM (inference engine for generation)", kind="software", declared="Apache-2.0", decision="keep", why="", url="https://github.com/vllm-project/vllm", kept=None)
json.dump(audit, open(P, "w"), indent=1)
print("audit entries", len(audit), "human by src", cnt, "gen by model", gcnt)
