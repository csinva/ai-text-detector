"""Collect the declared licence and licence-related README lines of every dataset / model used. python 11_license_cards.py"""
import re
from huggingface_hub import HfApi, hf_hub_download

a = HfApi()
DS = ["liamdugan/raid", "yaful/MAGE", "Jinyan1/COLING_2025_MGT_en", "d0rj/SemEval2024-task8", "Hello-SimpleAI/HC3", "liamdugan/roft",
      "open-text-detector/heterogeneous-ai-spans", "textsightai/ai-text-detection-benchmark-2026", "johnson/AI-writing-detector-dataset",
      "wasitaigeneratedcom/ai-detector-benchmark-2026-09", "Shivaay200005/veriguard-ai-text-detection", "author-ai-text-detect/longlamp-ai-detection-gpt5mini"]
MD = ["Qwen/Qwen3-Embedding-0.6B", "Qwen/Qwen3-0.6B-Base", "desklib/ai-text-detector-v1.01", "fakespot-ai/roberta-base-ai-text-detection-v1",
      "Hello-SimpleAI/chatgpt-detector-roberta", "openai-community/roberta-large-openai-detector"]
KEY = re.compile(r"licen|non-?commercial|research (use|purposes)|terms of|copyright|public domain|CC[ -]BY|attribution|redistribut|source|upstream|reddit|amazon|yelp|wiki|gutenberg|arxiv", re.I)
for kind, repos in (("dataset", DS), ("model", MD)):
    for r in repos:
        print("=" * 90); print(kind, r)
        try:
            i = a.dataset_info(r) if kind == "dataset" else a.model_info(r)
        except Exception as e:
            print("  ERR", str(e)[:100]); continue
        print("  tags:", [t for t in i.tags if t.startswith("license")], "| card license:", (i.card_data.license if i.card_data and hasattr(i.card_data, "license") else None))
        try:
            t = open(hf_hub_download(r, "README.md", repo_type=kind)).read()
        except Exception:
            print("  no README"); continue
        lines = [l.strip() for l in t.split("\n") if KEY.search(l) and len(l.strip()) > 20 and not l.strip().startswith(("path:", "- split", "data_files"))]
        for l in lines[:14]:
            print("   >", l[:260])
