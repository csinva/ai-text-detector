"""Round 3, part 1: wide sweep for AI-text detection datasets (any date) with a permissive or 'other' licence tag, ranked by downloads.
python 16_search_eval.py [min_downloads]"""
import re, sys
from huggingface_hub import HfApi

a = HfApi(); mind = int(sys.argv[1]) if len(sys.argv) > 1 else 30
OK = {"cc0-1.0", "cc-by-4.0", "cc-by-3.0", "cc-by-2.0", "cc-by-sa-4.0", "cc-by-sa-3.0", "mit", "apache-2.0", "odc-by", "pddl", "unlicense", "bsd-3-clause", "bsd-2-clause", "cdla-permissive-2.0", "cdla-permissive-1.0", "openrail", "other", "afl-3.0", "artistic-2.0", "wtfpl", "zlib", "bigscience-openrail-m"}
lic = lambda d: [t.split(":", 1)[1] for t in (d.tags or []) if t.startswith("license:")]
KEY = re.compile(r"detect|generated|human|gpt|llm|machine|humaniz|paraphras|authorship|turing|aigt|mgt|ai[-_ ]?text|ai[-_ ]?written|synthetic|chatgpt|deepfake|watermark|raid|mage|m4|ghostbuster|essay", re.I)
qs = ["ai generated text detection", "machine generated text detection", "ai text detector", "human vs ai text", "human vs machine", "human vs chatgpt", "chatgpt generated text", "llm generated text", "ai-generated essays",
      "ai generated abstracts", "ai generated reviews", "ai generated news", "ai generated code detection", "mixed human ai text", "ai edited text", "paraphrased ai text", "humanized text", "ai humanizer",
      "authorship attribution llm", "llm authorship", "gpt wiki intro", "wikipedia gpt", "text detection benchmark", "detector benchmark", "detection robustness attack", "adversarial machine generated text",
      "multi-generator detection", "domain shift detection ai text", "turingbench", "ghostbuster", "detectrl", "mgtbench", "opengpttext", "ai text pile", "daigt", "llm detect ai generated text", "persuade ai",
      "essay detection llm", "student essays ai", "ai peer review detection", "ai tweets", "social media ai generated", "fake reviews llm", "synthetic news detection", "ai scientific text detection", "ai text sentence level",
      "token level machine generated", "boundary machine generated text", "hybrid authorship", "co-written text", "human-ai collaborative writing", "writing assistance detection", "polished by llm", "grammarly ai", "ai rewritten"]
seen = {}
for q in qs:
    for sort in ("downloads", "createdAt", "likes"):
        try:
            for d in a.list_datasets(search=q, sort=sort, direction=-1, limit=40):
                if d.id in seen or d.downloads < mind or not (set(lic(d)) & OK) or not KEY.search(d.id): continue
                seen[d.id] = (d.downloads, d.created_at.date().isoformat() if d.created_at else "?", ",".join(lic(d))[:28])
        except Exception as e:
            print("err", q, str(e)[:60])
for k, v in sorted(seen.items(), key=lambda x: -x[1][0]): print(f"{k:66s} dl={v[0]:<8} {v[1]} {v[2]}")
print(len(seen), "candidates")
