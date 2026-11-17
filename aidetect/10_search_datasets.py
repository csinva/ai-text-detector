"""List recently created Hugging Face datasets about AI-generated text detection (created on or after a cutoff date).
python 10_search_datasets.py [cutoff YYYY-MM-DD]"""
import sys, datetime as dt
from huggingface_hub import HfApi

cut = dt.datetime.fromisoformat(sys.argv[1] if len(sys.argv) > 1 else "2026-09-15").replace(tzinfo=dt.timezone.utc)
a = HfApi(); seen = {}
qs = ["ai generated text", "ai-generated", "machine-generated", "llm-generated", "human vs ai", "ai text detection", "ai detector", "detect ai", "aigt", "mgt",
      "humanizer", "humanized", "paraphrase attack", "mixed authorship", "hybrid text", "ai essays", "ai reviews", "ai written", "gpt-5", "claude", "gemini 3",
      "authorship", "ai content detection", "synthetic text detection", "watermark", "llm text", "human-written", "ai or human", "turing"]
for q in qs:
    for sort in ("createdAt", "lastModified"):
        try:
            for d in a.list_datasets(search=q, sort=sort, direction=-1, limit=60):
                if d.created_at and d.created_at >= cut and d.id not in seen:
                    seen[d.id] = (d.created_at.date().isoformat(), d.downloads, d.gated, ",".join(t for t in (d.tags or []) if t.startswith("license:")))
        except Exception as e:
            print("err", q, e)
for k, v in sorted(seen.items(), key=lambda x: (-x[1][1], x[0])):
    print(k, *v)
print(len(seen), "datasets")
