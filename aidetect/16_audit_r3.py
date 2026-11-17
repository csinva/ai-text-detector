"""Round 3: licence audit entries for the new eval and human sources (idempotent). python 16_audit_r3.py"""
import json, os
P = os.path.join(os.environ["V45_DATA_ROOT"], "data/aidetect/licensed/license_audit.json"); a = json.load(open(P))
a = [x for x in a if not x["name"].startswith(("HumanizerBench, jialinyyzz", "HumanizerBench / ", "AI Humanizer Benchmark", "Common Pile / news", "Common Pile / usgpo"))]
a += [dict(name="HumanizerBench (eval)", kind="dataset", declared="CC BY 4.0", decision="keep", why="outputs of 19 commercial humanizers, June to November 2026, used for evaluation only", url="https://huggingface.co/datasets/HumanizerBench/humanizerbench", kept=2555, total=2577, extra=True),
      dict(name="AI Humanizer Benchmark (eval)", kind="dataset", declared="CC BY 4.0", decision="keep", why="outputs of 12 commercial humanizers, September to November 2026, used for evaluation only", url="https://huggingface.co/datasets/ai-humanizer-benchmark/ai-humanizer-benchmark", kept=1055, total=1056, extra=True),
      dict(name="jialinyyzz/humanizer-data, gohumanize, Lelonthecodeur/humanizer-5m", kind="dataset", declared="various", decision="drop", why="inspected, not used (licence or provenance not traceable)", url="", kept=0, total=None, extra=True),
      dict(name="Common Pile / news_filtered (Global Voices only)", kind="human text (negatives)", declared="CC BY 4.0 (per document)", decision="keep", why="Global Voices posts with a url year of 2017 or earlier (2004 to 2008); other hosts skipped", url="https://huggingface.co/common-pile/news_filtered", kept=10324, total=None, extra=True),
      dict(name="Common Pile / usgpo_filtered", kind="human text (negatives)", declared="Public Domain (US government works)", decision="keep", why="US GPO documents", url="https://huggingface.co/common-pile/usgpo_filtered", kept=4000, total=None, extra=True)]
json.dump(a, open(P, "w"), indent=1); print(len(a), "entries")
