"""Second search round: recent (since a cutoff) detection datasets with permissive licences, plus openly licensed text corpora families.
python 15_search2.py [cutoff YYYY-MM-DD]"""
import sys, datetime as dt
from huggingface_hub import HfApi

a = HfApi()
cut = dt.datetime.fromisoformat(sys.argv[1] if len(sys.argv) > 1 else "2026-10-01").replace(tzinfo=dt.timezone.utc)
OK = {"cc0-1.0", "cc-by-4.0", "cc-by-3.0", "cc-by-sa-4.0", "cc-by-sa-3.0", "mit", "apache-2.0", "odc-by", "pddl", "unlicense", "bsd-3-clause", "bsd-2-clause", "cdla-permissive-2.0", "cdla-permissive-1.0", "openrail", "other"}
lic = lambda d: [t.split(":", 1)[1] for t in (d.tags or []) if t.startswith("license:")]
print("== recent detection-related datasets with a permissive or unspecified-'other' licence (created >= cutoff)")
seen = {}
for q in ["ai text detection", "ai-generated text", "machine-generated text", "human vs ai", "llm-generated", "ai detector benchmark", "humanizer", "humanized text", "paraphrased ai", "mixed authorship", "ai-edited",
          "ai polished", "authorship llm", "detect llm", "ai essays", "ai reviews", "synthetic vs human", "human written", "ai or human", "genai detection", "gpt detection", "text provenance",
          "ai writing", "llm text classification", "token-level ai detection", "span detection ai", "boundary detection machine generated", "watermarked text"]:
    for sort in ("createdAt", "downloads"):
        try:
            for d in a.list_datasets(search=q, sort=sort, direction=-1, limit=50):
                if d.created_at and d.created_at >= cut and d.id not in seen and (set(lic(d)) & OK):
                    seen[d.id] = (d.created_at.date().isoformat(), d.downloads, ",".join(lic(d)))
        except Exception as e:
            print("err", q, str(e)[:60])
for k, v in sorted(seen.items(), key=lambda x: (-x[1][1], x[0])): print(f"{k:70s} {v[0]} dl={v[1]:<6} {v[2]}")
print(len(seen), "found\n\n== openly licensed text corpus families (any date)")
for q in ["common corpus", "common-corpus", "kl3m", "open license corpus", "public domain corpus", "openly licensed", "pd books", "institutional books", "pre-1950", "wikisource", "wikinews", "wikibooks", "wikiquote",
          "parliament debates", "congressional record", "subtitles", "podcast", "oral history", "letters", "newspapers public domain", "chronicling america", "legal opinions", "patents", "textbooks open", "oer"]:
    try:
        for d in a.list_datasets(search=q, sort="downloads", direction=-1, limit=6):
            if set(lic(d)) & (OK - {"other"}) and d.downloads > 1500: print(f"{d.id:62s} dl={d.downloads:<8} {','.join(lic(d))[:40]}   <- {q}")
    except Exception as e:
        pass
