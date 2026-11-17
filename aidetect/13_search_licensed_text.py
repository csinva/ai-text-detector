"""Search the Hub for openly licensed text corpora (human negatives) and list Common Pile subsets. python 13_search_licensed_text.py"""
from huggingface_hub import HfApi

a = HfApi()
OK = ("license:cc0-1.0", "license:cc-by-4.0", "license:cc-by-3.0", "license:cc-by-sa-4.0", "license:cc-by-sa-3.0", "license:mit", "license:apache-2.0", "license:odc-by", "license:pddl", "license:unlicense", "license:other")
print("== common-pile org")
for d in a.list_datasets(author="common-pile", limit=100, sort="downloads", direction=-1):
    lic = [t for t in (d.tags or []) if t.startswith("license")]
    print(f"{d.id:55s} {d.downloads:>9} {lic}")
print("\n== keyword search (permissive license tags only)")
seen = set()
for q in ["public domain text", "openly licensed text", "cc-by text corpus", "pre-2018", "books public domain", "gutenberg", "stackexchange", "wikipedia", "forum posts", "chat conversations",
          "blog posts", "news articles cc-by", "transcripts", "emails", "essays", "reviews cc-by", "tweets", "informal text", "reddit", "hacker news", "podcast transcripts", "student essays", "speeches",
          "legal text", "patents", "government documents", "scientific abstracts", "pubmed", "open access", "creative commons text"]:
    try:
        for d in a.list_datasets(search=q, sort="downloads", direction=-1, limit=15):
            lic = [t for t in (d.tags or []) if t.startswith("license:")]
            if d.id in seen or not any(l in OK for l in lic) or d.downloads < 800: continue
            seen.add(d.id); print(f"{d.id:60s} {d.downloads:>9} {lic[:2]}  <- {q}")
    except Exception as e:
        print("err", q, str(e)[:80])
