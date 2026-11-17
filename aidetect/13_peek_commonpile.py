"""Stream a couple of rows from Common Pile subsets: columns, metadata keys, licence and date fields. python 13_peek_commonpile.py"""
import json, itertools
from datasets import load_dataset

SUBS = ["stackexchange_filtered", "ubuntu_irc_filtered", "youtube_filtered", "news_filtered", "wikimedia_filtered", "pubmed_filtered", "arxiv_abstracts_filtered",
        "project_gutenberg_filtered", "pre_1929_books_filtered", "foodista_filtered", "oercommons_filtered", "doab_filtered", "uk_hansard_filtered", "usgpo_filtered",
        "libretexts_filtered", "pressbooks_filtered", "public_domain_review_filtered", "python_enhancement_proposals_filtered", "regulations_filtered", "library_of_congress_filtered"]
for s in SUBS:
    try:
        ds = load_dataset(f"common-pile/{s}", split="train", streaming=True)
        rows = list(itertools.islice(ds, 3))
        r = rows[0]; meta = r.get("metadata")
        meta = json.loads(meta) if isinstance(meta, str) else meta
        print(f"== {s}: cols={list(r.keys())} chars={[len(x['text']) for x in rows]} source={r.get('source')} added={str(r.get('added'))[:10]} created={str(r.get('created'))[:10]}")
        print("   metadata:", json.dumps(meta)[:330] if meta else None)
    except Exception as e:
        print(f"== {s}: ERR {str(e)[:150]}")
