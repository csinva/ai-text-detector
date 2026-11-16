"""Download the open datasets used for the AI-text detector into $V45_DATA_ROOT/data/aidetect/raw."""
import os, sys
from huggingface_hub import snapshot_download
raw = os.path.join(os.environ["V45_DATA_ROOT"], "data/aidetect/raw")
SETS = {
  "raid": ("liamdugan/raid", ["train.csv"]),
  "mage": ("yaful/MAGE", ["*.csv"]),
  "roft": ("liamdugan/roft", ["*.csv"]),
  "coling_en": ("Jinyan1/COLING_2025_MGT_en", ["data/*"]),
  "semeval8": ("d0rj/SemEval2024-task8", ["*.parquet"]),
  "hc3": ("Hello-SimpleAI/HC3", ["*.jsonl"]),
}
for k in (sys.argv[1:] or SETS):
    rid, pats = SETS[k]
    p = snapshot_download(rid, repo_type="dataset", local_dir=f"{raw}/{k}", allow_patterns=pats)
    print("done", k, p, flush=True)
