"""Download the sources that survive the licence audit into $V45_DATA_ROOT/data/aidetect/raw (then run 12_licensed_data.py --purge-raw).

RAID and MAGE ship as single files that also contain domains and generators the audit drops; 12_licensed_data.py filters them and, with
--purge-raw, deletes the raw copies. Sources dropped for licence reasons (M4GT / COLING-2025, SemEval-2024-8, HC3, RoFT, johnson, LongLaMP)
are deliberately not downloaded. WasItAIGenerated is read straight from the Hub by 12_licensed_data.py.
"""
import os, sys
from huggingface_hub import snapshot_download
raw = os.path.join(os.environ["V45_DATA_ROOT"], "data/aidetect/raw")
SETS = {
  "raid": ("liamdugan/raid", ["train.csv"]),
  "mage": ("yaful/MAGE", ["train.csv", "valid.csv", "test.csv"]),
  "spans": ("open-text-detector/heterogeneous-ai-spans", ["data/all/*.parquet", "README.md", "LICENSE.md"]),
  "textsight": ("textsightai/ai-text-detection-benchmark-2026", ["*"]),
}
for k in (sys.argv[1:] or SETS):
    rid, pats = SETS[k]
    p = snapshot_download(rid, repo_type="dataset", local_dir=f"{raw}/{k}", allow_patterns=pats)
    print("done", k, p, flush=True)
