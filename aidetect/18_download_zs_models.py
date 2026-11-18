#!/usr/bin/env python3
"""Download the checkpoints of the zero-shot baselines (all Apache-2.0 or MIT) to $V45_DATA_ROOT/data/aidetect/cache/hf. python 18_download_zs_models.py"""
from huggingface_hub import snapshot_download
for repo, rev, pats in [("tiiuae/falcon-7b", None, ["*.safetensors", "*.json", "*.txt"]), ("tiiuae/falcon-7b-instruct", None, ["*.safetensors", "*.json", "*.txt"]),
                        ("EleutherAI/gpt-j-6b", "float16", ["pytorch_model.bin", "*.json", "*.txt"]), ("EleutherAI/gpt-neo-2.7B", None, ["model.safetensors", "*.json", "*.txt"]),
                        ("google-t5/t5-large", None, ["model.safetensors", "*.json", "*.model"])]:
    print(repo, snapshot_download(repo, revision=rev, allow_patterns=pats), flush=True)
