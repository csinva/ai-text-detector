#!/usr/bin/env python3
"""Split app/model/detector.onnx into 48 MB shards (GitHub Pages serves files under 100 MB) and write manifest.json (names, sizes, sha256, version).
The browser downloads the shards, stores them in Cache Storage and concatenates them. python app/tools/shard.py [--chunk-mb 48]"""
import argparse, glob, hashlib, json, os

HERE = os.path.dirname(os.path.abspath(__file__)); M = os.path.join(HERE, "..", "model")
ap = argparse.ArgumentParser(); ap.add_argument("--chunk-mb", type=int, default=48); a = ap.parse_args()
src = os.path.join(M, "detector.onnx"); data = open(src, "rb").read(); n = a.chunk_mb * 1024 * 1024
for f in glob.glob(os.path.join(M, "detector.onnx.part*")): os.remove(f)
parts = []
for i in range(0, len(data), n):
    name = f"detector.onnx.part{i // n:03d}"; chunk = data[i:i + n]; open(os.path.join(M, name), "wb").write(chunk)
    parts.append(dict(name=name, bytes=len(chunk), sha256=hashlib.sha256(chunk).hexdigest()))
cfg = json.load(open(os.path.join(M, "config.json")))
man = dict(version=hashlib.sha256(data).hexdigest()[:12], total_bytes=len(data), parts=parts, tokenizer="tokenizer.json", tokenizer_bytes=os.path.getsize(os.path.join(M, "tokenizer.json")), window=cfg["window"], source=cfg["source"])
json.dump(man, open(os.path.join(M, "manifest.json"), "w"), indent=1); print(len(parts), "shards, version", man["version"])
