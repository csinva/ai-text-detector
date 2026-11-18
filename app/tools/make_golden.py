#!/usr/bin/env python3
"""Golden data for the browser tests: tokenizer ids and offsets from the `tokenizers` library, and model outputs (int8 ONNX in python) for the sample texts.
python app/tools/make_golden.py   ->  app/tools/golden.json"""
import json, os, sys
import numpy as np, onnxruntime as ort
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "..", "aidetect"))
from infer import TOK, token_to_word, windows
from tokenizers import Tokenizer

tk = Tokenizer.from_file(TOK); sess = ort.InferenceSession(os.path.join(HERE, "..", "model", "detector.onnx"), providers=["CPUExecutionProvider"])
js = open(os.path.join(HERE, "..", "js", "app.js")).read()
S = json.loads(js[js.index("const SAMPLES = ") + 16: js.index("};\nlet ready") + 1].replace("\n", " ")) if False else None
import re
samples = {k: bytes(v, "utf-8").decode("unicode_escape") if False else v for k, v in re.findall(r'^\s*(human|ai|mixed): "(.*)",?$', js, flags=re.M)}
edge = ["Hello, world!", "Don't stop; it's 3.14159 -- isn't it?\n\nNew paragraph.\n", "  leading and trailing   spaces  ", "naïve café – résumé ünïcode ﬁ", "你好，世界！ Mixed 中文 and English.", "def f(x):\n    return x**2  # square\n", "URL https://example.com/a_b?c=d&e=f#frag and e-mail a.b@c.org",
        "I'M LOUD, I'LL SHOUT, THEY'RE HERE", "tabs\tand\r\nwindows newlines\r\n", "Emoji 🙂 test 👍🏽 done", "x" * 300, "1234567890 " * 20, "á combining é (NFC)", ""]
tok = [dict(text=t, ids=tk.encode(t).ids, offsets=tk.encode(t).offsets) for t in list(samples.values()) + edge]
res = {}
for k, t in samples.items():
    enc = tk.encode(t); ids = np.array(enc.ids, dtype=np.int64); n = len(ids); ts = np.zeros(n); tc = np.zeros(n); dl = dw = 0.0
    for s, e in windows(n, 1024):
        a, b = sess.run(None, {"ids": ids[s:e]}); ts[s:e] += 1 / (1 + np.exp(-a)); tc[s:e] += 1; dl += float(b) * (e - s); dw += e - s
    tp = ts / np.maximum(tc, 1); res[k] = dict(text=t, doc_logit=dl / dw, doc_prob=1 / (1 + np.exp(-dl / dw)), tokens=n, words=token_to_word(t, enc.offsets, tp))
json.dump(dict(tokenizer=tok, model=res), open(os.path.join(HERE, "golden.json"), "w"), ensure_ascii=False)
print({k: (round(v["doc_prob"], 3), v["tokens"]) for k, v in res.items()}, len(tok), "tokenizer cases")
