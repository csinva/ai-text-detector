#!/usr/bin/env python3
"""Document-level scores of open-source detectors on the same held-out sets (max 512 tokens, as these models were trained).

  python 05_baselines.py --out baselines.json [--models desklib,fakespot,hello]
"""
import argparse, json, os, sys, time
import numpy as np, pandas as pd, torch
from transformers import AutoConfig, AutoModel, AutoModelForSequenceClassification, AutoTokenizer, PreTrainedModel

sys.path.insert(0, os.path.dirname(__file__))
from importlib import import_module
ev = import_module("04_eval")

REPOS = {"desklib": "desklib/ai-text-detector-v1.01", "fakespot": "fakespot-ai/roberta-base-ai-text-detection-v1",
         "openai-large": "openai-community/roberta-large-openai-detector"}


class Desklib(PreTrainedModel):  # architecture of the model card: backbone + mean pooling + linear head
    config_class = AutoConfig

    def __init__(self, config):
        super().__init__(config)
        self.model = AutoModel.from_config(config)
        self.classifier = torch.nn.Linear(config.hidden_size, 1)
        self.post_init()

    def forward(self, input_ids, attention_mask):
        h = self.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        m = attention_mask.unsqueeze(-1).to(h.dtype)
        return self.classifier((h * m).sum(1) / m.sum(1).clamp(min=1)).squeeze(-1)


def load(key):
    repo = REPOS[key]
    tok = AutoTokenizer.from_pretrained(repo)
    if key == "desklib":
        m = Desklib.from_pretrained(repo)
        return tok, m.cuda().eval().half(), lambda out: torch.sigmoid(out.float())
    m = AutoModelForSequenceClassification.from_pretrained(repo).cuda().eval().half()
    lab = {int(k): v.lower() for k, v in m.config.id2label.items()}
    ai = [k for k, v in lab.items() if any(s in v for s in ("fake", "ai", "chatgpt", "machine", "generated"))]
    ai = ai[0] if ai else 1
    if key == "openai-large":  # labels are only label_0 / label_1; label_0 is the machine-generated class (a score below chance with index 1 confirmed the orientation)
        ai = 0
    print(key, "labels", lab, "AI index", ai)
    return tok, m, lambda out: torch.softmax(out.logits.float(), -1)[:, ai]


@torch.no_grad()
def score(tok, m, post, texts, bs=64):
    order = np.argsort([len(t) for t in texts]); out = np.zeros(len(texts))
    for i in range(0, len(texts), bs):
        idx = order[i:i + bs]
        enc = tok([texts[j] for j in idx], truncation=True, max_length=512, padding=True, return_tensors="pt").to("cuda")
        o = m(**enc) if not isinstance(m, Desklib) else m(enc["input_ids"], enc["attention_mask"])
        out[idx] = post(o).cpu().numpy()
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--out"); ap.add_argument("--models", default="desklib,fakespot,openai-large")
    ap.add_argument("--sets", default=",".join(ev.DOC_SETS))
    a = ap.parse_args()
    res = json.load(open(a.out)) if a.out and os.path.exists(a.out) else {}  # re-running a subset of models updates the file
    for key in a.models.split(","):
        tok, m, post = load(key)
        res[key] = {}
        for name in a.sets.split(","):
            df = pd.read_parquet(f"{ev.D}/prep_lic3/eval/{name}.parquet")
            t0 = time.time(); p = score(tok, m, post, df.text.tolist())
            res[key][name] = dict(doc=ev.doc_metrics(df.label.values, p), seconds=time.time() - t0, **ev.breakdown(df, p, skip=name == "spans_test"))
            print(key, name, {k: round(v, 1) for k, v in res[key][name]["doc"].items()}, flush=True)
        del m; torch.cuda.empty_cache()
    if a.out: json.dump(res, open(a.out, "w"), indent=1)
