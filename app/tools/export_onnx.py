#!/usr/bin/env python3
"""Export the detector to a browser-sized ONNX model (int8) for onnxruntime-web, and write the tokenizer files.

  python app/tools/export_onnx.py --model $V45_DATA_ROOT/data/aidetect/runs/lic3_nano_r10/model.pt --out app/model

Graph: ids (T,) int64 -> token logits (T,), document logit () for ONE window of at most 1,024 tokens (windowing, overlap averaging and
word aggregation are done in JavaScript exactly as in aidetect/infer.py). The forward mirrors model.py (bidirectional attention).

Quantization (checked against the fp32 torch model by app/tools/verify_onnx.py):
  * embedding table: int8 per row (frozen voyage-4-nano embeddings, 155M parameters), gather + scale inside the graph
  * qkv, attention output and MLP gate/up projections: onnxruntime dynamic int8 (int8 weights per channel, uint8 activations)
  * MLP down projections: weight-only int8 (DequantizeLinear, folded to fp32 when the session loads). Quantizing their activations
    shifts the document logit (see report 01, CPU section), so they keep fp32 activations.
  * norms, rotary tables and the two heads stay fp32.
"""
import argparse, json, os, sys
import numpy as np, onnx, torch, torch.nn as nn, torch.nn.functional as F
from onnx import TensorProto, helper, numpy_helper

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "..", "aidetect"))
from model import Cfg, Detector, apply_rope, rms  # noqa: E402

TOK = "/data/new/prod_models/voyage-4-nano/tokenizer.json"


class OnnxDet(nn.Module):
    def __init__(self, m: Detector, max_pos=1024):
        super().__init__()
        self.c = c = m.c; self.layers = m.layers; self.norm = m.norm; self.tok_head = m.tok_head; self.doc_head = m.doc_head
        w = m.embed.weight.detach().float(); s = (w.abs().amax(1, keepdim=True) / 127).clamp(min=1e-8)
        self.register_buffer("eq", torch.round(w / s).clamp(-127, 127).to(torch.int8)); self.register_buffer("es", s)
        inv = 1.0 / (c.theta ** (torch.arange(0, c.hd, 2, dtype=torch.float32) / c.hd)); f = torch.outer(torch.arange(max_pos, dtype=torch.float32), inv); e = torch.cat([f, f], -1)
        self.register_buffer("cos", e.cos()); self.register_buffer("sin", e.sin())

    def forward(self, ids):
        c = self.c; T = ids.shape[0]; rep = c.n_heads // c.n_kv
        x = self.eq[ids].float() * self.es[ids]
        cos, sin = self.cos[:T], self.sin[:T]
        for l in self.layers:
            q, k, v = l.pre_attn(x, cos, sin)                               # (T,H,hd), (T,Hkv,hd), (T,Hkv,hd)
            q = q.transpose(0, 1)
            k = k.transpose(0, 1)[:, None].expand(c.n_kv, rep, T, c.hd).reshape(c.n_heads, T, c.hd)
            v = v.transpose(0, 1)[:, None].expand(c.n_kv, rep, T, c.hd).reshape(c.n_heads, T, c.hd)
            a = torch.softmax(torch.matmul(q, k.transpose(1, 2)) * (c.hd ** -0.5), -1)
            x = l.post_attn(x, torch.matmul(a, v).transpose(0, 1))
        h = rms(x, self.norm, c.eps)
        return self.tok_head(h).squeeze(-1), self.doc_head(h.mean(0)).squeeze(-1)


def weight_only_int8(model: onnx.ModelProto, names):
    """Replace the float weight of the named MatMul nodes by int8 + per-output-channel scale + DequantizeLinear."""
    inits = {i.name: i for i in model.graph.initializer}; new_nodes = []; drop = set()
    for n in model.graph.node:
        if n.name in names:
            b = n.input[1]; arr = numpy_helper.to_array(inits[b]); s = np.maximum(np.abs(arr).max(0) / 127, 1e-12).astype(np.float32)   # (N,)
            q = np.clip(np.round(arr / s), -127, 127).astype(np.int8)
            model.graph.initializer.extend([numpy_helper.from_array(q, b + "_q"), numpy_helper.from_array(s, b + "_s")]); drop.add(b)
            new_nodes.append(helper.make_node("DequantizeLinear", [b + "_q", b + "_s"], [b + "_dq"], axis=1, name=n.name + "_dq"))
            n.input[1] = b + "_dq"
        new_nodes.append(n)
    del model.graph.node[:]; model.graph.node.extend(new_nodes)
    keep = [i for i in model.graph.initializer if i.name not in drop]; del model.graph.initializer[:]; model.graph.initializer.extend(keep)
    return model


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--out", default=os.path.join(HERE, "..", "model")); ap.add_argument("--tmp", default=None)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); tmp = a.tmp or a.out
    m = Detector(Cfg()); m.load_state_dict(torch.load(a.model, map_location="cpu"), strict=True); m.eval(); m.init_rope("cpu")
    net = OnnxDet(m).eval(); ids = torch.randint(0, 150000, (300,))
    with torch.no_grad(): ref = net(ids)
    fp32 = os.path.join(tmp, "detector_fp32.onnx")
    torch.onnx.export(net, (ids,), fp32, input_names=["ids"], output_names=["tok", "doc"], dynamic_axes={"ids": {0: "T"}, "tok": {0: "T"}}, opset_version=17, dynamo=False)
    from onnxruntime.quantization import QuantType, quantize_dynamic
    g = onnx.load(fp32, load_external_data=True); mm = [n for n in g.graph.node if n.op_type == "MatMul"]
    import re
    inits = {i.name for i in g.graph.initializer}
    down = [n.name for n in mm if re.match(r"^/down(_\d+)?/", n.name)]; heads = [n.name for n in mm if "head" in n.name]
    dyn = [n.name for n in mm if n.input[1] not in inits]   # attention score and value products (no constant weight)
    print("MatMul nodes", len(mm), "down", len(down), "heads", len(heads), "activation x activation", len(dyn))
    assert len(down) == 12 and len(dyn) == 24
    q8 = os.path.join(tmp, "detector_int8_tmp.onnx")
    quantize_dynamic(fp32, q8, weight_type=QuantType.QInt8, per_channel=True, op_types_to_quantize=["MatMul"], nodes_to_exclude=down + heads + dyn)
    g = onnx.load(q8); g = weight_only_int8(g, set(down))
    final = os.path.join(a.out, "detector.onnx"); onnx.save(g, final)
    print("wrote", final, round(os.path.getsize(final) / 1e6, 1), "MB")
    for f in (fp32, q8): os.remove(f)
    # tokenizer: only what the JS BPE needs (vocab, merges, special tokens)
    t = json.load(open(TOK)); merges = [m_ if isinstance(m_, str) else " ".join(m_) for m_ in t["model"]["merges"]]
    json.dump({"vocab": t["model"]["vocab"], "merges": merges, "added": [x["content"] for x in t["added_tokens"]]}, open(os.path.join(a.out, "tokenizer.json"), "w"), separators=(",", ":"), ensure_ascii=False)
    json.dump({"window": 1024, "layers": 12, "source": os.path.basename(os.path.dirname(a.model))}, open(os.path.join(a.out, "config.json"), "w"))
    print("tokenizer", round(os.path.getsize(os.path.join(a.out, "tokenizer.json")) / 1e6, 1), "MB")


if __name__ == "__main__":
    main()
