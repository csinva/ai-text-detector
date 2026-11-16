"""Check model.py against HF Qwen3 (causal mode): packed flash (bf16), SDPA (fp32) and CPU SDPA hidden states."""
import os, sys, torch
sys.path.insert(0, os.path.dirname(__file__))
from model import Cfg, Detector, pack
from transformers import AutoModel, AutoTokenizer

P = "/data/new/prod_models/voyage-4-nano"
tok = AutoTokenizer.from_pretrained(P)
hf = AutoModel.from_pretrained(P, torch_dtype=torch.float32).cuda().eval()
texts = ["The quick brown fox jumps over the lazy dog. It was a sunny day.",
         "Another sentence, a bit longer than the first one, to test packing across sequences of different lengths."]
m = Detector(Cfg(causal=True)).load_nano().cuda(); m.init_rope("cuda"); m.eval()
seqs = [tok(t).input_ids for t in texts]
ids, pos, cu, seg, ms, n = pack(seqs, "cuda")
refs = [hf(input_ids=torch.tensor([s]).cuda()).last_hidden_state[0] for s in seqs]
with torch.no_grad():
    for impl, dt in [("flash", torch.bfloat16), ("sdpa", torch.float32)]:
        m.attn_impl = impl
        with torch.autocast("cuda", dtype=dt, enabled=dt != torch.float32):
            h = m.hidden(ids, pos, cu, ms).float()
        off = 0
        for s, ref in zip(seqs, refs):
            print(impl, "max abs diff", (h[off:off + len(s)] - ref).abs().max().item(), "mean |ref|", ref.abs().mean().item()); off += len(s)
m.cpu(); m.init_rope("cpu"); m.attn_impl = "sdpa"
ids, pos, cu, seg, ms, n = pack(seqs)
with torch.no_grad():
    h = m.hidden(ids, pos, cu, ms)
print("cpu max abs diff", (h[:len(seqs[0])] - refs[0].cpu()).abs().max().item())
