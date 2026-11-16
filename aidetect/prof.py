"""Profile one training micro-batch (fwd+bwd) and print GPU time by kernel category. python prof.py [budget]"""
import importlib.util, os, sys, torch
sys.path.insert(0, os.path.dirname(__file__))
spec = importlib.util.spec_from_file_location("tr", os.path.join(os.path.dirname(__file__), "03_train.py")); tr = importlib.util.module_from_spec(spec); spec.loader.exec_module(tr)
from model import Cfg, Detector
budget = int(sys.argv[1]) if len(sys.argv) > 1 else 65536
mode = sys.argv[2] if len(sys.argv) > 2 else "default"
dev = torch.device("cuda"); cfg = Cfg()
m = Detector(cfg).load_nano().to(dev); m.init_rope(dev); m.attn_impl = os.environ.get('ATTN', 'fa3'); m.embed.weight.requires_grad_(False)
for l in m.layers:
    l.pre_attn = torch.compile(l.pre_attn, mode=None if mode == "default" else mode); l.post_attn = torch.compile(l.post_attn, mode=None if mode == "default" else mode)
it = iter(tr.Stream("train", budget, 1024, 0.25, {}, 0))
batches = [next(it) for _ in range(6)]
n_mm = sum(p.numel() for l in m.layers for p in l.parameters() if p.dim() == 2)


def step(b):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        lt, ld, *_ = tr.losses(m, b, dev)
    (lt + ld).backward()


for b in batches[:3]: step(b)
torch.cuda.synchronize()
import time
t = time.time()
for b in batches[3:6]: step(b)
torch.cuda.synchronize(); dt = (time.time() - t) / 3
fl = sum(tr.flops_of(b, cfg, n_mm) for b in batches[3:6]) / 3
print(f"budget {budget} mode {mode}: {dt*1000:.1f} ms/microbatch, {budget/dt/1e3:.0f}k tok/s, MFU {fl/dt/tr.PEAK*100:.1f}% (fwd+bwd only)")
from torch.profiler import profile, ProfilerActivity
with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as prof:
    step(batches[3]); torch.cuda.synchronize()
ev = [e for e in prof.key_averages() if e.device_time_total > 0]
tot = sum(e.self_device_time_total for e in ev)
cat = {}
for e in ev:
    n = e.key.lower()
    c = "gemm" if ("gemm" in n or "cutlass" in n or "nvjet" in n or "sm90" in n or "cublas" in n) else "flash" if "flash" in n or "fmha" in n else "triton_fused" if "triton" in n else "other"
    cat[c] = cat.get(c, 0) + e.self_device_time_total
print({k: f"{v/tot*100:.1f}%" for k, v in cat.items()}, f"total {tot/1e3:.1f} ms")
for e in sorted(ev, key=lambda e: -e.self_device_time_total)[:14]:
    print(f"{e.self_device_time_total/1e3:8.2f} ms  {e.key[:110]}")
