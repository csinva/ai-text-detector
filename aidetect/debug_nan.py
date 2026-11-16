"""Replay the micro-batches of training step N (deterministic stream) through a checkpoint and look for non-finite activations."""
import importlib.util, os, sys, torch
sys.path.insert(0, os.path.dirname(__file__))
spec = importlib.util.spec_from_file_location("tr", os.path.join(os.path.dirname(__file__), "03_train.py")); tr = importlib.util.module_from_spec(spec); spec.loader.exec_module(tr)
from model import Cfg, Detector
step, ck, tok = int(sys.argv[1]), sys.argv[2], sys.argv[3]
accum, budget = 4, 65536
dev = torch.device("cuda")
m = Detector(Cfg()); m.load_state_dict(torch.load(ck, map_location="cpu")); m.to(dev); m.init_rope(dev); m.attn_impl = "fa3"
loader = torch.utils.data.DataLoader(tr.Stream("train", budget, 1024, 0.25, {"spans": 4.0, "raid_splice": 1.0}, 0, tok), batch_size=None, num_workers=8, prefetch_factor=6)
it = iter(loader)
first = (step - 1) * accum
for k in range(first + accum):
    b = next(it)
    if k < first - 2: continue
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        h = m.hidden(b["ids"].to(dev), b["pos"].to(dev), b["cu"].to(dev), b["max_s"])
    lens = (b["cu"][1:] - b["cu"][:-1]).tolist()
    print(f"batch {k} (step {k // accum + 1}): finite={bool(torch.isfinite(h).all())} max|h|={h.float().abs().max().item():.1f} nseq={b['nseq']} max_len={max(lens)} min_len={min(lens)} nreal={b['nreal']}", flush=True)
    if not torch.isfinite(h).all():
        bad = (~torch.isfinite(h).all(-1)).nonzero().flatten()
        seg = b["seg"].to(dev)[bad]; print("bad tokens", len(bad), "in sequences", seg.unique().tolist()[:10])
        s0 = int(seg[0]); a, e = int(b["cu"][s0]), int(b["cu"][s0 + 1]); print("seq", s0, "len", e - a, "first bad pos", int(b["pos"][bad[0]]), "ids head", b["ids"][a:a + 12].tolist())
        torch.save({k2: v for k2, v in b.items()}, os.path.join(os.path.dirname(ck), f"bad_batch_{k}.pt"))
