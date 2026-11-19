#!/usr/bin/env python3
"""Fine-tune voyage-4-nano into an AI-text detector (token head + document head) on one H100.

Packed variable-length sequences (flash-attn varlen, no padding), fixed token budget per micro-batch so torch.compile
sees static shapes, bf16 autocast over fp32 master weights, fused AdamW. Logs tokens/s and MFU every step.
Loss = BCE(token logit, soft token label) + BCE(document logit, mean label of the crop); optional heat map terms (--minority, --w_bnd, --w_sep) act on mixed documents only.
"""
import argparse, json, math, os, sys, time
import numpy as np, torch, torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from model import Cfg, Detector

D = os.path.join(os.environ["V45_DATA_ROOT"], "data/aidetect")
PEAK = 989.4e12  # H100 SXM dense bf16


class Stream(torch.utils.data.IterableDataset):
    def __init__(self, name, budget, max_len, p_short, weights, seed, tok="tok"):
        self.tok = tok
        self.name, self.budget, self.max_len, self.p_short, self.seed = name, budget, max_len, p_short, seed
        self.w = weights

    def __iter__(self):
        wi = torch.utils.data.get_worker_info()
        rng = np.random.default_rng(self.seed * 1000 + (wi.id if wi else 0))
        ids = np.load(f"{D}/{self.tok}/{self.name}_ids.npy", mmap_mode="r"); lab = np.load(f"{D}/{self.tok}/{self.name}_lab.npy", mmap_mode="r")
        off = np.load(f"{D}/{self.tok}/{self.name}_off.npy"); src = np.load(f"{D}/{self.tok}/{self.name}_src.npy")
        srcs = json.load(open(f"{D}/{self.tok}/srcs.json"))
        p = np.array([self.w.get(srcs[s], 1.0) for s in src], dtype=np.float64); p /= p.sum()
        n = len(src)
        while True:
            xs, ys, lens, tot = [], [], [], 0
            while tot < self.budget:
                d = rng.choice(n, p=p)
                a, b = off[d], off[d + 1]
                L = b - a
                if L < 8:
                    continue
                c = L if L < 16 else min(L, int(math.exp(rng.uniform(math.log(16), math.log(self.max_len)))) if rng.random() < self.p_short else min(L, self.max_len))
                c = min(c, self.budget - tot)  # the last document is cropped to fill the budget exactly, so there is never a dummy tail
                s = a + (rng.integers(0, L - c + 1) if L > c else 0)
                xs.append(np.asarray(ids[s:s + c], dtype=np.int64)); ys.append(np.asarray(lab[s:s + c], dtype=np.float32) / 255.0)
                lens.append(c); tot += c
            if tot < self.budget:  # dummy tail sequence, masked out of the loss
                xs.append(np.zeros(self.budget - tot, dtype=np.int64)); ys.append(np.zeros(self.budget - tot, dtype=np.float32)); lens.append(self.budget - tot)
            nreal = len(lens) - (1 if tot < self.budget else 0)
            L = torch.tensor(lens)
            cu = torch.zeros(len(lens) + 1, dtype=torch.int32); cu[1:] = L.cumsum(0)
            seg = torch.repeat_interleave(torch.arange(len(lens)), L)
            ids_t = torch.from_numpy(np.concatenate(xs)); y = torch.from_numpy(np.concatenate(ys))
            pos = (torch.arange(self.budget) - torch.repeat_interleave(cu[:-1].long(), L)).clamp(max=8191)
            valid = (seg < nreal).float()
            ydoc = torch.zeros(len(lens)).index_add_(0, seg, y) / L.clamp(min=1)
            yield dict(ids=ids_t, pos=pos, cu=cu, seg=seg, y=y, ydoc=ydoc, valid=valid, nseq=len(lens), nreal=nreal,
                       max_s=int(L.max()), sq=float((L[:nreal].double() ** 2).sum()), ntok=int(L[:nreal].sum()))


def flops_of(b, cfg, n_mm):
    # fwd+bwd = 6 N per token for the matmuls + 12 * d_attn * sum(s^2) per layer for (bidirectional) attention
    return 6 * n_mm * b["ntok"] + 12 * cfg.n_layers * cfg.n_heads * cfg.hd * b["sq"] * (0.5 if cfg.causal else 1.0)


def heat_terms(tok, y, v, seg, nseq, h):
    """Heat map specific loss terms. In a mixed document (3% to 97% AI tokens) the plain token average lets the model push every token towards the document label, so:
      minority weights: each class carries half of the document's token weight (weight 0.5 / class share, between 1 and min_cap),
      boundary weights: tokens within bnd_k tokens of an AI/human switch count (1 + w_bnd) times,
      separation margin: relu(sep_margin - (mean P(AI) on AI tokens - mean P(AI) on human tokens)) per mixed document, weight w_sep."""
    yh = (y > 0.5).float() * v; n = torch.zeros(nseq, device=y.device).index_add_(0, seg, v); na = torch.zeros(nseq, device=y.device).index_add_(0, seg, yh)
    fa = na / n.clamp(min=1); mixed = ((fa > 0.03) & (fa < 0.97)).float(); w = torch.ones_like(y)
    if h["minority"]:
        fs = fa[seg]; wm = torch.where(y > 0.5, 0.5 / fs.clamp(min=0.03), 0.5 / (1 - fs).clamp(min=0.03)).clamp(1.0, h["min_cap"]); w = torch.where(mixed[seg] > 0, wm, w)
    if h["w_bnd"] > 0:
        same = seg[1:] == seg[:-1]; flip = ((y[1:] > 0.5) != (y[:-1] > 0.5)) & same; bd = torch.zeros_like(y); bd[1:] += flip.float(); bd[:-1] += flip.float()
        bd = F.max_pool1d((bd > 0).float()[None, None], kernel_size=2 * h["bnd_k"] + 1, stride=1, padding=h["bnd_k"])[0, 0]; w = w * (1 + h["w_bnd"] * bd * mixed[seg])
    ls = tok.new_zeros(())
    if h["w_sep"] > 0:
        p = torch.sigmoid(tok.float()); hum = (1 - (y > 0.5).float()) * v
        ma = torch.zeros(nseq, device=y.device).index_add_(0, seg, p * yh) / na.clamp(min=1); nh = torch.zeros(nseq, device=y.device).index_add_(0, seg, hum)
        mh = torch.zeros(nseq, device=y.device).index_add_(0, seg, p * hum) / nh.clamp(min=1)
        ok = mixed * (na >= 4).float() * (nh >= 4).float(); ls = (F.relu(h["sep_margin"] - (ma - mh)) * ok).sum() / ok.sum().clamp(min=1)
    return w, ls


def losses(model, b, dev, heat=None):
    tok, doc = model(b["ids"].to(dev, non_blocking=True), b["pos"].to(dev, non_blocking=True), b["cu"].to(dev), b["seg"].to(dev, non_blocking=True), b["max_s"], b["nseq"])
    y, v = b["y"].to(dev, non_blocking=True), b["valid"].to(dev, non_blocking=True)
    bce = F.binary_cross_entropy_with_logits(tok, y, reduction="none"); ls = tok.new_zeros(())
    if heat:
        w, ls = heat_terms(tok, y, v, b["seg"].to(dev, non_blocking=True), b["nseq"], heat); bce = bce * w
    lt = (bce * v).sum() / v.sum().clamp(min=1)
    yd = b["ydoc"].to(dev)[: b["nreal"]]
    ld = F.binary_cross_entropy_with_logits(doc[: b["nreal"]], yd)
    return lt, ld, tok, doc, y, v, ls


@torch.no_grad()
def dev_eval(model, dev, budget, max_len, n_batches=12, tok="tok"):
    from sklearn.metrics import roc_auc_score
    model.eval()
    it = iter(torch.utils.data.DataLoader(Stream("dev", budget, max_len, 0.0, {}, 123, tok), batch_size=None, num_workers=0))
    lt = ld = 0; yt, pt, yd_, pd_ = [], [], [], []
    for _ in range(n_batches):
        b = next(it)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            a, c, tok, doc, y, v, _ = losses(model, b, dev)
        lt += a.item(); ld += c.item()
        m = v.bool().cpu(); yt.append((y.cpu()[m] > 0.5).numpy()); pt.append(tok.float().cpu()[m].numpy())
        yd = b["ydoc"][: b["nreal"]].numpy(); k = (yd < 0.02) | (yd > 0.98)
        yd_.append((yd[k] > 0.5)); pd_.append(doc.float().cpu()[: b["nreal"]].numpy()[k])
    model.train()
    yt, pt, yd_, pd_ = map(np.concatenate, (yt, pt, yd_, pd_))
    return dict(dev_tok_loss=lt / n_batches, dev_doc_loss=ld / n_batches, dev_tok_auc=float(roc_auc_score(yt, pt)),
                dev_doc_auc=float(roc_auc_score(yd_, pd_)), dev_doc_acc=float(((pd_ > 0) == yd_).mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--budget", type=int, default=65536, help="tokens per micro-batch")
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--total_tokens", type=float, default=300e6)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--head_lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--warmup", type=float, default=0.03)
    ap.add_argument("--max_len", type=int, default=1024)
    ap.add_argument("--p_short", type=float, default=0.25)
    ap.add_argument("--causal", type=int, default=0)
    ap.add_argument("--n_layers", type=int, default=12, help="layers used from the backbone (12 for voyage-4-nano, 28 for the full Qwen3-0.6B models)")
    ap.add_argument("--backbone", default="/data/new/prod_models/voyage-4-nano/model.safetensors", help="safetensors file or directory of the starting weights")
    ap.add_argument("--train_emb", type=int, default=0)
    ap.add_argument("--compile", type=int, default=1)
    ap.add_argument("--w_spans", type=float, default=4.0)
    ap.add_argument("--w_splice", type=float, default=1.0)
    ap.add_argument("--minority", type=int, default=0, help="heat map loss: class-balanced token weights inside mixed documents")
    ap.add_argument("--min_cap", type=float, default=8.0); ap.add_argument("--w_bnd", type=float, default=0.0, help="extra weight of tokens near an AI/human switch")
    ap.add_argument("--bnd_k", type=int, default=6); ap.add_argument("--w_sep", type=float, default=0.0, help="weight of the AI/human probability margin loss in mixed documents"); ap.add_argument("--sep_margin", type=float, default=0.85)
    ap.add_argument("--w_mix", type=float, default=1.0, help="sampling weight of the explicit mixed documents (mix_*) and of the layout-augmented ones (fmt_*)")
    ap.add_argument("--w_fr", type=float, default=1.0, help="sampling weight of the frontier-model documents (fr_ai, fr_human: Claude, DeepSeek-style and gpt-oss-120b texts and their matching human passages)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--eval_every", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--synthetic", type=int, default=0, help="reuse one batch (GPU-only throughput)")
    ap.add_argument("--tok", default="tok", help="tokenized corpus dir under data/aidetect (tok or tok_full)")
    ap.add_argument("--save_every", type=int, default=0)
    ap.add_argument("--init_weights", default="", help="weights-only checkpoint to continue from (fresh optimizer, 20-step re-warmup)")
    ap.add_argument("--start_step", type=int, default=0)
    ap.add_argument("--attn", default="fa3", choices=["fa3", "flash"])
    ap.add_argument("--compile_mode", default="default")
    ap.add_argument("--max_steps", type=int, default=0, help="stop early (benchmarks)")
    a = ap.parse_args()
    out = f"{D}/runs/{a.run}"; os.makedirs(out, exist_ok=True)
    json.dump(vars(a), open(f"{out}/args.json", "w"))
    torch.manual_seed(a.seed)
    dev = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    cfg = Cfg(causal=bool(a.causal))
    model = Detector(cfg, a.n_layers or None)
    model.load_backbone(a.backbone)
    model.to(dev); model.init_rope(dev); model.train(); model.attn_impl = a.attn
    model.embed.weight.requires_grad_(bool(a.train_emb))
    n_mm = sum(p.numel() for l in model.layers for p in l.parameters() if p.dim() == 2)
    print(f"matmul params {n_mm/1e6:.1f}M, layers {cfg.n_layers}", flush=True)
    if a.compile:
        for l in model.layers:
            l.pre_attn = torch.compile(l.pre_attn, mode=None if a.compile_mode == 'default' else a.compile_mode); l.post_attn = torch.compile(l.post_attn, mode=None if a.compile_mode == 'default' else a.compile_mode)
    heat = dict(minority=a.minority, min_cap=a.min_cap, w_bnd=a.w_bnd, bnd_k=a.bnd_k, w_sep=a.w_sep, sep_margin=a.sep_margin) if (a.minority or a.w_bnd or a.w_sep) else None
    heads = [p for n, p in model.named_parameters() if "head" in n]
    dec = [p for n, p in model.named_parameters() if p.requires_grad and p.dim() == 2 and "head" not in n]
    nod = [p for n, p in model.named_parameters() if p.requires_grad and p.dim() < 2 and "head" not in n]
    opt = torch.optim.AdamW([dict(params=dec, lr=a.lr, weight_decay=a.wd), dict(params=nod, lr=a.lr, weight_decay=0.0),
                             dict(params=heads, lr=a.head_lr, weight_decay=0.0)], betas=(0.9, 0.98), eps=1e-8, fused=True)
    base = [g["lr"] for g in opt.param_groups]
    tok_per_step = a.budget * a.accum
    start = 0; rewarm_from = -1
    if os.path.exists(f"{out}/ckpt.pt"):  # full resume after a preemption
        ck = torch.load(f"{out}/ckpt.pt", map_location="cpu"); model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); start = ck["step"]; del ck
        print(f"resumed from {out}/ckpt.pt at step {start}", flush=True)
    elif a.init_weights:
        model.load_state_dict(torch.load(a.init_weights, map_location="cpu")); start = a.start_step; rewarm_from = start
        print(f"continuing from weights {a.init_weights} at step {start} (fresh optimizer)", flush=True)
    steps = int(a.total_tokens // tok_per_step)
    if a.max_steps: steps = min(steps, a.max_steps)
    loader = torch.utils.data.DataLoader(Stream("train", a.budget, a.max_len, a.p_short, {"spans": a.w_spans, "raid_splice": a.w_splice, **{k: a.w_mix for k in ("mix_sent", "mix_block", "mix_para", "mix_word", "mix_cross", "fmt_human", "fmt_ai")}, "fr_ai": a.w_fr, "fr_human": a.w_fr}, a.seed + 7919 * start, a.tok),
                                         batch_size=None, num_workers=a.workers, prefetch_factor=6, pin_memory=True, persistent_workers=True)
    it = iter(loader)
    log = open(f"{out}/train_log.jsonl", "a")
    t_prev = time.time(); warm = max(1, int(a.warmup * steps)); fl_acc = 0.0; tk_acc = 0; dt_acc = 0.0
    n_skipped = 0
    for step in range(start + 1, steps + 1):
        f = step / warm if step <= warm else 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, steps - warm)))
        if rewarm_from >= 0: f *= min(1.0, (step - rewarm_from) / 20)
        for g, b0 in zip(opt.param_groups, base): g["lr"] = b0 * f
        torch.cuda.synchronize(); t0 = time.time(); fl = 0.0; tk = 0; ltt = ldd = lss = 0.0; mbs = []
        for _ in range(a.accum):
            b = b_syn if (a.synthetic and step > 1) else next(it)
            b_syn = b; mbs.append(b)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lt, ld, *rest = losses(model, b, dev, heat); ls = rest[-1]
            ((lt + ld + a.w_sep * ls) / a.accum).backward()
            fl += flops_of(b, cfg, n_mm); tk += b["ntok"]; ltt += lt.detach(); ldd += ld.detach(); lss += ls.detach()
        gn = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        if not torch.isfinite(gn):  # a non-finite step must not touch the weights: drop it and keep the batches for inspection
            n_skipped += 1; opt.zero_grad(set_to_none=True)
            torch.save(mbs, f"{out}/bad_step{step}.pt"); print(f"step {step}: non-finite grad norm, skipped ({n_skipped} so far)", flush=True)
            log.write(json.dumps(dict(step=step, skipped=True)) + "\n"); continue
        opt.step(); opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize(); dt = time.time() - t0
        rec = dict(step=step, tokens=step * tok_per_step, loss_tok=float(ltt) / a.accum, loss_doc=float(ldd) / a.accum, loss_sep=float(lss) / a.accum, gnorm=float(gn),
                   lr=opt.param_groups[0]["lr"], step_s=dt, tok_s=tk / dt, mfu=fl / dt / PEAK)
        if step % a.eval_every == 0 or step == steps:
            rec.update(dev_eval(model, dev, a.budget, a.max_len, tok=a.tok))
        if a.save_every and step % a.save_every == 0:
            torch.save(model.state_dict(), f"{out}/model.pt")
            torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), step=step), f"{out}/ckpt.tmp"); os.replace(f"{out}/ckpt.tmp", f"{out}/ckpt.pt")
        log.write(json.dumps(rec) + "\n"); log.flush()
        if step % 20 == 0 or "dev_doc_auc" in rec or step <= 3:
            print({k: round(v, 4) if isinstance(v, float) else v for k, v in rec.items()}, flush=True)
    torch.save({k: v for k, v in model.state_dict().items()}, f"{out}/model.pt")
    print("saved", out, flush=True)


if __name__ == "__main__":
    main()
