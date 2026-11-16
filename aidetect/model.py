"""voyage-4-nano backbone (Qwen3 layers) with a per-token head and a document head, packed/varlen forward.

GPU training/inference: flat packed tokens + cu_seqlens through flash-attn varlen (no padding).
CPU inference: the same weights with SDPA, one sequence at a time (no padding either).
Attention is bidirectional by default (a classifier sees the whole text); causal=True reproduces HF Qwen3.
"""
from __future__ import annotations
import torch, torch.nn as nn, torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class Cfg:
    dim: int = 1024
    n_layers: int = 12
    n_heads: int = 16
    n_kv: int = 8
    hd: int = 128
    ffn: int = 3072
    vocab: int = 151936
    theta: float = 1e6
    eps: float = 1e-6
    causal: bool = False
    max_pos: int = 8192


def rms(x, w, eps):
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)).to(x.dtype) * w.to(x.dtype)


def rope_tables(cfg: Cfg, device=None):
    inv = 1.0 / (cfg.theta ** (torch.arange(0, cfg.hd, 2, dtype=torch.float32, device=device) / cfg.hd))
    f = torch.outer(torch.arange(cfg.max_pos, dtype=torch.float32, device=device), inv)
    e = torch.cat([f, f], -1)
    return e.cos(), e.sin()


def apply_rope(x, cos, sin):  # x (T, H, hd); cos/sin (T, hd)
    h = x.shape[-1] // 2
    rot = torch.cat([-x[..., h:], x[..., :h]], -1)
    return x * cos[:, None, :].to(x.dtype) + rot * sin[:, None, :].to(x.dtype)


class Layer(nn.Module):
    def __init__(self, c: Cfg):
        super().__init__()
        self.c = c
        self.ln1 = nn.Parameter(torch.ones(c.dim)); self.ln2 = nn.Parameter(torch.ones(c.dim))
        self.qn = nn.Parameter(torch.ones(c.hd)); self.kn = nn.Parameter(torch.ones(c.hd))
        self.qkv = nn.Linear(c.dim, (c.n_heads + 2 * c.n_kv) * c.hd, bias=False)
        self.o = nn.Linear(c.n_heads * c.hd, c.dim, bias=False)
        self.gu = nn.Linear(c.dim, 2 * c.ffn, bias=False)
        self.down = nn.Linear(c.ffn, c.dim, bias=False)

    def pre_attn(self, x, cos, sin):
        c = self.c
        qkv = self.qkv(rms(x, self.ln1, c.eps))
        q, k, v = qkv.split([c.n_heads * c.hd, c.n_kv * c.hd, c.n_kv * c.hd], -1)
        T = x.shape[0]
        q = rms(q.view(T, c.n_heads, c.hd), self.qn, c.eps); k = rms(k.view(T, c.n_kv, c.hd), self.kn, c.eps)
        return apply_rope(q, cos, sin), apply_rope(k, cos, sin), v.view(T, c.n_kv, c.hd)

    def post_attn(self, x, a):
        x = x + self.o(a.reshape(x.shape[0], -1))
        g, u = self.gu(rms(x, self.ln2, self.c.eps)).chunk(2, -1)
        return x + self.down(F.silu(g) * u)


class Detector(nn.Module):
    def __init__(self, c: Cfg = Cfg(), n_layers: int | None = None):
        super().__init__()
        if n_layers: c.n_layers = n_layers
        self.c = c
        self.embed = nn.Embedding(c.vocab, c.dim)
        self.layers = nn.ModuleList(Layer(c) for _ in range(c.n_layers))
        self.norm = nn.Parameter(torch.ones(c.dim))
        self.tok_head = nn.Linear(c.dim, 1)
        self.doc_head = nn.Linear(c.dim, 1)
        self.register_buffer("cos", torch.zeros(1), persistent=False); self.register_buffer("sin", torch.zeros(1), persistent=False)
        self.attn_impl = "flash"
        self.attn_store = None

    def init_rope(self, device):
        cos, sin = rope_tables(self.c, device)
        self.cos, self.sin = cos, sin

    def attend(self, q, k, v, cu, max_s):
        if self.attn_impl == "fa3":  # FlashAttention-3 (Hopper): faster forward and backward than FA2
            import flash_attn_interface as fa3
            o = fa3.flash_attn_varlen_func(q, k, v, cu, cu, max_s, max_s, causal=self.c.causal)
            return o[0] if isinstance(o, tuple) else o
        if self.attn_impl == "flash":
            from flash_attn import flash_attn_varlen_func
            return flash_attn_varlen_func(q, k, v, cu, cu, max_s, max_s, causal=self.c.causal)
        outs = []  # sdpa, one sequence at a time
        cl = cu.tolist()
        rep = self.c.n_heads // self.c.n_kv
        for a, b in zip(cl[:-1], cl[1:]):
            qs, ks, vs = (t[a:b].transpose(0, 1) for t in (q, k, v))
            ks, vs = ks.repeat_interleave(rep, 0), vs.repeat_interleave(rep, 0)
            if self.attn_store is not None:  # explanation mode: keep the head-averaged attention matrix of this layer (T x T)
                w = torch.softmax(qs @ ks.transpose(-1, -2) / qs.shape[-1] ** 0.5, -1)
                self.attn_store.append(w.mean(0).detach())
            outs.append(F.scaled_dot_product_attention(qs[None], ks[None], vs[None], is_causal=self.c.causal)[0].transpose(0, 1))
        return torch.cat(outs, 0)

    def hidden(self, ids, pos, cu, max_s, embeds=None):
        x = self.embed(ids) if embeds is None else embeds  # embeds: (T, dim) input embeddings, used by integrated gradients
        if torch.is_autocast_enabled():  # bf16 residual stream under autocast (also keeps the compiled fused residual adds in one dtype)
            x = x.to(torch.get_autocast_gpu_dtype())
        cos, sin = self.cos[pos], self.sin[pos]
        for l in self.layers:
            q, k, v = l.pre_attn(x, cos, sin)
            x = l.post_attn(x, self.attend(q, k, v, cu, max_s))
        return rms(x, self.norm, self.c.eps)

    def heads(self, h, seg, nseq):
        """token logits (T,), doc logits (nseq,) from the mean of each sequence's hidden states."""
        tok = self.tok_head(h).squeeze(-1).float()
        s = torch.zeros(nseq, h.shape[-1], device=h.device, dtype=torch.float32).index_add_(0, seg, h.float())
        n = torch.zeros(nseq, device=h.device, dtype=torch.float32).index_add_(0, seg, torch.ones_like(seg, dtype=torch.float32))
        doc = self.doc_head((s / n.clamp(min=1)[:, None]).to(h.dtype)).squeeze(-1).float()
        return tok, doc

    def forward(self, ids, pos, cu, seg, max_s, nseq, embeds=None):
        return self.heads(self.hidden(ids, pos, cu, max_s, embeds), seg, nseq)

    @torch.no_grad()
    def load_nano(self, path="/data/new/prod_models/voyage-4-nano/model.safetensors"):
        from safetensors.torch import load_file
        sd = load_file(path)
        g = lambda k: sd[k].float()
        self.embed.weight.copy_(g("model.embed_tokens.weight")); self.norm.copy_(g("model.norm.weight"))
        for i, l in enumerate(self.layers):
            p = f"model.layers.{i}."
            l.ln1.copy_(g(p + "input_layernorm.weight")); l.ln2.copy_(g(p + "post_attention_layernorm.weight"))
            l.qn.copy_(g(p + "self_attn.q_norm.weight")); l.kn.copy_(g(p + "self_attn.k_norm.weight"))
            l.qkv.weight.copy_(torch.cat([g(p + f"self_attn.{n}_proj.weight") for n in "qkv"], 0))
            l.o.weight.copy_(g(p + "self_attn.o_proj.weight"))
            l.gu.weight.copy_(torch.cat([g(p + "mlp.gate_proj.weight"), g(p + "mlp.up_proj.weight")], 0))
            l.down.weight.copy_(g(p + "mlp.down_proj.weight"))
        nn.init.normal_(self.tok_head.weight, std=0.02); nn.init.zeros_(self.tok_head.bias)
        nn.init.normal_(self.doc_head.weight, std=0.02); nn.init.zeros_(self.doc_head.bias)
        return self


def pack(seqs, device=None, pad_to=None, pad_id=0):
    """list of 1-D token lists/arrays -> ids, pos, cu, seg, max_s (optionally padded to pad_to tokens with one dummy sequence)."""
    lens = [len(s) for s in seqs]
    tot = sum(lens)
    if pad_to and tot < pad_to:
        seqs = list(seqs) + [[pad_id] * (pad_to - tot)]; lens.append(pad_to - tot)
    ids = torch.cat([torch.as_tensor(s, dtype=torch.long) for s in seqs])
    L = torch.tensor(lens)
    cu = torch.zeros(len(lens) + 1, dtype=torch.int32); cu[1:] = L.cumsum(0)
    seg = torch.repeat_interleave(torch.arange(len(lens)), L)
    pos = torch.arange(len(ids)) - torch.repeat_interleave(cu[:-1].long(), L)
    out = [t.to(device) if device else t for t in (ids, pos, cu, seg)]
    return (*out, int(L.max()), len(lens))
