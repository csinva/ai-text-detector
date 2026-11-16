"""FA3 varlen forward/backward agrees with FA2 (bf16, GQA, bidirectional, packed)."""
import torch
import flash_attn_interface as fa3
from flash_attn import flash_attn_varlen_func as fa2

torch.manual_seed(0)
q = torch.randn(1024, 16, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
k = torch.randn(1024, 8, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
v = torch.randn(1024, 8, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
cu = torch.tensor([0, 200, 700, 1024], dtype=torch.int32, device="cuda")
g = torch.randn(1024, 16, 128, device="cuda", dtype=torch.bfloat16)
o3 = fa3.flash_attn_varlen_func(q, k, v, cu, cu, 500, 500, causal=False)
o3 = o3[0] if isinstance(o3, tuple) else o3
gr3 = torch.autograd.grad(o3, (q, k, v), g)
o2 = fa2(q, k, v, cu, cu, 500, 500, causal=False)
gr2 = torch.autograd.grad(o2, (q, k, v), g)
print("fwd diff", (o3 - o2).abs().max().item(), [(a - b).abs().max().item() for a, b in zip(gr3, gr2)], "scale", o2.abs().mean().item())
