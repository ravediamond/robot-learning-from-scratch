import math

import torch
import torch.nn.functional as F

import m1_attention
import m2_flow
import m3_cem
import m4_bootstrap


def test_m1_attention():
    torch.manual_seed(0)
    q, k, v = (torch.randn(2, 3, 5, 4) for _ in range(3))
    assert torch.allclose(m1_attention.causal_attention(q, k, v), F.scaled_dot_product_attention(q, k, v, is_causal=True), atol=1e-5)
    D, h = 8, 2
    x, Wqkv, Wo = torch.randn(2, 5, D), torch.randn(3 * D, D), torch.randn(D, D)
    ref = torch.nn.MultiheadAttention(D, h, bias=False, batch_first=True)
    with torch.no_grad():
        ref.in_proj_weight.copy_(Wqkv)
        ref.out_proj.weight.copy_(Wo)
    mask = torch.triu(torch.ones(5, 5, dtype=torch.bool), 1)
    assert torch.allclose(m1_attention.multi_head_self_attention(x, Wqkv, Wo, h), ref(x, x, x, attn_mask=mask)[0], atol=1e-4)


def test_m2_flow():
    class Exact(torch.nn.Module):                        # data = point mass at c: v = (c - x) / (1 - t)
        def forward(self, x, t, cond):
            return (cond[:, None, :] - x) / (1 - t[:, None, None]).clamp(min=1e-6)
    cond = torch.tensor([[1.0, -2.0], [0.5, 0.5]])
    out = m2_flow.sample(Exact(), cond, (2, 3, 2), 4, torch.Generator().manual_seed(0))
    assert torch.allclose(out, cond[:, None, :].expand(2, 3, 2), atol=1e-4)
    net = lambda x, t, c: torch.zeros_like(x)
    loss = m2_flow.flow_matching_loss(net, torch.zeros(4096, 1, 1), None, torch.Generator().manual_seed(0))
    assert abs(loss.item() - 1.0) < 0.06, "zero net, x1 = 0: target -x0 has unit variance"


def test_m3_cem():
    goal = torch.tensor([0.4, -0.3])
    dyn = lambda z, a: z + a
    cost = lambda z: ((z - goal) ** 2).sum(-1)
    seq = m3_cem.cem(dyn, cost, torch.zeros(2), H=4, A=2, N=300, K=30, iters=6, a_max=0.2,
                     generator=torch.Generator().manual_seed(0))
    assert seq.shape == (4, 2) and (seq.abs() <= 0.2 + 1e-6).all()
    assert (seq.sum(0) - goal).norm() < 0.1


def test_m4_bootstrap():
    a = torch.tensor([1, 1, 1, 0, 1, 1, 0, 1, 1, 1.0])
    b = torch.tensor([1, 0, 1, 0, 0, 1, 0, 1, 0, 1.0])
    d, lo, hi = m4_bootstrap.paired_success_diff_ci(a, b, 2000, 0)
    assert abs(d - 0.3) < 1e-9 and lo <= d <= hi
    lo, hi = m4_bootstrap.wilson(7, 10)
    assert abs(lo - 0.3968) < 1e-3 and abs(hi - 0.8922) < 1e-3
