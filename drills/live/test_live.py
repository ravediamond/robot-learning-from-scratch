import math

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from L01_causal_mha import CausalSelfAttention
from L02_norms import LayerNorm, RMSNorm
from L03_cross_entropy import cross_entropy
from L04_sampling import filter_logits, sample
from L05_rope import apply_rope, rope_cache
from L06_flow_matching import euler_sample, flow_matching_loss
from L07_train_step import lr_at, train
from L08_block import Block, block_params
from L09_iou_nms import box_iou, nms
from L10_quaternions import from_axis_angle, quat_conj, quat_mul, rotate


def test_L01_matches_torch_and_cache():
    torch.manual_seed(0)
    m = CausalSelfAttention(32, 4)
    ref = nn.MultiheadAttention(32, 4, batch_first=True)
    with torch.no_grad():
        ref.in_proj_weight.copy_(torch.cat([m.q.weight, m.k.weight, m.v.weight]))
        ref.in_proj_bias.copy_(torch.cat([m.q.bias, m.k.bias, m.v.bias]))
        ref.out_proj.weight.copy_(m.out.weight)
        ref.out_proj.bias.copy_(m.out.bias)
    x = torch.randn(2, 7, 32)
    mask = torch.triu(torch.ones(7, 7, dtype=torch.bool), 1)
    y, cache = m(x)
    assert torch.allclose(y, ref(x, x, x, attn_mask=mask)[0], atol=1e-5)
    assert cache["k"].shape == (2, 4, 7, 8)
    y1, c1 = m(x[:, :4])
    y2, c2 = m(x[:, 4:], cache=c1)
    assert torch.allclose(torch.cat([y1, y2], 1), y, atol=1e-5), "incremental decoding equals the full pass"
    assert c2["v"].shape == (2, 4, 7, 8)


def test_L02_norms():
    torch.manual_seed(0)
    x = torch.randn(3, 5, 16) * 4 + 2
    ln = LayerNorm(16)
    with torch.no_grad():
        ln.weight.uniform_()
        ln.bias.uniform_()
    assert torch.allclose(ln(x), F.layer_norm(x, (16,), ln.weight, ln.bias, 1e-5), atol=1e-5)
    rms = RMSNorm(16)
    y = rms(x)
    assert torch.allclose(y.pow(2).mean(-1), torch.ones(3, 5), atol=1e-4)
    assert torch.allclose(y, x / x.pow(2).mean(-1, keepdim=True).add(1e-6).sqrt(), atol=1e-6)


def test_L03_cross_entropy():
    torch.manual_seed(0)
    logits = torch.randn(10, 7) * 30
    t = torch.randint(0, 7, (10,))
    t[[2, 5]] = -100
    for eps in (0.0, 0.1):
        assert torch.allclose(cross_entropy(logits, t, label_smoothing=eps),
                              F.cross_entropy(logits, t, ignore_index=-100, label_smoothing=eps), atol=1e-5)
    big = torch.tensor([[1000.0, 0.0]])
    assert torch.isfinite(cross_entropy(big, torch.tensor([1])))
    assert cross_entropy(logits, torch.full((10,), -100)).item() == 0.0


def test_L04_filters_and_sampling():
    logits = torch.log(torch.tensor([[0.5, 0.2, 0.15, 0.1, 0.05]]))
    k2 = filter_logits(logits, top_k=2)
    assert torch.isinf(k2[0, 2:]).all() and torch.isfinite(k2[0, :2]).all()
    p = filter_logits(logits, top_p=0.8)
    assert torch.isfinite(p[0, :3]).all() and torch.isinf(p[0, 3:]).all(), "0.5 + 0.2 < 0.8 <= 0.85: keep three"
    assert torch.isfinite(filter_logits(logits, top_p=0.01)[0]).sum() == 1
    shuffled = logits[:, [3, 0, 4, 2, 1]]
    ps = filter_logits(shuffled, top_p=0.8)
    assert torch.isfinite(ps[0]).tolist() == [False, True, False, True, True], "positions must be restored"
    g = torch.Generator().manual_seed(0)
    assert sample(logits, 0.0, 0, 1.0, g).tolist() == [0]
    draws = torch.stack([sample(logits, 1.0, 2, 1.0, g) for _ in range(2000)])
    assert set(draws.flatten().tolist()) <= {0, 1}
    assert abs((draws == 0).float().mean().item() - 0.5 / 0.7) < 0.04


def test_L05_rope_relative():
    cos, sin = rope_cache(32, 16)
    assert cos.shape == (32, 8)
    torch.manual_seed(0)
    q, k = torch.randn(16), torch.randn(16)
    qs = apply_rope(q.expand(32, 16), cos, sin)
    ks = apply_rope(k.expand(32, 16), cos, sin)
    assert abs((qs[10] @ ks[7]).item() - (qs[5] @ ks[2]).item()) < 1e-4, "depends only on the offset"
    assert torch.allclose(qs[0], q)
    assert torch.allclose(qs.norm(dim=-1), q.norm().expand(32), atol=1e-5), "a rotation keeps the norm"


def test_L06_flow_matching():
    target = torch.tensor([[2.0, -1.0]])

    def exact(x, t, cond):                     # optimal velocity towards a single point
        return (target - x) / (1 - t).clamp(min=1e-3)

    g = torch.Generator().manual_seed(0)
    out = euler_sample(exact, torch.zeros(5, 1), 2, 10, g)
    assert torch.allclose(out, target.expand(5, 2), atol=1e-5)
    g1, g2 = torch.Generator().manual_seed(1), torch.Generator().manual_seed(1)
    x1 = torch.randn(64, 2)
    zero = lambda x, t, c: torch.zeros_like(x)
    loss = flow_matching_loss(zero, x1, torch.zeros(64, 1), g1)
    x0 = torch.randn(64, 2, generator=g2)
    assert torch.allclose(loss, (x1 - x0).pow(2).mean())


def test_L07_schedule_and_accumulation():
    assert lr_at(0, 1.0, 10, 100) == 0.1 and lr_at(9, 1.0, 10, 100) == 1.0
    assert abs(lr_at(55, 1.0, 10, 100) - 0.5) < 1e-12 and lr_at(100, 1.0, 10, 100, 0.1) == 0.1
    torch.manual_seed(0)
    a, b = nn.Linear(3, 1), nn.Linear(3, 1)
    b.load_state_dict(a.state_dict())
    xs, ys = torch.randn(8, 3), torch.randn(8, 1)
    micro = [(xs[i:i + 2], ys[i:i + 2]) for i in range(0, 8, 2)]
    lrs = train(a, micro, accum=4, base_lr=0.1, warmup=1, total_updates=1, max_norm=1e9)
    train(b, [(xs, ys)], accum=1, base_lr=0.1, warmup=1, total_updates=1, max_norm=1e9)
    assert lrs == [0.1]
    assert torch.allclose(a.weight, b.weight, atol=1e-6), "4 micro-batches of 2 == 1 batch of 8"
    c = nn.Linear(3, 1)
    w0 = c.weight.detach().clone()
    train(c, [(xs * 100, ys)], accum=1, base_lr=1.0, warmup=1, total_updates=1, max_norm=0.5)
    assert (c.weight - w0).norm() <= 0.5 + 1e-5, "clipped"


def test_L08_block():
    torch.manual_seed(0)
    blk = Block(64, 4)
    assert sum(p.numel() for p in blk.parameters()) == block_params(64) == 12 * 64 * 64 + 13 * 64
    x = torch.randn(2, 6, 64)
    y = blk(x)
    x2 = x.clone()
    x2[:, -1] += 10
    assert torch.allclose(blk(x2)[:, :-1], y[:, :-1], atol=1e-5), "causal: the past ignores the future"


def test_L09_iou_nms():
    a = torch.tensor([[0.0, 0, 2, 2], [1, 1, 3, 3]])
    b = torch.tensor([[0.0, 0, 2, 2], [2, 2, 4, 4], [10, 10, 11, 11]])
    iou = box_iou(a, b)
    assert iou.shape == (2, 3)
    assert torch.allclose(iou, torch.tensor([[1.0, 0.0, 0.0], [1 / 7, 1 / 7, 0.0]]), atol=1e-6)
    boxes = torch.tensor([[0.0, 0, 10, 10], [1, 1, 11, 11], [20, 20, 30, 30], [0, 0, 9, 10]])
    scores = torch.tensor([0.9, 0.8, 0.7, 0.95])
    assert nms(boxes, scores, 0.5) == [3, 2]
    assert nms(boxes, scores, 0.95) == [3, 0, 1, 2]


def test_L10_quaternions():
    q = from_axis_angle(torch.tensor([0.0, 0.0, 2.0]), torch.tensor(math.pi / 2))
    assert torch.allclose(rotate(q, torch.tensor([1.0, 0.0, 0.0])), torch.tensor([0.0, 1.0, 0.0]), atol=1e-6)
    torch.manual_seed(0)
    qs = F.normalize(torch.randn(5, 4), dim=-1)
    ps = F.normalize(torch.randn(5, 4), dim=-1)
    v = torch.randn(5, 3)
    assert torch.allclose(rotate(quat_mul(ps, qs), v), rotate(ps, rotate(qs, v)), atol=1e-5), "composition"
    assert torch.allclose(rotate(qs, v).norm(dim=-1), v.norm(dim=-1), atol=1e-5)
    assert torch.allclose(quat_mul(qs, quat_conj(qs)), torch.tensor([1.0, 0, 0, 0]).expand(5, 4), atol=1e-6)
    assert rotate(qs[:, None], torch.randn(5, 7, 3)).shape == (5, 7, 3)
