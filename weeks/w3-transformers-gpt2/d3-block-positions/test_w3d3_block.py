import torch
import torch.nn.functional as F

import w3d3_block as m


def test_block_shapes_and_identity():
    torch.manual_seed(0)
    b = m.Block(32, 4)
    x = torch.randn(2, 6, 32)
    assert b(x).shape == (2, 6, 32)
    with torch.no_grad():
        for lin in (b.attn.out_proj, b.mlp[2]):
            lin.weight.zero_()
            lin.bias.zero_()
    assert torch.allclose(b(x), x), "with zeroed output layers a pre-norm block must be the identity"


def test_block_causal_and_cross():
    torch.manual_seed(0)
    b = m.Block(32, 4)
    x = torch.randn(1, 6, 32)
    mask = torch.tril(torch.ones(6, 6, dtype=torch.bool))
    y1 = b(x, mask)
    x2 = x.clone()
    x2[0, 4:] += 5.0
    y2 = b(x2, mask)
    assert torch.allclose(y1[0, :4], y2[0, :4], atol=1e-5), "future tokens changed the past"
    assert b(x, context=torch.randn(1, 9, 32)).shape == (1, 6, 32)


def test_sinusoidal():
    pe = m.sinusoidal_positions(10, 8)
    assert pe.shape == (10, 8)
    assert torch.allclose(pe[0, 0::2], torch.zeros(4)) and torch.allclose(pe[0, 1::2], torch.ones(4))
    assert torch.isclose(pe[3, 2], torch.sin(torch.tensor(3 / 10000 ** (2 / 8))))


def test_rope_relative_and_norm():
    torch.manual_seed(0)
    q, k = torch.randn(1, 1, 8), torch.randn(1, 1, 8)
    rq = lambda t: m.apply_rope(q, torch.tensor([t]))
    rk = lambda t: m.apply_rope(k, torch.tensor([t]))
    d1 = (rq(5) * rk(2)).sum()
    d2 = (rq(13) * rk(10)).sum()
    assert torch.isclose(d1, d2, atol=1e-5), "q.k must depend only on the offset"
    x = torch.randn(2, 4, 7, 16)
    y = m.apply_rope(x, torch.arange(7))
    assert y.shape == x.shape and torch.allclose(y.norm(dim=-1), x.norm(dim=-1), atol=1e-5)
    assert torch.allclose(m.apply_rope(x, torch.zeros(7)), x)


def test_rope_matches_complex():
    torch.manual_seed(1)
    x = torch.randn(5, 6)
    pos = torch.arange(5).float()
    theta = 10000.0 ** (-torch.arange(0, 6, 2) / 6)
    z = torch.view_as_complex(x.view(5, 3, 2)) * torch.polar(torch.ones(5, 3), pos[:, None] * theta)
    assert torch.allclose(m.apply_rope(x, pos), torch.view_as_real(z).flatten(-2), atol=1e-5)


def test_patchify():
    img = torch.arange(2 * 3 * 4 * 6, dtype=torch.float32).view(2, 3, 4, 6)
    p = m.patchify(img, 2)
    assert p.shape == (2, 6, 12)
    assert torch.equal(p[0, 1], img[0, :, 0:2, 2:4].flatten()), "patch 1 is row 0, column 1"
    assert torch.equal(p[1, 3], img[1, :, 2:4, 0:2].flatten())
    assert torch.equal(p, F.unfold(img, 2, stride=2).transpose(1, 2))
