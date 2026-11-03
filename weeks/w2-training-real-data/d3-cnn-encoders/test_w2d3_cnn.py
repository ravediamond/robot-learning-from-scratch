import torch
import torch.nn as nn
import torch.nn.functional as F

import w2d3_cnn as m


def test_conv_unfold():
    torch.manual_seed(0)
    x, w, b = torch.randn(2, 3, 11, 9), torch.randn(5, 3, 3, 3), torch.randn(5)
    for s, p in [(1, 0), (2, 1), (1, 1)]:
        assert torch.allclose(m.conv2d_unfold(x, w, b, s, p), F.conv2d(x, w, b, s, p), atol=1e-5)
    assert torch.allclose(m.conv2d_unfold(x, w, None), F.conv2d(x, w), atol=1e-5)


def test_receptive_field():
    assert m.receptive_field([(3, 1)] * 3) == 7
    assert m.receptive_field([(3, 2)] * 3) == 15
    assert m.receptive_field([(5, 2), (3, 1), (3, 1)]) == 13


def test_spatial_softmax():
    feat = torch.full((1, 2, 5, 7), -1e4)
    feat[0, 0, 0, 6] = 0.0          # channel 0 peaks at top-right
    feat[0, 1, 4, 0] = 0.0          # channel 1 peaks at bottom-left
    out = m.SpatialSoftmax()(feat)
    assert out.shape == (1, 4)
    assert torch.allclose(out, torch.tensor([[1.0, -1.0, -1.0, 1.0]]), atol=1e-4)
    flat = m.SpatialSoftmax()(torch.zeros(3, 4, 6, 6))
    assert torch.allclose(flat, torch.zeros(3, 8), atol=1e-6)


def test_encoder():
    enc = m.CNNEncoder(out_dim=16)
    assert enc(torch.rand(2, 3, 64, 64)).shape == (2, 16)
    assert enc(torch.rand(1, 3, 96, 128)).shape == (1, 16), "any image size"
    assert not any(isinstance(x, nn.modules.batchnorm._BatchNorm) for x in enc.modules())
    assert any(isinstance(x, nn.GroupNorm) for x in enc.modules())


def test_random_shift():
    img = torch.zeros(64, 1, 9, 9)
    img[:, :, 4, 4] = 1.0
    out = m.random_shift(img, pad=2, generator=torch.Generator().manual_seed(0))
    assert out.shape == img.shape
    pos = out.flatten(1).argmax(1)
    r, c = pos // 9, pos % 9
    assert ((r - 4).abs() <= 2).all() and ((c - 4).abs() <= 2).all()
    assert len(set(zip(r.tolist(), c.tolist()))) > 10, "shifts must differ across samples"
    assert torch.equal(m.random_shift(img, 0), img)
    edge = torch.arange(25.0).reshape(1, 1, 5, 5).repeat(32, 1, 1, 1)
    sh = m.random_shift(edge, 2, torch.Generator().manual_seed(1))
    assert set(sh.unique().tolist()) <= set(range(25)), "pad by replicating the border, not zeros"
