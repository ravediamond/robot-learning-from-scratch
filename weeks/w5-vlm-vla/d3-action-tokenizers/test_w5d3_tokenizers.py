import math

import torch

import w5d3_tokenizers as m


def _smooth_chunks(n=64, H=50, D=6, seed=0):
    g = torch.Generator().manual_seed(seed)
    t = torch.linspace(0, 1, H)[None, :, None]
    a, f, p = torch.rand(n, 1, D, generator=g), 0.25 + 0.5 * torch.rand(n, 1, D, generator=g), 6 * torch.rand(n, 1, D, generator=g)
    return 0.8 * a * torch.sin(2 * math.pi * f * t + p)              # (n, H, D) in [-0.8, 0.8]


def test_bins():
    a = torch.tensor([-1.0, -0.999, 0.0, 0.999, 1.0, 5.0])
    tok = m.bin_encode(a, -1, 1, 4)
    assert tok.tolist() == [0, 0, 2, 3, 3, 3] and tok.dtype == torch.long
    assert torch.allclose(m.bin_decode(torch.tensor([0, 3]), -1, 1, 4), torch.tensor([-0.75, 0.75]))
    x = torch.rand(10000) * 2 - 1
    err = (m.bin_decode(m.bin_encode(x, -1, 1, 256), -1, 1, 256) - x).abs()
    assert err.max() <= 1 / 256 + 1e-6


def test_dct_orthonormal_and_roundtrip():
    C = m.dct_matrix(50)
    assert torch.allclose(C @ C.T, torch.eye(50), atol=1e-5)
    x = torch.randn(50, 3)
    assert torch.allclose(m.dct_decode(m.dct_encode(x, 50, 1e6), 50, 1e6), x, atol=1e-5)


def test_dct_beats_bins_per_token():
    chunks = _smooth_chunks()
    bin_err = torch.stack([(m.bin_decode(m.bin_encode(c, -1, 1, 256), -1, 1, 256) - c).abs().max() for c in chunks]).max()
    dct_err = torch.stack([(m.dct_decode(m.dct_encode(c, 10, 50.0), 50, 50.0) - c).abs().max() for c in chunks]).max()
    print(f"\nbins: 300 tokens, max err {bin_err:.4f}   DCT: 60 tokens, max err {dct_err:.4f}")
    assert dct_err < 0.06, "10 coefficients per dim should capture these smooth chunks"
