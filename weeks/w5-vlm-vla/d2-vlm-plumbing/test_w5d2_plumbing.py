import torch

import w5d2_plumbing as m


def test_pixel_shuffle():
    B, h, C, r = 2, 4, 3, 2
    x = torch.randn(B, h * h, C)
    out = m.pixel_shuffle_tokens(x, r)
    assert out.shape == (B, 4, C * 4)
    grid = x.view(B, h, h, C)
    for i in range(2):
        for j in range(2):
            block = grid[:, 2 * i:2 * i + 2, 2 * j:2 * j + 2, :]          # (B, r, r, C)
            assert torch.equal(out[:, i * 2 + j], block.reshape(B, -1)), f"token ({i},{j}) wrong"


def test_projector():
    p = m.Projector(12, 8)
    assert p(torch.randn(2, 5, 12)).shape == (2, 5, 8)


def test_block_mask():
    msk = m.block_attention_mask([2, 1, 2])
    exp = torch.tensor([[1, 1, 0, 0, 0], [1, 1, 0, 0, 0], [1, 1, 1, 0, 0], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]], dtype=torch.bool)
    assert torch.equal(msk, exp)
    c = m.block_attention_mask([2, 1, 2], causal_last=True)
    assert c[3].tolist() == [True, True, True, True, False] and c[4].tolist() == [True] * 5


def test_assemble():
    B, D = 2, 4
    cams = [torch.ones(B, 3, D), 2 * torch.ones(B, 3, D)]
    txt, tmask = 3 * torch.ones(B, 2, D), torch.tensor([[True, False], [True, True]])
    seq, km = m.assemble_prefix(cams, txt, tmask, 4 * torch.ones(B, 1, D))
    assert seq.shape == (B, 9, D) and seq[:, :, 0].tolist()[0] == [1, 1, 1, 2, 2, 2, 3, 3, 4]
    assert km.tolist() == [[True] * 7 + [False, True], [True] * 9]
