import inspect

import torch
import torch.nn.functional as F

import w3d1_attention as m


def test_causal_mask():
    assert m.causal_mask(3).tolist() == [[True, False, False], [True, True, False], [True, True, True]]


def test_padding_and_combine():
    pad = m.key_padding_mask(torch.tensor([2, 4]), 4)
    assert pad.shape == (2, 1, 4) and pad[0, 0].tolist() == [True, True, False, False]
    both = m.combine_masks(m.causal_mask(4), pad)
    assert both.shape == (2, 4, 4) and both[0, 3].tolist() == [True, True, False, False]
    assert m.combine_masks(None, None) is None


def test_matches_sdpa():
    torch.manual_seed(0)
    q, k, v = torch.randn(2, 3, 5, 8), torch.randn(2, 3, 7, 8), torch.randn(2, 3, 7, 4)
    assert torch.allclose(m.attention(q, k, v), F.scaled_dot_product_attention(q, k, v), atol=1e-5)
    q2 = torch.randn(2, 3, 6, 8)
    k2, v2 = torch.randn(2, 3, 6, 8), torch.randn(2, 3, 6, 4)
    cm = m.causal_mask(6)
    assert torch.allclose(m.attention(q2, k2, v2, cm), F.scaled_dot_product_attention(q2, k2, v2, attn_mask=cm), atol=1e-5)
    src = inspect.getsource(m)
    assert "scaled_dot_product_attention(" not in src.replace("F.scaled_dot_product_attention(q, k, v)", "")


def test_rows_sum_and_causality():
    torch.manual_seed(0)
    q, k = torch.randn(4, 6, 8), torch.randn(4, 6, 8)
    A = m.attention_weights(q, k, m.causal_mask(6))
    assert torch.allclose(A.sum(-1), torch.ones(4, 6), atol=1e-5)
    assert (A.triu(1) == 0).all()


def test_fully_masked_row_is_zero():
    q, k, v = torch.randn(1, 2, 4), torch.randn(1, 3, 4), torch.randn(1, 3, 4)
    mask = torch.tensor([[[False, False, False], [True, True, False]]])
    out = m.attention(q, k, v, mask)
    assert torch.isfinite(out).all() and (out[0, 0] == 0).all()
