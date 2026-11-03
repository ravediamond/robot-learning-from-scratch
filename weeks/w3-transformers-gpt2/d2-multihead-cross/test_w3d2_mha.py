import torch
import torch.nn as nn

import w3d2_mha as m


def _copy_from_torch(ours, ref, D):
    W, b = ref.in_proj_weight.data, ref.in_proj_bias.data
    with torch.no_grad():
        for i, lin in enumerate([ours.q_proj, ours.k_proj, ours.v_proj]):
            lin.weight.copy_(W[i * D:(i + 1) * D])
            lin.bias.copy_(b[i * D:(i + 1) * D])
        ours.out_proj.weight.copy_(ref.out_proj.weight)
        ours.out_proj.bias.copy_(ref.out_proj.bias)


def test_split_merge():
    x = torch.arange(24.0).view(1, 2, 12)
    s = m.split_heads(x, 3)
    assert s.shape == (1, 3, 2, 4) and s[0, 0].tolist() == [[0, 1, 2, 3], [12, 13, 14, 15]]
    assert torch.equal(m.merge_heads(s), x)


def test_self_attention_matches_torch():
    torch.manual_seed(0)
    D, h = 32, 4
    ref = nn.MultiheadAttention(D, h, batch_first=True)
    ours = m.MultiHeadAttention(D, h)
    _copy_from_torch(ours, ref, D)
    x = torch.randn(2, 7, D)
    out_ref, _ = ref(x, x, x, need_weights=False)
    assert torch.allclose(ours(x), out_ref, atol=1e-5)
    causal = torch.tril(torch.ones(7, 7, dtype=torch.bool))
    out_ref_c, _ = ref(x, x, x, attn_mask=~causal, need_weights=False)   # torch: True = blocked
    assert torch.allclose(ours(x, mask=causal), out_ref_c, atol=1e-5)
    assert ours.last_attn.shape == (2, h, 7, 7) and not ours.last_attn.requires_grad


def test_cross_attention_matches_torch():
    torch.manual_seed(1)
    D, h = 32, 8
    ref = nn.MultiheadAttention(D, h, batch_first=True)
    ours = m.MultiHeadAttention(D, h)
    _copy_from_torch(ours, ref, D)
    x, ctx = torch.randn(3, 5, D), torch.randn(3, 11, D)
    pad = torch.zeros(3, 11, dtype=torch.bool)
    pad[0, 8:] = True                                                     # torch: True = ignore
    out_ref, _ = ref(x, ctx, ctx, key_padding_mask=pad, need_weights=False)
    ours_mask = (~pad)[:, None, None, :]                                  # (B, 1, 1, Tk) True = attend
    assert torch.allclose(ours(x, ctx, ours_mask), out_ref, atol=1e-5)
