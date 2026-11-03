import math

import torch

import w3d4_gpt2 as m

TINY = m.GPT2Config(vocab_size=101, n_positions=16, n_embd=32, n_layer=3, n_head=4)


def test_param_count_124M():
    model = m.GPT2(m.GPT2Config())
    assert sum(p.numel() for p in model.parameters()) == 124_439_808


def test_names_match_gpt2():
    keys = set(m.GPT2(TINY).state_dict())
    for k in ["wte.weight", "wpe.weight", "h.0.ln_1.weight", "h.0.attn.c_attn.weight", "h.0.attn.c_attn.bias",
              "h.0.attn.c_proj.weight", "h.2.ln_2.bias", "h.1.mlp.c_fc.weight", "h.1.mlp.c_proj.bias",
              "ln_f.weight", "lm_head.weight"]:
        assert k in keys, f"missing {k}"
    assert m.GPT2(TINY).h[0].attn.c_attn.weight.shape == (96, 32)


def test_tying():
    model = m.GPT2(TINY)
    assert model.lm_head.weight is model.wte.weight


def test_forward_loss_and_init():
    torch.manual_seed(0)
    model = m.GPT2(TINY)
    idx = torch.randint(0, 101, (2, 16))
    tgt = torch.randint(0, 101, (2, 16))
    logits, loss = model(idx, tgt)
    assert logits.shape == (2, 16, 101) and loss.ndim == 0
    assert abs(loss.item() - math.log(101)) < 0.3, f"step-0 loss {loss.item():.2f}, expected ~ln(101)"
    assert model(idx)[1] is None
    t = tgt.clone()
    t[:, :8] = -1
    assert torch.isfinite(model(idx, t)[1])
    assert abs(model.wte.weight.std().item() - 0.02) < 0.003
    assert abs(model.h[0].attn.c_proj.weight.std().item() - 0.02 / math.sqrt(6)) < 0.003


def test_causal():
    torch.manual_seed(0)
    model = m.GPT2(TINY).eval()
    idx = torch.randint(0, 101, (1, 10))
    a = model(idx)[0]
    idx2 = idx.clone()
    idx2[0, 7:] = 0
    b = model(idx2)[0]
    assert torch.allclose(a[0, :7], b[0, :7], atol=1e-5) and not torch.allclose(a[0, 7:], b[0, 7:])
