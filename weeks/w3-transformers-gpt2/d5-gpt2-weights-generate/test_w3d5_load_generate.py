import math
import os

import pytest
import torch
from transformers import GPT2Config as HFConfig
from transformers import GPT2LMHeadModel

import w3d5_load_generate as m


def _pair(seed=0):
    torch.manual_seed(seed)
    hf = GPT2LMHeadModel(HFConfig(vocab_size=97, n_positions=24, n_embd=48, n_layer=2, n_head=4)).eval()
    ours = m.GPT2(m.GPT2Config(vocab_size=97, n_positions=24, n_embd=48, n_layer=2, n_head=4)).eval()
    return hf, ours


def test_load_matches_hf_logits():
    hf, ours = _pair()
    n = m.load_hf_state_dict(ours, hf.state_dict())
    assert n == len(ours.state_dict()) - 1, "every tensor once (lm_head is tied)"
    idx = torch.randint(0, 97, (2, 20))
    with torch.no_grad():
        assert torch.allclose(ours(idx)[0], hf(idx).logits, atol=1e-4)


def test_load_rejects_partial():
    hf, ours = _pair()
    sd = {k: v for k, v in hf.state_dict().items() if "h.1.mlp" not in k}
    with pytest.raises(ValueError):
        m.load_hf_state_dict(ours, sd)


def test_generate_greedy_matches_hf_and_crops():
    hf, ours = _pair(1)
    m.load_hf_state_dict(ours, hf.state_dict())
    idx = torch.randint(0, 97, (1, 5))
    out = m.generate(ours, idx, 8, temperature=0)
    ref = idx.clone()
    with torch.no_grad():
        for _ in range(8):
            ref = torch.cat([ref, hf(ref).logits[:, -1].argmax(-1, keepdim=True)], 1)
    assert torch.equal(out, ref)
    long = m.generate(ours, idx, 40, temperature=1.0, top_k=5, generator=torch.Generator().manual_seed(0))
    assert long.shape == (1, 45), "must crop the context beyond n_positions=24"
    again = m.generate(ours, idx, 40, temperature=1.0, top_k=5, generator=torch.Generator().manual_seed(0))
    assert torch.equal(long, again)


def test_char_tokenizer_and_batch():
    tok = m.CharTokenizer("hello world")
    assert tok.vocab_size == 8 and tok.decode(tok.encode("low hel")) == "low hel"
    data = torch.arange(100)
    x, y = m.get_batch(data, 10, 4, torch.Generator().manual_seed(0))
    assert x.shape == (4, 10) and torch.equal(y, x + 1)


def test_char_model_learns():
    text = "the baseline is the method. " * 60
    tok = m.CharTokenizer(text)
    data = torch.tensor(tok.encode(text))
    counts = torch.bincount(data).float()
    p = counts[counts > 0] / counts.sum()
    unigram = -(p * p.log()).sum().item()
    torch.manual_seed(0)
    model = m.GPT2(m.GPT2Config(vocab_size=tok.vocab_size, n_positions=32, n_embd=64, n_layer=2, n_head=4))
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    g = torch.Generator().manual_seed(0)
    for _ in range(150):
        x, y = m.get_batch(data, 32, 16, g)
        loss = model(x, y)[1]
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.5 * unigram, f"loss {loss.item():.2f} vs unigram {unigram:.2f}"


@pytest.mark.skipif(os.environ.get("GPT2_REAL") != "1", reason="set GPT2_REAL=1 (downloads 500 MB)")
def test_real_gpt2_small():
    from transformers import GPT2Tokenizer
    hf = GPT2LMHeadModel.from_pretrained("gpt2").eval()
    tok = GPT2Tokenizer.from_pretrained("gpt2")
    ours = m.GPT2(m.GPT2Config()).eval()
    m.load_hf_state_dict(ours, hf.state_dict())
    idx = torch.tensor([tok.encode("A robot arm picks up the bowl and")])
    with torch.no_grad():
        assert torch.allclose(ours(idx)[0], hf(idx).logits, atol=1e-3)
    out = m.generate(ours, idx, 30, temperature=0)
    print("\n" + tok.decode(out[0].tolist()))
