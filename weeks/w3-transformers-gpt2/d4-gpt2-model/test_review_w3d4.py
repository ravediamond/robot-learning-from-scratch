import torch

from review_w3d4 import TiedLM


def test_review_really_tied():
    m = TiedLM(50, 8)
    assert sum(p.numel() for p in m.parameters()) == 50 * 8 + 8 * 8 + 8
    opt = torch.optim.SGD(m.parameters(), lr=1.0)
    m(torch.tensor([[1, 2, 3]])).sum().backward()
    opt.step()
    assert torch.equal(m.lm_head.weight, m.wte.weight), "the two matrices drifted apart after one step"
