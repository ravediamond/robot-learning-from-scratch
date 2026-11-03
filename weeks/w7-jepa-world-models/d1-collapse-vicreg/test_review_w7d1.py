import torch

from review_w7d1 import variance_term


def test_review_batch_dimension():
    v = torch.tensor([[3.0, -3.0, 0.0, 1.0]])          # one vector, its entries spread out
    collapsed = v.repeat(64, 1)                           # every sample identical: total collapse
    assert variance_term(collapsed).item() > 0.9, "a collapsed batch must be penalised"
    healthy = torch.randn(4096, 4, generator=torch.Generator().manual_seed(0)) * 2
    assert variance_term(healthy).item() < 0.01
