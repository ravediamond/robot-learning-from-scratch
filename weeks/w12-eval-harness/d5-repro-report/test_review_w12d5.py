import torch

from review_w12d5 import weights_hash


def test_review_large_tensors_are_fully_hashed():
    torch.manual_seed(0)
    a = {"w": torch.randn(2_000)}
    b = {"w": a["w"].clone()}
    b["w"][1_000] += 0.5
    assert weights_hash(a) != weights_hash(b)
    assert weights_hash(a) == weights_hash({"w": a["w"].clone()})
