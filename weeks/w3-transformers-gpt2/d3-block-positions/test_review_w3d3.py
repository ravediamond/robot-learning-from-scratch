import torch

from review_w3d3 import Block


def test_review_identity_at_zero_output():
    torch.manual_seed(0)
    b = Block(16, 2)
    with torch.no_grad():
        for lin in (b.attn.out_proj, b.mlp[2]):
            lin.weight.zero_()
            lin.bias.zero_()
    x = torch.randn(2, 5, 16)
    assert torch.allclose(b(x), x), "the block destroys its input: where is the residual path?"
