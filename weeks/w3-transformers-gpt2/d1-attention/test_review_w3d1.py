import torch
import torch.nn.functional as F

from review_w3d1 import causal_self_attention


def test_review_causal():
    torch.manual_seed(0)
    x = torch.randn(2, 5, 8)
    out = causal_self_attention(x)
    assert torch.isfinite(out).all(), "nan in the output"
    ref = F.scaled_dot_product_attention(x, x, x, is_causal=True)
    assert torch.allclose(out, ref, atol=1e-5)
