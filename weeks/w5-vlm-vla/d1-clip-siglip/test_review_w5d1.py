import torch

from review_w5d1 import contrastive_loss


def test_review_scale_invariant():
    torch.manual_seed(0)
    a, b = torch.randn(8, 16), torch.randn(8, 16)
    l1 = contrastive_loss(a, b)
    assert torch.isclose(l1, contrastive_loss(50 * a, 0.01 * b), atol=1e-5), \
        "the loss must not depend on the norm of the embeddings: only on their directions"
