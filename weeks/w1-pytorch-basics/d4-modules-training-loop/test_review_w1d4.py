import torch

from review_w1d4 import Policy, val_loss


def test_review_val_is_deterministic_and_safe():
    torch.manual_seed(0)
    p = Policy()
    p.train()
    x, y = torch.randn(64, 6), torch.randn(64, 6)
    a, b = val_loss(p, x, y), val_loss(p, x, y)
    assert a == b, "validation loss changes between identical calls"
    assert p.training, "val_loss must give the model back in the mode it received it"
