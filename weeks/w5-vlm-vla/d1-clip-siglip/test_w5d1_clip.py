import math

import torch
import torch.nn.functional as F

import shapes_data
import w5d1_clip as m


def test_clip_loss_formula():
    torch.manual_seed(0)
    a, b = torch.randn(6, 4), torch.randn(6, 4)
    z, u = F.normalize(a, dim=-1), F.normalize(b, dim=-1)
    logits = 5.0 * z @ u.T
    ref = (F.cross_entropy(logits, torch.arange(6)) + F.cross_entropy(logits.T, torch.arange(6))) / 2
    assert torch.isclose(m.clip_loss(a, b, torch.tensor(math.log(5.0))), ref, atol=1e-5)
    assert torch.isclose(m.clip_loss(3 * a, 0.1 * b, torch.tensor(math.log(5.0))), ref, atol=1e-5)


def test_siglip_formula():
    torch.manual_seed(0)
    a, b = torch.randn(5, 4), torch.randn(5, 4)
    z, u = F.normalize(a, dim=-1), F.normalize(b, dim=-1)
    logits = 2.0 * z @ u.T - 1.0
    labels = 2 * torch.eye(5) - 1
    ref = -F.logsigmoid(labels * logits).mean()
    assert torch.isclose(m.siglip_loss(a, b, torch.tensor(math.log(2.0)), torch.tensor(-1.0)), ref, atol=1e-5)


def test_zero_shot():
    tr_i, tr_c = shapes_data.make_shapes(4000, torch.Generator().manual_seed(0))
    va_i, va_c = shapes_data.make_shapes(500, torch.Generator().manual_seed(1))
    torch.manual_seed(0)
    model = m.train_clip(tr_i, tr_c)
    acc = m.zero_shot_accuracy(model, va_i, va_c, shapes_data.all_captions())
    print(f"\nzero-shot accuracy {acc:.3f} (chance {1 / 24:.3f})")
    assert acc > 0.9
