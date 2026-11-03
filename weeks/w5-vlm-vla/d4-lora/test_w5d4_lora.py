import torch
import torch.nn as nn

import w5d4_lora as m


def _net():
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(8, 32), nn.ReLU(), nn.Sequential(nn.Linear(32, 32), nn.ReLU()), nn.Linear(32, 4))


def test_init_identity_and_freeze():
    base = nn.Linear(8, 5)
    lo = m.LoRALinear(base, r=2, alpha=4)
    x = torch.randn(3, 8)
    assert torch.allclose(lo(x), base(x))
    assert not base.weight.requires_grad and not base.bias.requires_grad
    assert lo.A.shape == (2, 8) and lo.B.shape == (5, 2) and lo.A.requires_grad and lo.scale == 2.0
    assert m.trainable_params(lo) == 2 * 8 + 5 * 2


def test_merge():
    lo = m.LoRALinear(nn.Linear(8, 5), r=3)
    with torch.no_grad():
        lo.B.normal_()
    x = torch.randn(4, 8)
    assert torch.allclose(lo.merged()(x), lo(x), atol=1e-5)


def test_apply_and_train():
    net = _net()
    x, y = torch.randn(256, 8), torch.randn(256, 4)
    before = nn.functional.mse_loss(net(x), y).item()
    frozen = {k: v.clone() for k, v in net.state_dict().items()}
    m.apply_lora(net, ["0", "2.0"], r=4)
    assert isinstance(net[2][0], m.LoRALinear) and isinstance(net[0], m.LoRALinear)
    assert m.trainable_params(net) == 4 * (8 + 32) + 4 * (32 + 32)
    opt = torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr=1e-2)
    for _ in range(300):
        loss = nn.functional.mse_loss(net(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.85 * before, "LoRA did not learn"
    assert torch.equal(net[3].weight, frozen["3.weight"]), "a frozen layer moved"
    assert torch.equal(net[0].base.weight, frozen["0.weight"])
