import inspect
import math

import torch
import torch.nn as nn

import w2d2_stability as m


def test_initial_losses():
    y = torch.tensor([[1.0, -1.0], [3.0, 1.0]])
    assert abs(m.initial_mse_if_zero(y) - 3.0) < 1e-6
    assert abs(m.initial_ce(10) - math.log(10)) < 1e-9


def test_warmup_cosine():
    f = lambda s: m.warmup_cosine_lr(s, total=1000, warmup=100, peak=1e-3, final=1e-5)
    assert f(0) == 0.0
    assert abs(f(50) - 5e-4) < 1e-12
    assert abs(f(100) - 1e-3) < 1e-12
    assert abs(f(550) - (1e-5 + (1e-3 - 1e-5) * 0.5)) < 1e-9
    assert abs(f(1000) - 1e-5) < 1e-12 and abs(f(5000) - 1e-5) < 1e-12
    assert all(f(s) >= f(s + 1) for s in range(100, 1000))


def test_clip_matches_torch():
    torch.manual_seed(0)
    net1 = nn.Sequential(nn.Linear(5, 7), nn.ReLU(), nn.Linear(7, 3))
    net2 = nn.Sequential(nn.Linear(5, 7), nn.ReLU(), nn.Linear(7, 3))
    net2.load_state_dict(net1.state_dict())
    x = torch.randn(4, 5) * 100
    for n in (net1, net2):
        n(x).pow(2).sum().backward()
    net1[0].bias.grad = None                      # a param without grad must be skipped
    net2[0].bias.grad = None
    ours = m.clip_grad_norm(list(net1.parameters()), 1.0)
    ref = torch.nn.utils.clip_grad_norm_([p for p in net2.parameters() if p.grad is not None], 1.0)
    assert isinstance(ours, float) and abs(ours - ref.item()) / ref.item() < 1e-5
    for p1, p2 in zip(net1.parameters(), net2.parameters()):
        if p1.grad is not None:
            assert torch.allclose(p1.grad, p2.grad, atol=1e-6)
    assert "torch.nn.utils" not in inspect.getsource(m.clip_grad_norm)


def test_clip_noop_when_small():
    p = torch.zeros(3, requires_grad=True)
    p.grad = torch.tensor([0.1, 0.0, 0.0])
    m.clip_grad_norm([p], 1.0)
    assert torch.equal(p.grad, torch.tensor([0.1, 0.0, 0.0]))


class _Log(nn.Module):
    def forward(self, x):
        return torch.log(x)


def test_first_nonfinite():
    net = nn.Sequential(nn.Linear(3, 3), nn.ReLU(), nn.Sequential(nn.Linear(3, 3), _Log()), nn.Linear(3, 1))
    with torch.no_grad():
        net[2][0].weight.fill_(-1.0)
        net[2][0].bias.fill_(-1.0)                # always negative -> log gives nan
    name = m.first_nonfinite_module(net, torch.randn(4, 3))
    assert name == "2.1", f"got {name}"
    assert all(len(mod._forward_hooks) == 0 for mod in net.modules()), "hooks not removed"
    assert m.first_nonfinite_module(nn.Linear(3, 3), torch.randn(2, 3)) is None
