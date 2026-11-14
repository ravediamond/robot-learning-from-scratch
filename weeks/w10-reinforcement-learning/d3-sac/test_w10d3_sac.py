import math

import torch
from torch import nn

import rl_envs
from w10d3_sac import ReplayBuffer, actor_loss, alpha_loss, q_target, soft_update, squashed_sample, train_sac


def test_squashed_sample_logp_and_bounds():
    g = torch.Generator().manual_seed(0)
    mean, log_std = torch.zeros(2000, 1), torch.zeros(2000, 1)
    a, logp = squashed_sample(mean, log_std, g)
    assert a.abs().max() < 1 and logp.shape == (2000,)
    u = torch.atanh(a.clamp(-0.999999, 0.999999))
    expected = (-0.5 * u.pow(2) - 0.5 * math.log(2 * math.pi) - torch.log(1 - a.pow(2) + 1e-6)).sum(-1)
    assert torch.allclose(logp, expected, atol=1e-3)
    # a proper density on (-1, 1): E[1 / p(a)] over samples = length of the support = 2
    assert abs((1 / logp.exp()).mean().item() - 2.0) < 0.15


def test_squashed_sample_is_reparameterised():
    mean = torch.zeros(4, 2, requires_grad=True)
    log_std = torch.zeros(4, 2, requires_grad=True)
    a, logp = squashed_sample(mean, log_std, torch.Generator().manual_seed(1))
    (a.sum() + logp.sum()).backward()
    assert mean.grad.abs().sum() > 0 and log_std.grad.abs().sum() > 0


def test_soft_update():
    t, s = nn.Linear(2, 2), nn.Linear(2, 2)
    w_t, w_s = t.weight.clone(), s.weight.clone()
    soft_update(t, s, 0.1)
    assert torch.allclose(t.weight, 0.9 * w_t + 0.1 * w_s)
    assert torch.equal(s.weight, w_s)


def test_targets_and_losses():
    y = q_target(torch.tensor([1.0, 1.0]), torch.tensor([False, True]), torch.tensor([10.0, 10.0]),
                 torch.tensor([8.0, 12.0]), torch.tensor([-2.0, -2.0]), alpha=0.5, gamma=0.9)
    assert torch.allclose(y, torch.tensor([1 + 0.9 * (8 + 1.0), 1.0]))
    assert not y.requires_grad
    al = actor_loss(torch.tensor([3.0]), torch.tensor([2.0]), torch.tensor([-1.0]), alpha=0.5)
    assert torch.isclose(al, torch.tensor(-2.5))
    log_alpha = torch.zeros(1, requires_grad=True)
    alpha_loss(log_alpha, torch.tensor([-5.0, -5.0]), target_entropy=-2.0).backward()
    assert log_alpha.grad.item() > 0, "entropy (5) above target (2): alpha must go DOWN (positive gradient)"


def test_replay_wraps():
    b = ReplayBuffer(5, 1, 1)
    for i in range(4):
        x = torch.full((2, 1), float(i))
        b.add(x, x, torch.full((2,), float(i)), x, torch.zeros(2, dtype=torch.bool))
    assert b.size == 5 and b.ptr == 3
    assert sorted(b.rew.tolist()) == [1.0, 2.0, 2.0, 3.0, 3.0]
    o, a, r, o2, d = b.sample(64, torch.Generator().manual_seed(0))
    assert o.shape == (64, 1) and r.min() >= 1.0


def test_sac_learns():
    actor, _ = train_sac(steps=1500, seed=0)
    ret = rl_envs.evaluate(lambda o: torch.tanh(actor(o)[0]))["return"].mean().item()
    assert ret > -9.0, f"deterministic return {ret:.1f}; random is about -22"
