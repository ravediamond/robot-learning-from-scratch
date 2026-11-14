import torch

import offline_data
import rl_envs
from w10d5_recap import episode_returns, fit_value, improvement_indicator, n_step_advantage, train_conditioned


def test_n_step_advantage_formula():
    T, n_ep = 4, 1
    data = {"rew": torch.tensor([1.0, 2.0, 3.0, 4.0]), "t": torch.arange(4), "obs": torch.zeros(4, 4)}
    V = lambda obs, t: 10.0 * (t.float() + 1)
    adv = n_step_advantage(data, V, n=2, gamma=0.5)
    expected = torch.tensor([1 + 0.5 * 2 + 0.25 * 30 - 10, 2 + 0.5 * 3 + 0.25 * 40 - 20,
                             3 + 0.5 * 4 + 0.0 - 30, 4 + 0.0 - 40])
    assert torch.allclose(adv, expected)


def test_indicator_is_per_task():
    adv = torch.cat([torch.linspace(-1, 1, 100), torch.linspace(-60, -40, 100)])
    task = torch.cat([torch.zeros(100), torch.ones(100)]).long()
    ind = improvement_indicator(adv, task, q=0.7)
    assert ind.dtype == torch.bool
    assert abs(ind[:100].float().mean().item() - 0.3) < 0.02
    assert abs(ind[100:].float().mean().item() - 0.3) < 0.02


def test_recap_pipeline_beats_bc():
    data = offline_data.make_dataset(seed=0, per_step=True)
    V = fit_value(data, seed=0)
    G = episode_returns(data, 0.95)
    v = V(data["obs"], data["t"])
    assert ((v - G).pow(2).mean() / G.var()).item() < 0.4, "the value function should explain part of the returns"
    adv = n_step_advantage(data, V, n=1, gamma=0.95)
    ind = improvement_indicator(adv, torch.zeros(len(adv), dtype=torch.long), q=0.7)
    assert ind[data["kind"] == 0].float().mean() > 0.6, "most expert steps are better than usual"
    assert ind[data["kind"] == 2].float().mean() < 0.15
    pol = train_conditioned(data, ind, seed=0)
    good = rl_envs.evaluate(lambda o: pol(o, 1))["return"].mean().item()
    bad = rl_envs.evaluate(lambda o: pol(o, 0))["return"].mean().item()
    bc = train_conditioned(data, torch.ones_like(ind), seed=0)
    plain = rl_envs.evaluate(lambda o: bc(o, 1))["return"].mean().item()
    assert good > -8.0, f"I=1 return {good:.1f}"
    assert good > plain + 8 and good > bad + 8
