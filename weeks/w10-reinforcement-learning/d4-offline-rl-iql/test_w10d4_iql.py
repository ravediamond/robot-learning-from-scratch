import torch

import offline_data
import rl_envs
from w10d4_iql import awr_weights, expectile_loss, train_bc, train_iql


def test_expectile_loss():
    d = torch.tensor([1.0, -1.0, 2.0])
    assert torch.isclose(expectile_loss(d, 0.8), torch.tensor((0.8 * 1 + 0.2 * 1 + 0.8 * 4) / 3))
    assert torch.isclose(expectile_loss(d, 0.5), 0.5 * d.pow(2).mean())


def test_expectile_leans_to_the_top():
    x = torch.tensor([0.0, 0.0, 0.0, 10.0])
    grid = torch.linspace(0, 10, 1001)
    best = lambda tau: grid[torch.stack([expectile_loss(x - c, tau) for c in grid]).argmin()].item()
    assert abs(best(0.5) - 2.5) < 0.02
    assert best(0.9) > 6.0


def test_awr_weights():
    adv = torch.tensor([0.0, 1.0, 100.0], requires_grad=True)
    w = awr_weights(adv, beta=2.0, w_max=50.0)
    assert torch.allclose(w, torch.tensor([1.0, 7.389056, 50.0]), atol=1e-4)
    assert not w.requires_grad


def test_iql_beats_bc_on_mixed_data():
    data = offline_data.make_dataset(seed=0)
    bc = rl_envs.evaluate(train_bc(data, seed=0))
    iql = rl_envs.evaluate(train_iql(data, seed=0))
    still = rl_envs.evaluate(lambda o: torch.zeros(len(o), 2))["return"].mean().item()
    assert bc["return"].mean().item() < still, "BC on this mix walks away from the goal"
    assert iql["return"].mean().item() > -9.0, f"IQL return {iql['return'].mean():.1f}"
    assert iql["success"].float().mean() > bc["success"].float().mean() + 0.5
