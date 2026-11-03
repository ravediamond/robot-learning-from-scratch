import pytest
import torch

import sim2d
import w4d5_flow_policy as m


def test_rollout_with_expert_and_straight_line():
    side = {}

    def expert(obs, g):                       # the expert, one step re-applied H times (fine for a test)
        a = sim2d.expert_action(obs, torch.ones(len(obs)))
        return a[:, None, :].expand(-1, 4, 2)

    r = m.rollout(expert, 50, 1, seed=0)
    assert r["success"].all() and not r["collision"].any()
    straight = lambda obs, g: sim2d.clip_action(sim2d.GOAL - obs)[:, None, :].expand(-1, 8, 2)
    r = m.rollout(straight, 50, 4, seed=0)
    assert r["collision"].all() and not r["success"].any()
    assert (r["steps"] > 0).all() and (r["steps"] < sim2d.T_MAX).all()


def test_masked_mse():
    p, t = torch.zeros(1, 3, 2), torch.ones(1, 3, 2)
    t[0, 2] = 100.0
    assert torch.isclose(m.masked_mse(p, t, torch.tensor([[True, True, False]])), torch.tensor(1.0))


@pytest.fixture(scope="module")
def policies():
    d = sim2d.make_demos(200, 8, seed=0)
    return m.train_mse_policy(d, 8), m.train_flow_policy(d, 8)


def test_shapes(policies):
    mse, flow = policies
    obs = torch.zeros(3, 2)
    assert mse.act(obs).shape == (3, 8, 2) and flow.act(obs, torch.Generator().manual_seed(0)).shape == (3, 8, 2)


def test_flow_beats_mse_closed_loop(policies):
    mse, flow = policies
    r_m, r_f = m.rollout(mse.act, 200, 4, seed=1), m.rollout(flow.act, 200, 4, seed=1)
    cm, cf = r_m["collision"].float().mean().item(), r_f["collision"].float().mean().item()
    print(f"\ncollision  MSE {cm:.2f} {sim2d.wilson(int(cm * 200), 200)}   flow {cf:.2f} {sim2d.wilson(int(cf * 200), 200)}")
    assert cm > 0.30 and cf < 0.15 and r_f["success"].float().mean() > 0.85
