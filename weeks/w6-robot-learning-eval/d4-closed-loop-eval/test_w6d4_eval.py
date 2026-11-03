import pytest
import torch

import policies_ref as P
import sim2d
import w6d4_eval as m


def test_fixed_starts_and_no_mutation():
    a, b = m.fixed_starts(10, 3), m.fixed_starts(10, 3)
    assert torch.equal(a, b) and not torch.equal(a, m.fixed_starts(10, 4))
    keep = a.clone()
    straight = lambda obs, g: sim2d.clip_action(sim2d.GOAL - obs)[:, None, :].expand(-1, 4, 2)
    out = m.evaluate(straight, a, 4)
    assert torch.equal(a, keep), "evaluate must not modify the starts"
    assert out["collision"].all() and not out["success"].any()


def test_evaluate_expert_and_determinism():
    starts = m.fixed_starts(30, 0)
    up = lambda obs, g: sim2d.expert_action(obs, torch.ones(len(obs)))[:, None, :].expand(-1, 1, 2)
    out = m.evaluate(up, starts, 1)
    assert out["success"].all() and (out["steps"] > 10).all()
    noisy = lambda obs, g: up(obs, g) + 0.02 * torch.randn(len(obs), 1, 2, generator=g)
    o1, o2 = m.evaluate(noisy, starts, 1, seed=5), m.evaluate(noisy, starts, 1, seed=5)
    assert torch.equal(o1["steps"], o2["steps"]), "same seed must give the same outcomes"


def test_paired_counts():
    a = torch.tensor([1, 1, 0, 0, 1], dtype=torch.bool)
    b = torch.tensor([1, 0, 1, 0, 0], dtype=torch.bool)
    assert m.paired_counts(a, b) == {"n11": 1, "n10": 2, "n01": 1, "n00": 1}


@pytest.fixture(scope="module")
def trained():
    demos = sim2d.make_demos(240, 8, seed=0)
    val_ep = demos["episode"] >= 200
    train = {k: v[~val_ep] for k, v in demos.items()}
    val = {k: v[val_ep] for k, v in demos.items()}
    return P.train_mse_policy(train, 8), P.train_flow_policy(train, 8), val


def test_val_loss_ranks_wrong(trained):
    mse, flow, val = trained
    v_m, v_f = m.val_chunk_mse(mse.act, val, seed=3), m.val_chunk_mse(flow.act, val, seed=3)
    starts = m.fixed_starts(200, 1)
    s_m, s_f = m.evaluate(mse.act, starts, 4, seed=2)["success"], m.evaluate(flow.act, starts, 4, seed=2)["success"]
    pc = m.paired_counts(s_m, s_f)
    print(f"\nval MSE  mse-head {v_m:.2e}  flow {v_f:.2e}   success  mse-head {s_m.float().mean():.2f}  flow {s_f.float().mean():.2f}  {pc}")
    assert v_m < v_f, "the MSE head wins on val MSE (it predicts the mean)"
    assert s_f.float().mean() > s_m.float().mean() + 0.2, "the flow head wins closed loop"
    assert pc["n01"] > 5 * max(pc["n10"], 1)
