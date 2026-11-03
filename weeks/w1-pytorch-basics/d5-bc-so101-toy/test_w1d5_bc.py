import pytest
import torch

import w1d5_bc as m
from so101_toy import generate_episodes


@pytest.fixture(scope="module")
def data():
    eps = generate_episodes(60, seed=0)
    tr, va = m.split_by_episode(len(eps), 0.2, seed=0)
    return eps, tr, va


def test_split(data):
    eps, tr, va = data
    assert len(va) == 12 and len(tr) == 48
    assert set(tr).isdisjoint(va) and set(tr) | set(va) == set(range(60))
    assert m.split_by_episode(60, 0.2, seed=0) == (tr, va), "must be deterministic for a seed"
    assert sorted(va) != list(range(12)), "shuffle before splitting"


def test_stack_frames(data):
    eps, tr, _ = data
    obs, act = m.stack_frames(eps, tr)
    assert obs.shape[1] == 12 and act.shape[1] == 6
    assert len(obs) == sum(len(eps[i]["obs"]) for i in tr)


def test_norm_stats(data):
    eps, tr, va = data
    obs, act = m.stack_frames(eps, tr)
    s = m.compute_norm_stats(obs, act)
    assert set(s) == {"obs_mean", "obs_std", "act_mean", "act_std"}
    assert s["obs_mean"].shape == (12,) and s["act_std"].shape == (6,)
    assert torch.allclose(s["obs_mean"], obs.mean(0))
    const = torch.ones(10, 12)
    assert (m.compute_norm_stats(const, torch.ones(10, 6))["obs_std"] >= 1e-2).all()


def test_policy_buffers_and_io():
    stats = {"obs_mean": torch.zeros(12), "obs_std": torch.ones(12),
             "act_mean": torch.full((6,), 50.0), "act_std": torch.full((6,), 10.0)}
    p = m.BCPolicy(stats, hidden=32)
    keys = set(p.state_dict())
    assert {"obs_mean", "obs_std", "act_mean", "act_std"} <= keys, "stats must be buffers"
    assert all(not b.requires_grad for b in p.buffers())
    assert p(torch.randn(5, 12)).shape == (5, 6)
    with torch.no_grad():
        z = p.forward_normalized(torch.randn(5, 12))
        assert torch.allclose(p(torch.zeros(1, 12)), p.forward_normalized(torch.zeros(1, 12)) * 10 + 50)
    assert z.shape == (5, 6)


@pytest.fixture(scope="module")
def trained(data):
    eps, tr, va = data
    tr_obs, tr_act = m.stack_frames(eps, tr)
    torch.manual_seed(0)
    return m.train_bc(tr_obs, tr_act, epochs=40, seed=0)


def test_beats_baseline(data, trained):
    eps, tr, va = data
    va_obs, va_act = m.stack_frames(eps, va)
    base = m.hold_baseline_mse(va_obs, va_act)
    mse = m.raw_mse(trained, va_obs, va_act)
    print(f"\nval MSE  BC: {mse:.2f}   hold baseline: {base:.2f}")
    assert not trained.training, "return the policy in eval mode"
    assert mse < 0.75 * base, f"BC {mse:.2f} does not clearly beat the baseline {base:.2f} (bar: 75%)"


def test_save_load_roundtrip(trained, tmp_path):
    path = str(tmp_path / "policy.pt")
    m.save_policy(trained, path)
    loaded = m.load_policy(path)
    x = torch.randn(7, 12) * 50
    with torch.no_grad():
        assert torch.allclose(trained(x), loaded(x), atol=1e-5), "reloaded policy differs"
    assert torch.equal(loaded.obs_mean, trained.obs_mean), "stats not restored from file"
