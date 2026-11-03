import pytest
import torch

import pixel_world as W
import w7d5_plan as m
from probes_ref import mlp_probe


def test_cem_on_known_dynamics():
    class Additive(torch.nn.Module):          # z_next = z + a (exact), D = 2
        def forward(self, z, a):
            return z + a
    z0 = torch.zeros(4, 2)
    target = torch.tensor([[0.3, 0.3], [-0.3, 0.1], [0.0, -0.4], [0.2, -0.2]])
    a = m.cem_plan(Additive(), z0, m.latent_cost(target), H=5, N=300, K=30, iters=5, generator=torch.Generator().manual_seed(0))
    assert a.shape == (4, 2) and (a.abs() <= W.AMAX + 1e-6).all()
    assert ((a * target).sum(-1) > 0).all(), "first action must point toward the goal"


def test_probe_cost_shape():
    probe = torch.nn.Linear(8, 2)
    c = m.probe_cost(probe, torch.zeros(3, 2))(torch.randn(3, 7, 8))
    assert c.shape == (3, 7)


@pytest.fixture(scope="module")
def model():
    from w7d4_lewm import train_lewm
    tr = W.make_episodes(200, 20, seed=0)
    enc, pred = train_lewm(tr, 0.1)
    with torch.no_grad():
        probe = mlp_probe(enc(tr["obs"].flatten(0, 1)), tr["pos"].flatten(0, 1))
    return enc, pred, probe


def test_planning(model):
    enc, pred, probe = model
    g = torch.Generator().manual_seed(5)
    start, goal = torch.rand(100, 2, generator=g) * 1.6 - 0.8, torch.rand(100, 2, generator=g) * 1.6 - 0.8
    ok = lambda f: ((f - goal).norm(dim=-1) < 0.15).float().mean().item()
    rand = lambda z, gen: (torch.rand(len(z), 2, generator=gen) * 2 - 1) * W.AMAX
    s_rand = ok(m.run_planner(enc, pred, start, goal, None, 25, 0, act_fn=rand))
    s_lat = ok(m.run_planner(enc, pred, start, goal, lambda zg, gp: m.latent_cost(zg), 25, 0))
    s_prb = ok(m.run_planner(enc, pred, start, goal, lambda zg, gp: m.probe_cost(probe, gp), 25, 0))
    print(f"\nsuccess  random {s_rand:.2f}  latent-L2 {s_lat:.2f}  probe-decoded {s_prb:.2f}")
    assert s_lat > s_rand + 0.2 and s_prb > 0.7


def test_surprise(model):
    enc, pred, _ = model
    s, lab = m.surprise_scores(enc, pred, 50, 0.6, 3)
    assert s.shape == lab.shape == (1000,) and lab.sum() == 50
    pos, neg = s[lab], s[~lab]
    auroc = (pos[:, None] > neg[None]).float().mean().item()
    print(f"\nsurprise AUROC {auroc:.3f}")
    assert auroc > 0.95
