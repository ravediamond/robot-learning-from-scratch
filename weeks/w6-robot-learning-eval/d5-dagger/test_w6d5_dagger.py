import pytest
import torch

import dagger_world as w
import w6d5_dagger as m


def test_aggregate_uses_expert_labels():
    O, A = torch.zeros(3, 2), torch.zeros(3, 2)
    V = torch.tensor([[-0.5, -0.3], [-0.4, 0.2]])
    O2, A2 = m.aggregate(O, A, V)
    assert O2.shape == (5, 2) and torch.allclose(A2[3:], w.expert(V))


def test_run_expert_like_policy_and_record():
    class Expertish(torch.nn.Module):
        def forward(self, pos):
            return w.expert(pos) / 0.05
    r = m.run(Expertish(), 50, seed=0, record=True)
    assert r["success"].all() and r["visited"].shape[1] == 2 and len(r["visited"]) > 50 * 20


@pytest.fixture(scope="module")
def results():
    O, A = w.expert_demos(10, seed=0)
    bc = m.train_bc(O, A)
    dg, n = m.dagger(O, A, iterations=2)
    return m.run(bc, 200, seed=1), m.run(dg, 200, seed=1), n, len(O)


def test_dagger_beats_bc(results):
    r_bc, r_dg, n, n0 = results
    s_bc, s_dg = r_bc["success"], r_dg["success"]
    n01, n10 = int((~s_bc & s_dg).sum()), int((s_bc & ~s_dg).sum())
    print(f"\nBC {s_bc.float().mean():.2f}  DAgger {s_dg.float().mean():.2f}  (labels {n0} -> {n})  dagger-only {n01}  bc-only {n10}")
    assert s_bc.float().mean() < 0.7 and s_dg.float().mean() > 0.9
    assert n01 >= 30 and n10 <= 3 and n > n0
