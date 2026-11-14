import torch

import rl_envs
from w10d2_helpers import GaussianPolicy, collect, value_net
from w10d2_ppo import normalize, ppo_clip_loss, ppo_update, train_ppo, value_loss


def obj(rho, A, clip=0.2):
    r = torch.tensor([rho], requires_grad=True)
    loss = ppo_clip_loss(r.log(), torch.zeros(1), torch.tensor([A]), clip)
    loss.backward()
    return -loss.item(), -r.grad.item()


def test_clip_values_and_gradients():
    assert obj(1.0, 1.0) == (1.0, 1.0)
    v, g = obj(1.5, 1.0)
    assert abs(v - 1.2) < 1e-6 and abs(g) < 1e-6, "good action already pushed past 1+eps: no more gradient"
    v, g = obj(0.5, 1.0)
    assert abs(v - 0.5) < 1e-6 and abs(g - 1.0) < 1e-6, "good action made LESS likely: not clipped"
    v, g = obj(0.5, -1.0)
    assert abs(v + 0.8) < 1e-6 and abs(g) < 1e-6
    v, g = obj(1.5, -1.0)
    assert abs(v + 1.5) < 1e-6 and abs(g + 1.0) < 1e-6, "bad action made MORE likely: not clipped"


def test_normalize_and_value_loss():
    a = normalize(torch.tensor([1.0, 2.0, 3.0, 10.0]))
    assert abs(a.mean().item()) < 1e-6 and abs(a.std().item() - 1) < 1e-4
    assert torch.isclose(value_loss(torch.tensor([1.0, 3.0]), torch.tensor([0.0, 1.0])), torch.tensor(1.25))


def test_update_changes_both_nets_and_keeps_batch():
    torch.manual_seed(0)
    policy, value = GaussianPolicy(4, 2), value_net(4)
    opt = torch.optim.Adam(list(policy.parameters()) + list(value.parameters()), lr=1e-3)
    batch = collect(rl_envs.Reach2D(32, seed=0), policy, value)
    before_p = policy.net[0].weight.clone()
    before_v = value[0].weight.clone()
    stored = batch["logp"].clone()
    ppo_update(policy, value, opt, batch, generator=torch.Generator().manual_seed(0))
    assert not torch.equal(before_p, policy.net[0].weight) and not torch.equal(before_v, value[0].weight)
    assert torch.equal(stored, batch["logp"]), "stored log-probs are the OLD policy: never recompute them in place"


def test_ppo_solves_reach():
    policy, hist = train_ppo(iters=40, seed=0)
    ret = rl_envs.evaluate(lambda o: policy.dist(o).mean)["return"].mean().item()
    assert ret > -8.0, f"mean return {ret:.1f}; random is about -21, the expert about -3"
    assert hist[-1] > hist[0] + 5
