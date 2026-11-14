import torch

from w10d1_pg import GaussianPolicy, discounted_returns, gae, reinforce_loss, train_reinforce


def brute_returns(r, d, gamma):
    T, N = r.shape
    G = torch.zeros(T, N)
    for n in range(N):
        for t in range(T):
            acc, disc = 0.0, 1.0
            for k in range(t, T):
                acc += disc * r[k, n].item()
                if d[k, n]:
                    break
                disc *= gamma
            G[t, n] = acc
    return G


def test_returns_match_brute_force():
    g = torch.Generator().manual_seed(0)
    r = torch.randn(12, 5, generator=g)
    d = torch.rand(12, 5, generator=g) < 0.2
    assert torch.allclose(discounted_returns(r, d, 0.9), brute_returns(r, d, 0.9), atol=1e-5)
    ones = torch.ones(3, 1)
    assert torch.allclose(discounted_returns(ones, torch.zeros(3, 1, dtype=torch.bool), 0.9).flatten(),
                          torch.tensor([2.71, 1.9, 1.0]))


def test_gae_limits():
    g = torch.Generator().manual_seed(1)
    r, v = torch.randn(10, 4, generator=g), torch.randn(10, 4, generator=g)
    d = torch.rand(10, 4, generator=g) < 0.15
    d[-1] = True
    last = torch.randn(4, generator=g)
    a1 = gae(r, v, d, last, 0.95, 1.0)
    assert torch.allclose(a1, discounted_returns(r, d, 0.95) - v, atol=1e-5), "lambda=1 is return minus value"
    a0 = gae(r, v, d, last, 0.95, 0.0)
    nxt = torch.cat([v[1:], last[None]])
    assert torch.allclose(a0, r + 0.95 * (~d).float() * nxt - v, atol=1e-5), "lambda=0 is the TD error"


def test_gae_bootstraps_when_not_done():
    r = torch.zeros(2, 1)
    v = torch.zeros(2, 1)
    d = torch.zeros(2, 1, dtype=torch.bool)
    adv = gae(r, v, d, torch.tensor([10.0]), 0.5, 1.0)
    assert torch.allclose(adv.flatten(), torch.tensor([2.5, 5.0]))


def test_log_prob_sums_action_dims():
    torch.manual_seed(0)
    p = GaussianPolicy(4, 2)
    obs, a = torch.randn(7, 4), torch.randn(7, 2)
    lp = p.log_prob(obs, a)
    assert lp.shape == (7,)
    assert torch.allclose(lp, p.dist(obs).log_prob(a).sum(-1))
    assert p.log_prob(torch.randn(3, 5, 4), torch.randn(3, 5, 2)).shape == (3, 5)


def test_reinforce_loss_weight_has_no_grad():
    logp = torch.tensor([0.5, -1.0], requires_grad=True)
    G = torch.tensor([2.0, 4.0], requires_grad=True)
    b = torch.tensor([1.0, 1.0], requires_grad=True)
    loss = reinforce_loss(logp, G, b)
    assert torch.isclose(loss, torch.tensor(-(0.5 * 1.0 + -1.0 * 3.0) / 2))
    loss.backward()
    assert G.grad is None and b.grad is None
    assert torch.allclose(logp.grad, torch.tensor([-0.5, -1.5]))


def test_reinforce_learns():
    hist = train_reinforce(iters=150, use_baseline=True, seed=0)
    start, end = sum(hist[:5]) / 5, sum(hist[-5:]) / 5
    assert end > start + 5, f"{start:.1f} -> {end:.1f}: return should improve by more than 5"
