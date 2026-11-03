import torch
import torch.nn as nn
import torch.nn.functional as F

import w1d4_training as m


def _data(n=256, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 6, generator=g)
    y = torch.sin(x[:, :3]) + 0.5 * x[:, 3:]
    return x, y


def _batches(x, y, bs=32):
    return [(x[i:i + bs], y[i:i + bs]) for i in range(0, len(x), bs)]


def test_mlp_shapes_and_params():
    net = m.MLP(6, 64, 3, n_hidden=2)
    assert net(torch.randn(5, 6)).shape == (5, 3)
    assert net(torch.randn(2, 7, 6)).shape == (2, 7, 3), "Linear works on any leading dims"
    assert m.count_params(net) == (6 * 64 + 64) + (64 * 64 + 64) + (64 * 3 + 3)
    assert m.count_params(m.MLP(6, 64, 3, n_hidden=0)) == 6 * 3 + 3


def test_count_trainable_only():
    net = m.MLP(4, 8, 2, n_hidden=1)
    total = m.count_params(net)
    for p in list(net.parameters())[:2]:
        p.requires_grad_(False)
    assert m.count_params(net) == total - (4 * 8 + 8)
    assert m.count_params(net, trainable_only=False) == total


def test_train_epoch_lowers_loss():
    torch.manual_seed(0)
    x, y = _data()
    net = m.MLP(6, 64, 3)
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    first = m.train_epoch(net, _batches(x, y), opt, F.mse_loss)
    for _ in range(30):
        last = m.train_epoch(net, _batches(x, y), opt, F.mse_loss)
    assert isinstance(first, float)
    assert last < 0.3 * first


def test_evaluate_deterministic_no_graph_restores_mode():
    torch.manual_seed(0)
    x, y = _data(64)
    net = m.MLP(6, 64, 3, dropout=0.5)
    net.train()
    a = m.evaluate(net, _batches(x, y), F.mse_loss)
    b = m.evaluate(net, _batches(x, y), F.mse_loss)
    assert a == b, "evaluate must be deterministic (dropout off)"
    assert net.training, "evaluate must restore train mode"
    seen = []
    net.register_forward_hook(lambda mod, inp, out: seen.append(out.requires_grad))
    m.evaluate(net, _batches(x, y), F.mse_loss)
    assert not any(seen), "evaluate built a graph: use torch.no_grad()"


def test_evaluate_weighted_mean():
    net = nn.Identity()
    batches = [(torch.zeros(3, 1), torch.ones(3, 1)), (torch.zeros(1, 1), torch.zeros(1, 1))]
    assert abs(m.evaluate(net, batches, F.mse_loss) - 0.75) < 1e-6, "weight each batch by its size"


def test_overfit_one_batch():
    torch.manual_seed(0)
    x, y = _data(32)
    final = m.overfit_one_batch(m.MLP(6, 128, 3), (x, y), F.mse_loss)
    assert final < 1e-3, f"could not overfit 32 samples (final loss {final:.4f}): the loop is broken"
