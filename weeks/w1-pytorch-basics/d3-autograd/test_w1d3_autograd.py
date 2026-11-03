import torch

import w1d3_autograd as m


def test_linear_mse_grads_matches_autograd():
    torch.manual_seed(0)
    N, I, O = 8, 5, 3
    x, y = torch.randn(N, I), torch.randn(N, O)
    W = torch.randn(O, I, requires_grad=True)
    b = torch.randn(O, requires_grad=True)
    ((x @ W.T + b - y) ** 2).mean().backward()
    dW, db = m.linear_mse_grads(x, W.detach(), b.detach(), y)
    assert dW.shape == W.shape and db.shape == b.shape, "gradient must have the shape of the parameter"
    assert torch.allclose(dW, W.grad, atol=1e-5), "dW wrong: check the 1/(N*O) factor and the transpose"
    assert torch.allclose(db, b.grad, atol=1e-5)


def test_linear_mse_grads_no_backward():
    import inspect
    assert ".backward(" not in inspect.getsource(m.linear_mse_grads), "by hand, no autograd"


def test_finite_diff():
    x = torch.tensor([1.0, -2.0, 0.5], dtype=torch.float64)

    def f(v):
        return (v ** 3).sum() + torch.sin(v[0]) * v[1]

    g = m.finite_diff_grad(f, x)
    x_ad = x.clone().requires_grad_(True)
    f(x_ad).backward()
    assert g.shape == x.shape
    assert torch.allclose(g, x_ad.grad, atol=1e-6)
    assert torch.equal(x, torch.tensor([1.0, -2.0, 0.5], dtype=torch.float64)), "x was modified"


def test_fit_line():
    torch.manual_seed(0)
    x = torch.randn(200)
    y = 2.5 * x - 1.0 + 0.01 * torch.randn(200)
    w, b = m.fit_line(x, y)
    assert isinstance(w, float) and isinstance(b, float)
    assert abs(w - 2.5) < 0.05 and abs(b + 1.0) < 0.05


def test_stop_gradient():
    torch.manual_seed(0)
    enc = torch.nn.Linear(4, 4)
    obs = torch.randn(6, 4)
    pred = torch.randn(6, 4, requires_grad=True)
    target = enc(obs)
    loss = m.stop_gradient_mse(pred, target)
    loss.backward()
    assert pred.grad is not None and pred.grad.abs().sum() > 0
    assert enc.weight.grad is None, "gradient leaked into the target encoder"
    assert torch.isclose(loss, ((pred - target) ** 2).mean())
