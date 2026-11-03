import ast
import inspect
import textwrap

import torch

import w1d2_broadcast as m


def _no_loops(fn):
    tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
    loops = [n for n in ast.walk(tree) if isinstance(n, (ast.For, ast.While, ast.comprehension))]
    assert not loops, f"{fn.__name__}: no Python loops or comprehensions allowed"


def test_pairwise_sq_dist():
    torch.manual_seed(0)
    a, b = torch.randn(5, 3), torch.randn(7, 3)
    d = m.pairwise_sq_dist(a, b)
    assert d.shape == (5, 7)
    assert torch.allclose(d, torch.cdist(a, b) ** 2, atol=1e-4)
    _no_loops(m.pairwise_sq_dist)


def test_select_per_row():
    x = torch.tensor([[1.0, 2, 3], [4, 5, 6]])
    out = m.select_per_row(x, torch.tensor([2, 0]))
    assert torch.equal(out, torch.tensor([3.0, 4.0]))
    _no_loops(m.select_per_row)


def test_masked_mean():
    x = torch.zeros(3, 4, 2)
    x[0, :2] = 2.0        # valid
    x[0, 2:] = 100.0      # padding garbage: must be ignored
    x[1] = 1.0
    mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 1], [0, 0, 0, 0]], dtype=torch.bool)
    out = m.masked_mean(x, mask)
    assert out.shape == (3, 2)
    assert torch.allclose(out[0], torch.tensor([2.0, 2.0])), "padding leaked or wrong denominator"
    assert torch.allclose(out[1], torch.tensor([1.0, 1.0]))
    assert torch.all(out[2] == 0) and torch.isfinite(out).all()
    _no_loops(m.masked_mean)


def test_batched_matvec():
    torch.manual_seed(0)
    W, x = torch.randn(4, 3, 5), torch.randn(4, 5)
    assert torch.allclose(m.batched_matvec(W, x), (W @ x[..., None]).squeeze(-1), atol=1e-5)
    assert "einsum" in inspect.getsource(m.batched_matvec)


def test_best_of_n():
    torch.manual_seed(0)
    B, N, H, D = 4, 8, 5, 6
    cand = torch.randn(B, N, H, D)
    scores = torch.randn(B, N)
    best = m.best_of_n(cand, scores)
    assert best.shape == (B, H, D)
    for b in range(B):
        assert torch.equal(best[b], cand[b, scores[b].argmax()])
    _no_loops(m.best_of_n)
