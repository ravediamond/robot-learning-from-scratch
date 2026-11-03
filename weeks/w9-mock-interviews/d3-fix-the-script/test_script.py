import inspect

import torch

import broken_bc as b
import sim2d


def test_a():
    d = sim2d.make_demos(30, b.H, seed=0)
    tr, va = b.split(d)
    assert set(tr["episode"].tolist()).isdisjoint(va["episode"].tolist())


def test_b():
    src = inspect.getsource(b.main)
    assert src.index("split(") < src.index("norm_stats("), "statistics must come from the training split only"


def test_c():
    p, t = torch.ones(2, 4, 2), torch.zeros(2, 4, 2)
    m = torch.tensor([[1, 1, 1, 1], [1, 0, 0, 0]], dtype=torch.float32)
    assert torch.isclose(b.masked_loss(p, t, m), torch.tensor(1.0))


def test_d(monkeypatch):
    seen = []
    real_lr_at = b.lr_at
    monkeypatch.setattr(b, "lr_at", lambda step, total, *a, **k: (seen.append(real_lr_at(step, total, *a, **k)), seen[-1])[1])
    d = sim2d.make_demos(20, b.H, seed=0)
    b.train(d, b.norm_stats(d), steps=300)
    assert max(seen) > 0.9e-3, "the learning rate actually used must reach its peak during training"


def test_e():
    d = sim2d.make_demos(20, b.H, seed=0)
    pol = b.Policy(b.norm_stats(d))
    pol.train()
    assert b.val_loss(pol, d) == b.val_loss(pol, d), "validation must be deterministic"


def test_f(tmp_path):
    d = sim2d.make_demos(20, b.H, seed=0)
    stats = b.norm_stats(d)
    stats = {k: v * 3 + 1 for k, v in stats.items()}
    pol = b.Policy(stats)
    b.save(pol, str(tmp_path / "p.pt"))
    re = b.load(str(tmp_path / "p.pt"))
    x = torch.randn(5, 2)
    pol.eval()
    re.eval()
    assert torch.allclose(re.act(x), pol.act(x)), "the reloaded policy must act exactly like the saved one"


def test_g(monkeypatch):
    calls = []
    for cls in (torch.optim.Optimizer, torch.nn.Module):
        orig = cls.zero_grad
        monkeypatch.setattr(cls, "zero_grad", lambda self, *a, _o=orig, **k: (calls.append(1), _o(self, *a, **k))[1])
    d = sim2d.make_demos(20, b.H, seed=0)
    b.train(d, b.norm_stats(d), steps=10)
    assert len(calls) >= 10, "gradients must be cleared at every step"
