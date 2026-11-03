import torch

from w1d1_tensors import (action_chunks, flatten_time, normalize, pad_episodes,
                          pick_device, to_tensor, unflatten_time)


def test_to_tensor():
    t = to_tensor([[1, 2, 3], [4, 5, 6]])
    assert t.shape == (2, 3)
    assert t.dtype == torch.float32, "ints must become float32, not int64"


def test_flatten_roundtrip_and_order():
    x = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4)
    f = flatten_time(x)
    assert f.shape == (6, 4)
    assert torch.equal(f[3], x[1, 0]), "row B*t order: sample 1, time 0 must be row 3"
    assert torch.equal(unflatten_time(f, 2, 3), x)


def test_flatten_non_contiguous():
    x = torch.randn(3, 6, 5).transpose(1, 2)  # (3, 5, 6), not contiguous
    assert flatten_time(x).shape == (15, 6)


def test_pad_episodes():
    eps = [torch.ones(3, 2), 2 * torch.ones(5, 2), 3 * torch.ones(1, 2)]
    padded, mask = pad_episodes(eps)
    assert padded.shape == (3, 5, 2)
    assert mask.shape == (3, 5) and mask.dtype == torch.bool
    assert mask.sum().item() == 9
    assert mask[0].tolist() == [True, True, True, False, False]
    assert torch.all(padded[0, 3:] == 0)
    assert torch.all(padded[1] == 2)


def test_normalize():
    x = torch.tensor([[1.0, 5.0], [3.0, 5.0]])
    mean = x.mean(dim=0)
    std = x.std(dim=0)
    z = normalize(x, mean, std)
    assert z.shape == x.shape
    assert torch.allclose(z[:, 0], torch.tensor([-0.7071, 0.7071]), atol=1e-3)
    assert torch.isfinite(z).all(), "joint with std 0 must not give inf/nan"
    z3 = normalize(torch.zeros(4, 7, 2), mean, std)
    assert z3.shape == (4, 7, 2), "must broadcast over any leading dims"


def test_action_chunks():
    T, D, H = 10, 6, 4
    a = torch.arange(T * D, dtype=torch.float32).reshape(T, D)
    c = action_chunks(a, H)
    assert c.shape == (T - H + 1, H, D), f"got {tuple(c.shape)}: is the window dim in the right place?"
    for t in range(T - H + 1):
        assert torch.equal(c[t], a[t:t + H])


def test_pick_device():
    d = pick_device()
    assert isinstance(d, torch.device)
    assert d.type in {"mps", "cuda", "cpu"}
    torch.zeros(1, device=d)  # must be usable
