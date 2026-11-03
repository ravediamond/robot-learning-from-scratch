import pytest
import torch

import vla_world as w
import w5d5_mini_vla as m


def test_shapes_and_prefix_once():
    torch.manual_seed(0)
    model = m.MiniVLA()
    img, tok, st = torch.rand(3, 3, 16, 16), w.tokens_for(torch.tensor([0, 1, 0])), torch.zeros(3, 2)
    assert model.encode_prefix(img, tok, st).shape == (3, 21, 64)
    assert model(img, tok, st, torch.randn(3, 4, 2), torch.rand(3)).shape == (3, 4, 2)
    calls = []
    orig = model.encode_prefix
    model.encode_prefix = lambda *a: (calls.append(1), orig(*a))[1]
    out = model.act(img, tok, st, torch.Generator().manual_seed(0))
    assert out.shape == (3, 4, 2) and len(calls) == 1, "act must encode the prefix once and reuse it"


def test_language_reaches_the_actions():
    torch.manual_seed(0)
    model = m.MiniVLA().eval()
    img, st = torch.rand(2, 3, 16, 16), torch.zeros(2, 2)
    x, t = torch.randn(2, 4, 2), torch.rand(2)
    a = model(img, w.tokens_for(torch.tensor([0, 0])), st, x, t)
    b = model(img, w.tokens_for(torch.tensor([1, 1])), st, x, t)
    assert not torch.allclose(a, b), "changing the instruction must change the output"


@pytest.fixture(scope="module")
def trained():
    return m.train_vla(w.make_vla_demos(400, seed=0))


def test_success_and_instruction_following(trained):
    act = lambda img, tok, st, g: trained.act(img, tok, st, g)
    r = w.run_episodes(act, 200, seed=1)
    rs = w.run_episodes(act, 200, seed=1, swap_instruction=True)
    s, ss = r["success"].float().mean().item(), rs["success"].float().mean().item()
    print(f"\nsuccess {s:.2f}   with swapped instruction {ss:.2f}")
    assert s > 0.85 and ss > 0.85
