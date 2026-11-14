import torch

import toy_flow
from w14d3_rtc import guided_sample, seam_jump, shift_previous, soft_mask


def test_shift_previous():
    p = torch.arange(8.0)
    assert shift_previous(p, 3).tolist() == [3, 4, 5, 6, 7, 7, 7, 7]
    assert shift_previous(p, 0).tolist() == p.tolist()


def test_soft_mask():
    m = soft_mask(10, fixed=2, overlap=6)
    assert m.tolist()[:2] == [1.0, 1.0] and m[6:].sum() == 0
    assert torch.allclose(m[2:6], torch.tensor([0.8, 0.6, 0.4, 0.2]))
    assert (m[1:] <= m[:-1]).all()


def test_unguided_is_the_plain_sampler():
    g1, g2 = torch.Generator().manual_seed(3), torch.Generator().manual_seed(3)
    a = guided_sample(lambda x, t: toy_flow.velocity(x, t, 0.0), torch.zeros(toy_flow.H), torch.zeros(toy_flow.H), 20, 5.0, g1)
    b = toy_flow.sample(0.0, 20, g2)
    assert torch.allclose(a, b, atol=1e-6)


def test_rtc_keeps_the_mode_and_smooths_the_seam():
    prev = toy_flow.modes(0.0)[0]
    target = shift_previous(prev, 5)
    start = target[0].item()
    mask = soft_mask(toy_flow.H, fixed=3, overlap=8)
    g = torch.Generator().manual_seed(0)
    naive = [toy_flow.sample(start, 20, g) for _ in range(200)]
    rtc = [guided_sample(lambda x, t: toy_flow.velocity(x, t, start), target, mask, 20, 30.0, g) for _ in range(200)]
    switch = lambda cs: sum(1 for c in cs if c[-1] < start) / len(cs)
    assert switch(naive) > 0.3 and switch(rtc) < 0.05
    jn = sum(seam_jump(target, c, 3) for c in naive) / 200
    jr = sum(seam_jump(target, c, 3) for c in rtc) / 200
    assert jr < 0.5 * jn
    assert seam_jump(torch.tensor([0.0, 1.0, 2.0]), torch.tensor([9.0, 9.0, 2.5]), 2) == 1.5
