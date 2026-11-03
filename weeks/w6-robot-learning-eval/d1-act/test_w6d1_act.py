import math

import torch

import sim2d
import w6d1_act as m


def test_shapes_and_act():
    pol = m.ACTPolicy(H=8)
    obs, ch = torch.randn(5, 2), torch.randn(5, 8, 2)
    pred, mu, lv = pol(obs, ch, torch.randn(5, 4))
    assert pred.shape == (5, 8, 2) and mu.shape == (5, 4) and lv.shape == (5, 4)
    a0, a1 = pol.act(obs), pol.act(obs)
    assert torch.equal(a0, a1) and torch.allclose(a0, pol.decode(obs, torch.zeros(5, 4)) * sim2d.MAX_STEP)
    s = pol.act(obs, torch.Generator().manual_seed(0), sample_z=True)
    assert not torch.allclose(s, a0)


def test_loss():
    pred, tgt = torch.zeros(2, 3, 2), torch.ones(2, 3, 2)
    tgt[1, 2] = 50.0
    mask = torch.tensor([[1, 1, 1], [1, 1, 0]], dtype=torch.float32)
    mu, lv = torch.zeros(2, 4), torch.zeros(2, 4)
    assert torch.isclose(m.act_loss(pred, tgt, mask, mu, lv, 1.0), torch.tensor(1.0))
    mu2 = torch.full((2, 4), 1.0)                     # KL = 0.5 * 4 * 1 = 2 per sample
    assert torch.isclose(m.act_loss(pred, tgt, mask, mu2, lv, 0.5), torch.tensor(2.0))


def test_temporal_ensembler():
    H, mm = 3, 0.5
    te = m.TemporalEnsembler(H, mm)
    outs = []
    for k in range(5):
        chunk = torch.full((1, H, 1), float(k)) + torch.arange(H).float()[None, :, None] * 0.0
        outs.append(te.step(chunk).item())
    w = torch.exp(-mm * torch.arange(3.0))
    w = w / w.sum()
    assert abs(outs[0] - 0.0) < 1e-6
    w2 = torch.exp(-mm * torch.arange(2.0))
    assert abs(outs[1] - (w2 / w2.sum() * torch.tensor([0.0, 1.0])).sum().item()) < 1e-6
    assert abs(outs[4] - (w * torch.tensor([2.0, 3.0, 4.0])).sum().item()) < 1e-6, "only the last H chunks, oldest weighted most"


def test_ensembler_reads_the_right_index():
    te = m.TemporalEnsembler(4, 0.0)
    for k in range(4):
        out = te.step(torch.arange(4.0)[None, :, None] + 10 * k)    # chunk made at k: values 10k + (0..3)
    # at t = 3: chunk0[3]=3, chunk1[2]=12, chunk2[1]=21, chunk3[0]=30 -> mean 16.5
    assert abs(out.item() - 16.5) < 1e-6


def test_train_fits():
    d = sim2d.make_demos(100, 8, seed=0)
    pol = m.train_act(d, 8, steps=1500)
    with torch.no_grad():
        pred = pol.decode(d["obs"], pol.encode(d["obs"], d["chunks"] / sim2d.MAX_STEP)[0])
    err = ((pred - d["chunks"] / sim2d.MAX_STEP).abs().mean(-1) * d["mask"]).sum() / d["mask"].sum()
    assert not pol.training and err < 0.15, f"reconstruction L1 {err:.3f}"
