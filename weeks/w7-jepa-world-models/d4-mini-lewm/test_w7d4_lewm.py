import pytest
import torch

import pixel_world as W
import w7d4_lewm as m


def test_shapes_and_residual():
    enc, pred = m.Encoder(), m.Predictor()
    z = enc(torch.rand(3, 1, 32, 32))
    assert z.shape == (3, 8) and pred(z, torch.zeros(3, 2)).shape == (3, 8)
    with torch.no_grad():
        for p in pred.parameters():
            p.zero_()
    assert torch.allclose(pred(z, torch.ones(3, 2)), z), "zeroed MLP: the predictor must return z (residual)"


@pytest.fixture(scope="module")
def data():
    return W.make_episodes(200, 20, seed=0), W.make_episodes(40, 20, seed=1)


def test_collapse_without_regulariser(data):
    tr, va = data
    enc, pred = m.train_lewm(tr, 0.0, steps=500)
    r = m.latent_report(enc, pred, tr, va)
    print(f"\nlambda 0: {r}")
    assert r["z_std"] < 0.01


def test_sigreg_world_model(data):
    tr, va = data
    enc, pred = m.train_lewm(tr, 0.1)
    r = m.latent_report(enc, pred, tr, va)
    print(f"\nlambda 0.1: {r}")
    assert r["z_std"] > 0.5 and r["ratio"] < 1 / 3 and r["mlp_r2"] > 0.9
