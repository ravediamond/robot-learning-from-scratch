import numpy as np

from w13d3_mixtures import ablation_effect, epochs_per_source, fit_power_law, item_weights, temperature_weights


def test_temperature():
    s = {"public": 1000, "ours": 100, "sim": 10}
    p1 = temperature_weights(s, 1)
    assert abs(p1["public"] - 1000 / 1110) < 1e-12
    p3 = temperature_weights(s, 3)
    assert abs(sum(p3.values()) - 1) < 1e-12
    assert abs(p3["public"] / p3["ours"] - 10 ** (1 / 3)) < 1e-9
    big = temperature_weights(s, 1e9)
    assert all(abs(v - 1 / 3) < 1e-6 for v in big.values())


def test_item_weights_hit_source_probabilities():
    src = ["a"] * 900 + ["b"] * 90 + ["c"] * 10
    p = {"a": 0.5, "b": 0.3, "c": 0.2}
    w = item_weights(src, p)
    assert abs(w.sum() - 1) < 1e-12 and len(w) == 1000
    assert abs(w[:900].sum() - 0.5) < 1e-12 and abs(w[990:].sum() - 0.2) < 1e-12
    rng = np.random.default_rng(0)
    draws = rng.choice(1000, size=20000, p=w)
    assert abs((draws >= 990).mean() - 0.2) < 0.01


def test_epochs():
    s = {"public": 1000, "ours": 100}
    e = epochs_per_source(s, {"public": 0.5, "ours": 0.5}, 400)
    assert e == {"public": 0.2, "ours": 2.0}


def test_power_law():
    n = np.array([50.0, 100, 200, 400, 800])
    err = 3.0 * n ** -0.5
    a, b = fit_power_law(n, err)
    assert abs(a - 3.0) < 1e-9 and abs(b - 0.5) < 1e-12
    a, b = fit_power_law(np.array([100.0, 400]), np.array([0.40, 0.20]))
    assert abs(b - 0.5) < 1e-12 and abs(a * 1600 ** -b - 0.10) < 1e-12


def test_ablation_pairing():
    r = ablation_effect([0.62, 0.70, 0.55], [0.58, 0.66, 0.50])
    assert r["effect"] == 0.0433 and r["se"] == 0.0033
    assert r["lo"] > 0, "paired by seed, the effect is clearly positive"
    assert r["unpaired_se"] > 10 * r["se"], "unpaired, the seed noise hides it"
