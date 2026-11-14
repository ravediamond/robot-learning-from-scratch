import random

from w12d2_compare import holm, mcnemar_p, min_detectable_difference, pairwise_table, rank_probabilities


def test_mcnemar():
    assert mcnemar_p([1, 1, 0], [1, 1, 0]) == 1.0
    a = [0] * 10 + [1] * 20
    b = [1] * 10 + [1] * 20
    assert abs(mcnemar_p(a, b) - 2 / 2 ** 10) < 1e-12
    a, b = [0, 0, 1, 1, 0], [1, 1, 0, 1, 0]                    # n01 = 2, n10 = 1
    assert abs(mcnemar_p(a, b) - 1.0) < 1e-12
    assert abs(mcnemar_p([0] * 8 + [1] * 2, [1] * 8 + [0] * 2) - 2 * (1 + 10 + 45) / 1024) < 1e-12


def test_holm():
    adj = holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert {k: round(v, 6) for k, v in adj.items()} == {"a": 0.03, "c": 0.06, "b": 0.06}
    assert holm({"x": 0.6, "y": 0.5}) == {"y": 1.0, "x": 1.0}
    assert holm({"only": 0.02}) == {"only": 0.02}


def test_pairwise_table():
    out = {"c": [1, 1, 1, 1, 0, 0], "a": [0, 0, 0, 0, 0, 1], "b": [1, 0, 1, 0, 1, 0]}
    rows = pairwise_table(out)
    assert [(r["a"], r["b"]) for r in rows] == [("a", "b"), ("a", "c"), ("b", "c")]
    ab = rows[0]
    assert ab["n01"] == 3 and ab["n10"] == 1 and abs(ab["diff"] - 2 / 6) < 1e-12
    assert all(r["p_holm"] >= r["p"] for r in rows)
    assert max(r["p_holm"] for r in rows) <= 1.0


def test_rank_probabilities():
    rng = random.Random(1)
    good = [1 if rng.random() < 0.9 else 0 for _ in range(60)]
    bad = [1 if rng.random() < 0.3 else 0 for _ in range(60)]
    p = rank_probabilities({"good": good, "bad": bad, "same": list(bad)}, 500, 0)
    assert abs(sum(p.values()) - 1) < 1e-9
    assert p["good"] > 0.99 and abs(p["bad"] - p["same"]) < 1e-9
    tie = rank_probabilities({"x": [1, 0] * 20, "y": [1, 0] * 20}, 200, 0)
    assert tie == {"x": 0.5, "y": 0.5}, "identical outcomes split every resample"
    assert rank_probabilities({"x": good, "y": bad}, 300, 5) == rank_probabilities({"x": good, "y": bad}, 300, 5)


def test_min_detectable_difference():
    assert min_detectable_difference(50, 0.5) == 0.27
    assert min_detectable_difference(200, 0.5) == 0.14
    assert min_detectable_difference(50, 0.8) == 0.18
    assert min_detectable_difference(5, 0.9) is None
