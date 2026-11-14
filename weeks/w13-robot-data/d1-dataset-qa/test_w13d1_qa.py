import numpy as np

import fake_fleet
from w13d1_qa import content_hash, dead_joints, limit_fraction, longest_frozen_run, qa_report, robust_z


def test_frozen_run():
    assert longest_frozen_run(np.array([1, 2, 2, 2, 3, 3])) == 3
    assert longest_frozen_run(np.array([5])) == 1 and longest_frozen_run(np.array([])) == 0
    assert longest_frozen_run(np.array([1, 2, 3])) == 1


def test_dead_joints_and_limits():
    s = np.zeros((10, 3))
    s[:, 0] = np.linspace(0, 1, 10)
    s[:, 2] = np.linspace(0, 0.5, 10)
    assert dead_joints(s) == [1]
    assert dead_joints(np.zeros((10, 3))) == [], "nothing moves: not a dead-joint problem"
    lim = np.array([[-1.0, 1.0]] * 3)
    assert limit_fraction(s, lim) == 0.1, "only the last frame reaches joint 0's upper limit"
    assert limit_fraction(s, lim, margin=0.0) == 0.1
    assert limit_fraction(s - 2.0, lim) == 1.0


def test_robust_z_resists_masking():
    x = np.array([300.0] * 50 + [310.0] * 49 + [6000.0])
    x[10] = 45.0
    z = robust_z(x)
    assert abs(z[-1]) > 3.5 and abs(z[10]) > 3.5
    naive = (x - x.mean()) / x.std()
    assert abs(naive[10]) < 3, "the outlier hides the other outlier with mean and sd"
    assert (robust_z(np.ones(5)) == 0).all()


def test_content_hash_ignores_metadata():
    eps, _ = fake_fleet.make_fleet(n=160, seed=1)
    assert content_hash(eps[150]) == content_hash(eps[20])
    assert content_hash(eps[150]) != content_hash(eps[21])
    e = dict(eps[5])
    e["state"] = e["state"].astype(np.float32)
    assert content_hash(e) != content_hash(eps[5]) or np.array_equal(e["state"], eps[5]["state"])


def test_qa_report_finds_every_injected_defect():
    eps, truth = fake_fleet.make_fleet(seed=0)
    rep = qa_report(eps, fake_fleet.LIMITS, fps=30)
    q = rep["quarantine"]
    for i in truth["frozen"]:
        assert "frozen_camera" in q[i]
    for i in truth["dead"]:
        assert "dead_joint" in q[i]
    assert "limits" in q[45]
    assert "length_outlier" in q[60] and "length_outlier" in q[99]
    assert q[150] == ["duplicate"] and 20 not in q
    assert rep["duplicates"] == [(20, 150)]
    expected = set().union(*truth.values())
    assert set(q) == expected, f"false alarms: {set(q) - expected}"
