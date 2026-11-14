import numpy as np

from w14d4_safety import SafetyFilter, clamp_limits, is_stale, rate_limit

LIM = np.array([[-1.0, 1.0], [0.0, 2.0]])


def test_primitives():
    assert np.allclose(clamp_limits(np.array([1.5, -0.3]), LIM, 0.1), [0.9, 0.1])
    assert np.allclose(rate_limit(np.array([0.5, 0.5]), np.array([0.9, 0.48]), 0.05), [0.55, 0.48])
    assert is_stale(10.0, 10.76, 0.2) and not is_stale(10.75, 10.76, 0.2) and not is_stale(1.0, 1.2, 0.2)


def test_filter_sequence():
    f = SafetyFilter(LIM, max_step=0.1, max_age=0.2, margin=0.0, initial=np.array([0.0, 1.0]))
    out = f.step(np.array([5.0, 1.05]), capture_time=0.0, now=0.01)
    assert np.allclose(out, [0.1, 1.05]), "clamped to 1.0, then rate-limited to +0.1"
    out[0] = 99.0
    assert f.last[0] == 0.1, "return a copy"
    assert np.allclose(f.step(np.array([np.nan, 1.0]), 0.02, 0.03), [0.1, 1.05])
    assert np.allclose(f.step(np.array([0.2, 1.0]), capture_time=0.0, now=0.5), [0.1, 1.05])
    assert np.allclose(f.step(np.array([0.2, 1.0]), 0.5, 0.51), [0.2, 1.0])
    assert f.events == ["nonfinite", "stale"]


def test_estop_latches_until_reset():
    f = SafetyFilter(LIM, 0.5, 0.2, 0.0, initial=np.zeros(2) + np.array([0.0, 1.0]))
    f.estop()
    for k in range(3):
        assert np.allclose(f.step(np.array([0.3, 1.3]), k, k + 0.01), [0.0, 1.0])
    f.reset()
    assert np.allclose(f.step(np.array([0.3, 1.3]), 5, 5.01), [0.3, 1.3])
    assert f.events == ["estop", "latched", "latched", "latched", "reset"]


def test_default_initial_is_mid_range():
    f = SafetyFilter(LIM, 0.1, 0.2)
    assert np.allclose(f.last, [0.0, 1.0])
