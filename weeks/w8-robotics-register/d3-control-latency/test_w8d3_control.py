import torch

import w8d3_control as m


def test_pid_terms():
    p = m.PID(2.0, 0.5, 0.1, 0.1)
    assert abs(p.step(1.0) - (2.0 + 0.5 * 0.1 + 0.0)) < 1e-12
    assert abs(p.step(3.0) - (6.0 + 0.5 * 0.4 + 0.1 * 2.0 / 0.1)) < 1e-12


def test_plant_integral_removes_offset():
    xp = m.simulate_plant(m.PID(2.0, 0.0, 0.0, 0.05), 400, 0, -0.3)
    xi = m.simulate_plant(m.PID(2.0, 1.0, 0.0, 0.05), 400, 0, -0.3)
    assert len(xp) == 401 and xp[0] == 0.0
    assert abs(1.0 - xp[-1] - 0.15) < 0.01, "P only: steady error = disturbance / kp"
    assert abs(1.0 - xi[-1]) < 0.01


def test_delay_makes_high_gain_oscillate():
    lo = m.simulate_plant(m.PID(1.0, 0.0, 0.0, 0.05), 400, 5, 0.0)
    hi = m.simulate_plant(m.PID(12.0, 0.0, 0.0, 0.05), 400, 5, 0.0)
    assert max(lo) < 1.05 and max(hi) > 1.3


def test_chunk_modes():
    r = {mode: m.run_chunked(mode, delay=6) for mode in ("naive", "shift", "blend")}
    assert r["naive"]["commands"].shape == (240,)
    assert r["shift"]["tracking_err"] < 0.6 * r["naive"]["tracking_err"], "shift removes the stale-command lag"
    assert r["blend"]["max_jump"] < 0.7 * r["shift"]["max_jump"], "blending smooths the boundary"
    z = m.run_chunked("shift", delay=1)
    assert z["tracking_err"] < r["shift"]["tracking_err"] + 0.02
