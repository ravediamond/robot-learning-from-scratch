import mujoco
import numpy as np

from w11d2_mujoco import control_step, ee_jacobian, ee_pos, ik_step, load, n_substeps, pd_torque, set_q, solve_ik


def test_substeps():
    assert n_substeps(0.02, 0.002) == 10
    assert n_substeps(1 / 30, 0.002) == 17
    assert n_substeps(0.0005, 0.002) == 1


def test_pd_torque():
    q, qd, qdes = np.array([0.0, 1.0]), np.array([1.0, 0.0]), np.array([1.0, 1.0])
    assert np.allclose(pd_torque(q, qd, qdes, 10, 2), [8.0, 0.0])
    assert np.allclose(pd_torque(q, qd, qdes, 10, 2, bias=np.array([1.0, -1.0])), [9.0, -1.0])


def test_control_step_time_and_limits():
    model = load()
    data = mujoco.MjData(model)
    for _ in range(50):
        control_step(model, data, np.full(5, 1.5), 500.0, 1.0, 0.02)
        assert np.abs(data.ctrl).max() <= 3.0 + 1e-9, "torque must respect the actuator limits"
    assert abs(data.time - 1.0) < 1e-6, f"50 control steps of 20 ms = 1.0 s of physics, got {data.time:.3f}"


def test_gravity_compensation_removes_steady_state_error():
    model = load()
    q_des = np.array([0.0, 1.2, 0.0, 0.0, 0.0])
    errs = {}
    for comp in (False, True):
        data = mujoco.MjData(model)
        for _ in range(150):
            control_step(model, data, q_des, 20.0, 1.0, 0.02, gravity_comp=comp)
        errs[comp] = abs(q_des[1] - data.qpos[1])
    assert errs[False] > 0.01, "PD alone sags under gravity"
    assert errs[True] < errs[False] / 3


def test_jacobian_matches_finite_differences():
    model = load()
    rng = np.random.default_rng(0)
    q = rng.uniform(-1, 1, 5)
    data = set_q(model, mujoco.MjData(model), q)
    J = ee_jacobian(model, data)
    assert J.shape == (3, 5)
    p0 = ee_pos(model, data)
    for j in range(5):
        dq = np.zeros(5)
        dq[j] = 1e-6
        p1 = ee_pos(model, set_q(model, mujoco.MjData(model), q + dq))
        assert np.allclose((p1 - p0) / 1e-6, J[:, j], atol=1e-4)


def test_ik_step_moves_closer_and_keeps_data():
    model = load()
    data = set_q(model, mujoco.MjData(model), np.array([0.2, 0.3, -0.4, 0.1, 0.0]))
    q_before = data.qpos.copy()
    target = ee_pos(model, data) + np.array([0.02, 0.01, -0.02])
    q_new = ik_step(model, data, target)
    assert np.allclose(data.qpos, q_before), "ik_step must not modify data"
    assert np.abs(q_new - q_before).max() <= 0.2 + 1e-9
    d1 = np.linalg.norm(target - ee_pos(model, set_q(model, mujoco.MjData(model), q_new)))
    assert d1 < 0.5 * np.linalg.norm([0.02, 0.01, -0.02])


def test_solve_ik_reaches_targets():
    model = load()
    rng = np.random.default_rng(1)
    for _ in range(6):
        q_true = rng.uniform(-1.0, 1.0, 5)
        target = ee_pos(model, set_q(model, mujoco.MjData(model), q_true))
        q, err = solve_ik(model, target, np.array([0.0, 0.3, -0.3, 0.2, 0.0]))
        assert err < 2e-3, f"error {err * 1000:.1f} mm"
        assert (q >= model.jnt_range[:, 0] - 1e-9).all() and (q <= model.jnt_range[:, 1] + 1e-9).all()
