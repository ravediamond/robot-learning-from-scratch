from pathlib import Path

import mujoco
import numpy as np

import review_w11d2


def test_review_simulated_time_matches_the_log(monkeypatch):
    model = mujoco.MjModel.from_xml_path(str(Path(__file__).with_name("so101_like.xml")))
    calls = {"n": 0}
    real_step = mujoco.mj_step

    def counting_step(m, d):
        calls["n"] += 1
        return real_step(m, d)

    monkeypatch.setattr(review_w11d2.mujoco, "mj_step", counting_step)
    out = review_w11d2.replay(model, np.zeros((50, 5)))
    assert out.shape == (50, 5)
    assert calls["n"] == 850, f"50 control steps at 30 Hz with 2 ms physics = 50 x 17 steps, got {calls['n']}"
