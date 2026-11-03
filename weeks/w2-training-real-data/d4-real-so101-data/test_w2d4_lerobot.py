import numpy as np
import pytest

import w2d4_lerobot as m
from fake_lerobot import make_fake_dataset


@pytest.fixture(scope="module")
def root(tmp_path_factory):
    return make_fake_dataset(tmp_path_factory.mktemp("ds"), n_episodes=6, lag=2, seed=0)


def test_info(root):
    info = m.read_info(root)
    assert info["fps"] == 30 and info["total_episodes"] == 6


def test_load_frames(root):
    df = m.load_frames(root)
    assert len(df) == m.read_info(root)["total_frames"]
    assert (np.diff(df["index"].to_numpy()) == 1).all() and df.index[0] == 0


def test_to_episodes(root):
    eps = m.to_episodes(m.load_frames(root))
    assert [e["episode_index"] for e in eps] == list(range(6))
    e = eps[2]
    assert e["obs"].shape == (len(e["timestamp"]), 6) and e["obs"].dtype == np.float32
    assert e["action"].shape == e["obs"].shape and e["task_index"] == 0
    assert (np.diff(e["timestamp"]) > 0).all()


def test_to_episodes_unsorted_rows(root):
    df = m.load_frames(root).sample(frac=1.0, random_state=1)
    eps = m.to_episodes(df)
    assert (np.diff(eps[3]["timestamp"]) > 0).all(), "rows inside an episode must be put in frame order"


def test_check_episode(root):
    eps = m.to_episodes(m.load_frames(root))
    checks = [m.check_episode(e, 30) for e in eps]
    assert [c["gaps"] for c in checks] == [0, 1, 0, 0, 0, 0]
    assert not any(c["nonfinite"] for c in checks)
    assert checks[0]["idle_start"] == 7 and checks[3]["idle_start"] == 7, "frames 0..7 are identical: 7 idle frames after frame 0"
    assert checks[1]["idle_start"] == 0
    bad = dict(eps[1])
    bad["obs"] = eps[1]["obs"].copy()
    bad["obs"][5, 2] = np.nan
    assert m.check_episode(bad, 30)["nonfinite"]


def test_lag(root):
    eps = m.to_episodes(m.load_frames(root))
    assert all(m.action_state_lag(e) == 2 for e in eps)
