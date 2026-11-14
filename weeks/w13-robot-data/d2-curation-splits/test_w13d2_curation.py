import numpy as np

from w13d2_curation import Ledger, dataset_fingerprint, group_split, near_duplicates, resample


def test_resample():
    t = np.stack([np.linspace(0, 1, 11), np.linspace(0, 2, 11)], axis=1)
    r = resample(t, 3)
    assert r.shape == (3, 2) and np.allclose(r, [[0, 0], [0.5, 1], [1, 2]])
    assert np.allclose(resample(t, 11), t)


def test_near_duplicates_ignore_speed():
    base = np.linspace(0, 1, 240)[:, None] * np.array([[1, -1, 0.5, 0, 0, 1]])
    slow = np.linspace(0, 1, 360)[:, None] * np.array([[1, -1, 0.5, 0, 0, 1]])
    other = np.linspace(0, 1, 240)[:, None] * np.array([[-1, 1, 0.5, 0, 0, 1]])
    noisy = base + 0.01 * np.random.default_rng(0).standard_normal(base.shape)
    assert near_duplicates([base, other, slow, noisy]) == [(0, 2), (0, 3), (2, 3)]


def test_group_split_never_shares_groups():
    groups = [f"s{i // 10}" for i in range(200)] + ["big"] * 30
    tr, va = group_split(groups, 0.2, seed=1)
    assert sorted(tr + va) == list(range(230))
    assert not {groups[i] for i in tr} & {groups[i] for i in va}
    assert len(va) >= 0.2 * 230 and len(va) <= 0.2 * 230 + 30
    assert (tr, va) == group_split(groups, 0.2, seed=1)
    assert group_split(groups, 0.2, seed=2) != (tr, va)


def test_fingerprint():
    h = ["aa", "bb", "cc"]
    assert dataset_fingerprint(h) == dataset_fingerprint(["cc", "aa", "bb"])
    assert dataset_fingerprint(h) != dataset_fingerprint(["aa", "bb", "dd"])
    assert len(dataset_fingerprint(h)) == 64


def test_ledger():
    led = Ledger()
    led.remove(3, "frozen_camera", "qa-v1")
    led.remove(7, "duplicate", "qa-v1")
    led.restore(3, "qa-v2")
    led.remove(9, "dead_joint", "qa-v2")
    assert led.removed() == {7, 9}
    assert led.apply(list(range(10))) == [0, 1, 2, 3, 4, 5, 6, 8]
    assert len(led.entries) == 4, "append only"
