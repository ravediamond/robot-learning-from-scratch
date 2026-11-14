import numpy as np

from w13d5_flywheel import allocate_collection, correction_dataset, intervention_segments, review_queue


def test_segments():
    f = np.array([int(c) for c in "0001111000011"])
    assert intervention_segments(f, 2) == [(1, 7), (9, 13)]
    assert intervention_segments(f, 0) == [(3, 7), (11, 13)]
    assert intervention_segments(np.array([1, 1, 0, 1]), 1) == [(0, 4)], "touching segments merge"
    assert intervention_segments(np.zeros(5, int), 3) == []


def test_corrections_use_executed_actions():
    T = 10
    log = {"obs": np.arange(T)[:, None] * 1.0, "executed_action": np.full((T, 2), 7.0),
           "policy_action": np.full((T, 2), -1.0), "intervention": np.array([0, 0, 0, 0, 1, 1, 1, 0, 0, 0])}
    ds = correction_dataset(log, context=2)
    assert len(ds) == 1 and ds[0]["segment"] == (2, 7)
    assert (ds[0]["action"] == 7.0).all() and ds[0]["obs"][:, 0].tolist() == [2, 3, 4, 5, 6]


def test_review_queue():
    eps = [{"id": 1, "failure_score": 0.9, "uncertainty": 0.0}, {"id": 2, "failure_score": 0.5, "uncertainty": 0.9},
           {"id": 3, "failure_score": 0.1, "uncertainty": 0.1}, {"id": 4, "failure_score": 0.9, "uncertainty": 0.0}]
    assert review_queue(eps, 3) == [2, 1, 4]


def test_allocation():
    alloc = allocate_collection([0.1, 0.5, 0.2, 0.0, 0.6, 0.3, 0.1, 0.05], [10] * 8, 40)
    assert sum(alloc) == 40 and alloc[3] == 0
    assert alloc[4] == max(alloc) and alloc[1] >= alloc[5] >= alloc[2]
    assert allocate_collection([0.5, 0.5], [10, 10], 3) == [2, 1]
