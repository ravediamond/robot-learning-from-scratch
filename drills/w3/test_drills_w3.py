import random
import statistics

from p10_k_closest import k_closest
from p11_merge_k_sorted import merge_k_sorted
from p12_running_median import RunningMedian
from p13_top_k_frequent import top_k_frequent


def test_p10():
    assert k_closest([(3, 3), (1, 0), (-2, 2), (0, -1)], 2) == [(1, 0), (0, -1)]


def test_p11():
    assert merge_k_sorted([[1, 4, 9], [], [2, 3, 10], [0]]) == [0, 1, 2, 3, 4, 9, 10]


def test_p12():
    rm, seen = RunningMedian(), []
    rnd = random.Random(0)
    for _ in range(101):
        x = rnd.uniform(0, 100)
        rm.add(x)
        seen.append(x)
        assert abs(rm.median() - statistics.median(seen)) < 1e-9


def test_p13():
    assert top_k_frequent(["slip", "miss", "slip", "drop", "miss", "slip"], 2) == ["slip", "miss"]
    assert top_k_frequent(["b", "a"], 1) == ["a"]
