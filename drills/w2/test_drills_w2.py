import random

from p06_pair_sum_sorted import pair_sum_sorted
from p07_dedupe_sorted import dedupe_sorted
from p08_longest_under_budget import longest_under_budget
from p09_three_sum_zero import three_sum_zero


def test_p06():
    assert pair_sum_sorted([1, 2, 4, 7, 11], 15) == (2, 4)
    assert pair_sum_sorted([1, 2], 10) is None and pair_sum_sorted([], 0) is None


def test_p07():
    a = [1, 1, 2, 3, 3, 3, 7]
    k = dedupe_sorted(a)
    assert k == 4 and a[:k] == [1, 2, 3, 7] and dedupe_sorted([]) == 0


def test_p08():
    assert longest_under_budget([2, 1, 3, 1, 1, 1, 5], 4) == 3
    assert longest_under_budget([5, 6], 4) == 0 and longest_under_budget([], 3) == 0


def test_p09():
    assert three_sum_zero([-1, 0, 1, 2, -1, -4]) == [(-1, -1, 2), (-1, 0, 1)]
    assert three_sum_zero([0, 0, 0, 0]) == [(0, 0, 0)]
    rnd = random.Random(0)
    a = [rnd.randint(-6, 6) for _ in range(25)]
    brute = sorted({tuple(sorted((a[i], a[j], a[k]))) for i in range(25) for j in range(i + 1, 25) for k in range(j + 1, 25) if a[i] + a[j] + a[k] == 0})
    assert three_sum_zero(a) == brute
