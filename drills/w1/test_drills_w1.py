import random

from p01_two_sum import two_sum
from p02_valid_brackets import valid_blocks
from p03_sliding_max import sliding_max
from p04_merge_intervals import merge_segments
from p05_first_bad import first_bad


def test_p01_two_sum():
    assert two_sum([2, 7, 11, 15], 9) == (0, 1)
    assert two_sum([3, 3], 6) == (0, 1)
    assert two_sum([1, 2, 3], 100) is None
    assert two_sum([], 1) is None
    assert two_sum([-3, 4, 3, 90], 0) == (0, 2)
    assert two_sum([1, 5, 1, 5], 6) == (0, 1)


def test_p02_valid_blocks():
    assert valid_blocks("move{grip(close)[wait]}")
    assert valid_blocks("")
    assert not valid_blocks("([)]")
    assert not valid_blocks("((")
    assert not valid_blocks(")")
    assert valid_blocks("no brackets at all")


def test_p03_sliding_max():
    assert sliding_max([1, 3, -1, -3, 5, 3, 6, 7], 3) == [3, 3, 5, 5, 6, 7]
    assert sliding_max([4], 1) == [4]
    assert sliding_max([1, 2], 3) == []
    rnd = random.Random(0)
    xs = [rnd.randint(-50, 50) for _ in range(300)]
    for k in (1, 5, 37):
        assert sliding_max(xs, k) == [max(xs[i:i + k]) for i in range(len(xs) - k + 1)]


def test_p04_merge_segments():
    assert merge_segments([(8, 10), (1, 3), (2, 6), (11, 12)]) == [(1, 6), (8, 12)]
    assert merge_segments([]) == []
    assert merge_segments([(1, 4), (2, 3)]) == [(1, 4)]
    assert merge_segments([(1, 2), (4, 5)]) == [(1, 2), (4, 5)]


def test_p05_first_bad():
    for n in (0, 1, 2, 17, 1000):
        for bad_from in sorted({0, n // 3, n - 1, n}):
            if bad_from < 0:
                continue
            calls = []

            def is_bad(i, b=bad_from):
                calls.append(i)
                return i >= b

            assert first_bad(n, is_bad) == bad_from
            assert len(calls) <= max(1, n.bit_length() + 1), f"too many calls: {len(calls)}"
