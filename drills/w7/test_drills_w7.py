from p26_min_rooms import min_robots
from p27_non_overlapping import min_removals
from p28_lcs import lcs_length
from p29_knapsack import knapsack


def test_p26():
    assert min_robots([(0, 30), (5, 10), (15, 20)]) == 2 and min_robots([(0, 5), (5, 10)]) == 1 and min_robots([]) == 0


def test_p27():
    assert min_removals([(1, 2), (2, 3), (3, 4), (1, 3)]) == 1 and min_removals([(1, 2), (1, 2), (1, 2)]) == 2


def test_p28():
    plan = ["reach", "grasp", "lift", "move", "place"]
    done = ["reach", "lift", "grasp", "move", "place"]
    assert lcs_length(plan, done) == 4 and lcs_length([], plan) == 0


def test_p29():
    assert knapsack([1, 3, 4, 5], [1, 4, 5, 7], 7) == 9 and knapsack([5], [10], 4) == 0
