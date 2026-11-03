from p14_count_islands import count_islands
from p15_grid_shortest_path import grid_shortest_path
from p16_flood_fill import flood_fill
from p17_spread_time import spread_time


def test_p14():
    g = [[1, 1, 0, 0], [0, 1, 0, 1], [1, 0, 0, 1], [0, 0, 1, 1]]
    assert count_islands(g) == 3 and count_islands([]) == 0


def test_p15():
    g = [[0, 0, 0], [1, 1, 0], [0, 0, 0]]
    assert grid_shortest_path(g, (0, 0), (2, 0)) == 6
    assert grid_shortest_path([[0, 1], [1, 0]], (0, 0), (1, 1)) == -1


def test_p16():
    img = [[1, 1, 0], [1, 0, 0], [1, 1, 1]]
    out = flood_fill(img, 0, 0, 5)
    assert out == [[5, 5, 0], [5, 0, 0], [5, 5, 5]] and img[0][0] == 1
    big = [[0] * 300 for _ in range(300)]
    assert flood_fill(big, 0, 0, 1)[299][299] == 1, "must not hit the recursion limit"


def test_p17():
    assert spread_time([[2, 1, 1], [1, 1, 0], [0, 1, 1]]) == 4
    assert spread_time([[2, 1, 1], [0, 1, 1], [1, 0, 1]]) == -1 and spread_time([[0, 2]]) == 0
