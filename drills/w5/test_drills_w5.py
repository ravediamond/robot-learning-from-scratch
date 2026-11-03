from p18_union_find import count_components
from p19_topo_order import topo_order
from p20_dijkstra import dijkstra
from p21_cycle_directed import has_cycle


def test_p18():
    assert count_components(5, [(0, 1), (1, 2), (3, 4)]) == 2 and count_components(3, []) == 3


def test_p19():
    assert topo_order(4, [(0, 2), (1, 2), (2, 3)]) == [0, 1, 2, 3]
    assert topo_order(3, [(0, 1), (1, 2), (2, 0)]) is None


def test_p20():
    d = dijkstra(4, [(0, 1, 4), (0, 2, 1), (2, 1, 2), (1, 3, 1)], 0)
    assert d == [0.0, 3.0, 1.0, 4.0]
    assert dijkstra(2, [], 0)[1] == float("inf")


def test_p21():
    assert has_cycle(3, [(0, 1), (1, 2), (2, 0)]) and not has_cycle(3, [(0, 1), (0, 2), (1, 2)])
    chain = [(i, i + 1) for i in range(5000)]
    assert not has_cycle(5001, chain), "iterative: no recursion limit"
