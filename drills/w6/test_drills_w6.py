from p22_climb import ways_to_climb
from p23_coin_change import min_coins
from p24_lis import lis_length
from p25_edit_distance import edit_distance


def test_p22():
    assert ways_to_climb(5) == 8 and ways_to_climb(0) == 1 and ways_to_climb(4, (1, 3)) == 3


def test_p23():
    assert min_coins([1, 5, 11], 15) == 3 and min_coins([2], 3) == -1 and min_coins([1], 0) == 0


def test_p24():
    assert lis_length([10, 9, 2, 5, 3, 7, 101, 18]) == 4 and lis_length([]) == 0 and lis_length([3, 3, 3]) == 1


def test_p25():
    assert edit_distance("kitten", "sitting") == 3 and edit_distance("", "abc") == 3 and edit_distance("grasp", "grasp") == 0
