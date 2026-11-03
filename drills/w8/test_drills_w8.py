from p30_lru import LRUCache
from p31_trie import Trie
from p32_ring_buffer import RingBuffer
from p33_rotate_matrix import rotate_90


def test_p30():
    c = LRUCache(2)
    c.put(1, "a")
    c.put(2, "b")
    assert c.get(1) == "a"
    c.put(3, "c")
    assert c.get(2) is None and c.get(1) == "a" and c.get(3) == "c"


def test_p31():
    t = Trie()
    for w in ["pick", "pick up", "place", "push"]:
        t.insert(w)
    assert t.search("pick") and not t.search("pic") and t.starts_with("pic") and not t.starts_with("q")


def test_p32():
    r = RingBuffer(3)
    for x in [1, 2, 3, 4]:
        r.push(x)
    assert r.values() == [2, 3, 4] and abs(r.mean() - 3.0) < 1e-12
    r2 = RingBuffer(5)
    r2.push(10)
    assert r2.values() == [10] and r2.mean() == 10


def test_p33():
    m = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    rotate_90(m)
    assert m == [[7, 4, 1], [8, 5, 2], [9, 6, 3]]
