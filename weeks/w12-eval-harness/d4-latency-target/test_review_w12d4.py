from review_w12d4 import p99


def test_review_tail_is_measured_not_assumed():
    assert p99([10.0] * 95 + [200.0] * 5) == 200.0
