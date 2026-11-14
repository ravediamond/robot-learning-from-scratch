from review_w14d5 import canary_is_better


def test_review_rates_not_counts():
    assert not canary_is_better(20, 40, 90, 900), "0.5 per hour vs 0.1 per hour"
    assert canary_is_better(2, 40, 90, 900)
