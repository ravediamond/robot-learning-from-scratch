from review_w12d2 import diff_ci


def test_review_pairing_is_kept():
    a = [1] * 25 + [0] * 25
    b = list(a)
    b[0], b[30] = 0, 1                     # one start flips each way
    lo, hi = diff_ci(a, b)
    assert hi - lo < 0.15, f"CI width {hi - lo:.2f}: the policies agree on 48 of 50 starts"
