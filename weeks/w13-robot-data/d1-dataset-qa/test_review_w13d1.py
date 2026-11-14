import numpy as np

from review_w13d1 import length_outliers


def test_review_small_outliers_not_masked():
    rng = np.random.default_rng(0)
    lengths = rng.integers(280, 360, size=100).astype(float)
    lengths[0] = 72000.0
    lengths[[5, 6, 7]] = 60.0
    assert set(length_outliers(lengths)) == {0, 5, 6, 7}
