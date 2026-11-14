from review_w13d3 import sampling_weights


def test_review_source_totals_match_targets():
    src = ["public"] * 1000 + ["ours"] * 100 + ["sim"] * 10
    w = sampling_weights(src, {"public": 0.4, "ours": 0.4, "sim": 0.2})
    assert abs(w[1000:1100].sum() - 0.4) < 1e-9
    assert abs(w[1100:].sum() - 0.2) < 1e-9
