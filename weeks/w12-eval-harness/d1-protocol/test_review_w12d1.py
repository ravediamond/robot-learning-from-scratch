from review_w12d1 import schedule


def test_review_policies_share_the_session():
    rows = schedule(["A", "B"], n_starts=10, repeats=2)
    mean_trial = {}
    for p in "AB":
        idx = [r["trial"] for r in rows if r["policy"] == p]
        mean_trial[p] = sum(idx) / len(idx)
    assert abs(mean_trial["A"] - mean_trial["B"]) <= 0.5, mean_trial
