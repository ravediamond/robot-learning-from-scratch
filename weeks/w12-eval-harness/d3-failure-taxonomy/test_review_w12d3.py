from review_w12d3 import cause_rates


def test_review_rates_per_episode():
    recs = [{"policy": "new", "outcome": "success", "failure_tag": None}] * 45 + \
           [{"policy": "new", "outcome": "failure", "failure_tag": "grasp"}] * 5 + \
           [{"policy": "base", "outcome": "success", "failure_tag": None}] * 30 + \
           [{"policy": "base", "outcome": "failure", "failure_tag": "grasp"}] * 10 + \
           [{"policy": "base", "outcome": "failure", "failure_tag": "timeout"}] * 10
    assert cause_rates(recs, "new") == {"grasp": 0.1}
    assert cause_rates(recs, "base") == {"grasp": 0.2, "timeout": 0.2}
