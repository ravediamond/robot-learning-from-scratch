from w12d3_failures import biggest_drop, breakdown, cohen_kappa, stage_funnel


def test_kappa():
    assert cohen_kappa(["a", "b", "a"], ["a", "b", "a"]) == 1.0
    r1 = ["g"] * 50 + ["p"] * 50
    r2 = ["g"] * 40 + ["p"] * 10 + ["g"] * 10 + ["p"] * 40
    assert abs(cohen_kappa(r1, r2) - 0.6) < 1e-12
    r1 = ["g"] * 90 + ["p"] * 10
    r2 = ["g"] * 80 + ["p"] * 10 + ["g"] * 10
    assert abs(cohen_kappa(r1, r2) - (0.8 - 0.82) / 0.18) < 1e-12
    assert cohen_kappa(["g"] * 5, ["g"] * 5) == 1.0
    assert abs(cohen_kappa(["a", "b"], ["b", "a"]) + 1.0) < 1e-12


def rec(policy, outcome, tag=None, stage=0):
    return {"policy": policy, "outcome": outcome, "failure_tag": tag, "stage_reached": stage}


def test_breakdown_rates_over_all_episodes():
    recs = [rec("A", "success", stage=4)] * 45 + [rec("A", "failure", "grasp", 1)] * 5 + \
           [rec("A", "aborted")] * 3 + [rec("B", "failure", "collision", 0)] * 2
    b = breakdown(recs, "A")
    assert set(b) == {"perception", "grasp", "transport", "placement", "collision", "timeout", "safety_stop"}
    assert b["grasp"]["count"] == 5 and abs(b["grasp"]["rate"] - 0.1) < 1e-12, "aborted rollouts are excluded"
    assert 0.04 < b["grasp"]["lo"] < 0.05 and 0.21 < b["grasp"]["hi"] < 0.22
    assert b["collision"] == {"count": 0, "rate": 0.0, "lo": 0.0, "hi": b["collision"]["hi"]}


def test_stage_funnel():
    counts = [0] * 4 + [1] * 16 + [2] * 2 + [3] * 3
    recs = [rec("C", "failure", "grasp", k) for k in counts] + [rec("C", "success", stage=4)] * 25
    f = stage_funnel(recs, "C")
    assert [r["reached"] for r in f] == [46, 30, 28, 25]
    assert [round(r["conditional"], 3) for r in f] == [0.92, round(30 / 46, 3), round(28 / 30, 3), round(25 / 28, 3)]
    assert f[-1]["cumulative"] == 0.5
    assert biggest_drop(f) == "grasp"
    none = stage_funnel([rec("D", "failure", "perception", 0)] * 3, "D")
    assert [r["conditional"] for r in none] == [0.0, None, None, None]
    assert biggest_drop(none) == "reach"
