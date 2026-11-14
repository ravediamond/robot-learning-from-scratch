from collections import Counter

from w12d1_protocol import blind_aliases, expected_success, make_schedule, validate_record

POL = ["act", "smolvla", "vla_jepa", "mine"]


def test_schedule_shape_and_balance():
    s = make_schedule(POL, n_starts=10, repeats=5, seed=0)
    assert len(s) == 200 and [r["trial"] for r in s] == list(range(200))
    blocks = {}
    for r in s:
        blocks.setdefault(r["block"], []).append(r)
    assert len(blocks) == 50
    for rows in blocks.values():
        assert sorted(r["policy"] for r in rows) == sorted(POL), "every policy once per block"
        assert len({(r["start_id"], r["repeat"]) for r in rows}) == 1, "a block is one start of one repeat"
    pairs = Counter((r["policy"], r["start_id"]) for r in s)
    assert set(pairs.values()) == {5} and len(pairs) == 40
    for r in range(5):
        assert sorted(x["start_id"] for x in s if x["repeat"] == r and x["policy"] == "act") == list(range(10))


def test_rotation_balances_positions():
    s = make_schedule(POL, n_starts=10, repeats=4, seed=3)       # 40 blocks, a multiple of 4
    pos = Counter()
    for r in s:
        pos[(r["policy"], r["trial"] % 4)] += 1
    assert set(pos.values()) == {10}, "each policy holds each position in the block equally often"
    first_rows = [r for r in s if r["trial"] % 4 == 0]
    assert len({r["policy"] for r in first_rows[:4]}) == 4, "consecutive blocks start with different policies"


def test_schedule_is_seeded():
    assert make_schedule(POL, 5, 2, seed=1) == make_schedule(POL, 5, 2, seed=1)
    assert make_schedule(POL, 5, 2, seed=1) != make_schedule(POL, 5, 2, seed=2)


def test_blind_aliases():
    a = blind_aliases(POL, seed=0)
    assert sorted(a) == sorted(POL) and sorted(a.values()) == ["A", "B", "C", "D"]
    assert a == blind_aliases(POL, seed=0)
    assert any(blind_aliases(POL, seed=k) != dict(zip(POL, "ABCD")) for k in range(5))


def good():
    return {"trial": 3, "alias": "B", "start_id": 7, "outcome": "failure", "failure_tag": "grasp",
            "duration_s": 41.5, "interventions": 0, "operator": "rs"}


def test_validate_record():
    assert validate_record(good()) == []
    assert validate_record({**good(), "outcome": "success", "failure_tag": None}) == []
    rec = good()
    del rec["operator"], rec["trial"]
    assert sorted(validate_record(rec)) == ["missing operator", "missing trial"]
    assert len(validate_record({**good(), "failure_tag": "bad luck"})) == 1
    assert len(validate_record({**good(), "outcome": "success"})) == 1, "a success cannot carry a failure tag"
    assert len(validate_record({**good(), "outcome": "win", "failure_tag": None})) == 1
    assert len(validate_record({**good(), "duration_s": 0})) == 1
    assert len(validate_record({**good(), "interventions": True})) == 1
    assert len(validate_record({**good(), "start_id": -1, "operator": ""})) == 2


def test_blocking_cancels_drift():
    drift = lambda i: 0.9 - 0.4 * i / 99
    seq = [{"trial": i, "policy": "A" if i < 50 else "B"} for i in range(100)]
    e = expected_success(seq, drift)
    assert abs(e["A"] - e["B"]) > 0.19, "identical policies look 20 points apart"
    blocked = expected_success(make_schedule(["A", "B"], 10, 5, seed=0), drift)
    assert abs(blocked["A"] - blocked["B"]) < 0.01
