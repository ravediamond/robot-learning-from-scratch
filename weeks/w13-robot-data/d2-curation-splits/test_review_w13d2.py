from review_w13d2 import split


def test_review_sessions_do_not_leak():
    sessions = [f"s{i // 10}" for i in range(400)]
    train, val = split(sessions)
    assert len(val) >= 80
    assert not {sessions[i] for i in train} & {sessions[i] for i in val}
