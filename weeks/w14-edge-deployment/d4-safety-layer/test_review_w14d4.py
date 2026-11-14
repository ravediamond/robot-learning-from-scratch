from review_w14d4 import fresh


def test_review_buffered_frame_is_stale():
    assert not fresh({"capture_ts": 10.0, "recv_ts": 10.9}, now=10.91)
    assert fresh({"capture_ts": 10.85, "recv_ts": 10.9}, now=10.91)
