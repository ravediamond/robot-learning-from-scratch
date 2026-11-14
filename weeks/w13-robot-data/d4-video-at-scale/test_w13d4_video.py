from w13d4_video import (ShuffleBuffer, assign_shards, compressed_tb, decode_cores_needed, expected_decoded_frames,
                         raw_bytes_per_hour)


def test_budgets():
    assert raw_bytes_per_hour(640, 480, 30) == 640 * 480 * 3 * 30 * 3600
    assert abs(compressed_tb(10_000, 3, 1.0) - 13.5) < 1e-9
    assert expected_decoded_frames(1) == 1 and expected_decoded_frames(60) == 30.5
    assert decode_cores_needed(4000, 400, 30.5) == 305
    assert decode_cores_needed(4000, 400, 1) == 10


def test_shards_disjoint_and_complete():
    seen = []
    for rank in range(4):
        for worker in range(3):
            seen += assign_shards(50, 4, 3, rank, worker)
    assert sorted(seen) == list(range(50))
    assert assign_shards(50, 4, 3, 1, 2) == [5, 17, 29, 41]
    assert assign_shards(4, 4, 3, 3, 2) == []


def test_shuffle_buffer():
    sb = ShuffleBuffer(100, seed=0)
    out = [x for x in (sb.push(i) for i in range(1000)) if x is not None] + sb.drain()
    assert sorted(out) == list(range(1000)) and sb.buf == []
    assert out != list(range(1000))
    displacement = sum(abs(i - x) for i, x in enumerate(out)) / 1000
    assert 20 < displacement < 300, "items move by about the buffer size, not across the whole stream"
    one = ShuffleBuffer(1, 0)
    assert [one.push(i) for i in range(5)] == [None, 0, 1, 2, 3]
