from review_w13d4 import my_shards


def test_review_each_shard_read_once():
    all_reads = [s for r in range(2) for w in range(8) for s in my_shards(64, 2, 8, r, w)]
    assert sorted(all_reads) == list(range(64))
