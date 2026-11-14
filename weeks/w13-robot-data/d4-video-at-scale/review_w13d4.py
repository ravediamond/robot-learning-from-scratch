"""REVIEW. Shard assignment of the streaming dataset (2 GPUs × 8 loader workers). Training reports 1 epoch done
but the loss curve looks like 8 epochs, and a quarter of the shards are never read. Find the bug, fix, replace:
BUG:
"""


def my_shards(n_shards: int, world_size: int, num_workers: int, rank: int, worker: int) -> list[int]:
    return list(range(rank, n_shards, world_size))
