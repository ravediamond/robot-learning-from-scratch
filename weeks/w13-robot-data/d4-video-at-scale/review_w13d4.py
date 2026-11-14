"""REVIEW. SOLUTION.

BUG: the worker id is ignored, so the 8 workers of a rank all read the same shards (each sample 8 times per
"epoch"), and with uneven splits some shards are never assigned. Split by the global reader id
rank * num_workers + worker, stepping by world_size * num_workers.
"""


def my_shards(n_shards: int, world_size: int, num_workers: int, rank: int, worker: int) -> list[int]:
    R = world_size * num_workers
    return list(range(rank * num_workers + worker, n_shards, R))
