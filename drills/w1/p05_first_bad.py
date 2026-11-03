from typing import Callable


def first_bad(n: int, is_bad: Callable[[int], bool]) -> int:
    """Checkpoints 0..n-1. Once a checkpoint is bad, all later ones are bad (monotone).
    Return the first bad index, or n if none is bad. Call is_bad O(log n) times.

    (Same pattern: first frame where the gripper is closed; git bisect; the smallest
    chunk size that meets a latency budget.)
    """
    # TODO
    raise NotImplementedError
