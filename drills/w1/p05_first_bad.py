from typing import Callable


def first_bad(n: int, is_bad: Callable[[int], bool]) -> int:
    lo, hi = 0, n                        # answer in [lo, hi]; hi = n means "none bad"
    while lo < hi:
        mid = (lo + hi) // 2
        if is_bad(mid):
            hi = mid
        else:
            lo = mid + 1
    return lo
