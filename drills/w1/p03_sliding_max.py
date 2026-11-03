def sliding_max(torque: list[float], k: int) -> list[float]:
    """Max torque over every window of k consecutive control steps.

    Returns len(torque) - k + 1 values (empty list if k > len(torque)).
    Target: O(n) with a monotonic deque, not O(n*k).

    sliding_max([1, 3, -1, -3, 5, 3, 6, 7], 3) -> [3, 3, 5, 5, 6, 7]
    """
    # TODO
    raise NotImplementedError
