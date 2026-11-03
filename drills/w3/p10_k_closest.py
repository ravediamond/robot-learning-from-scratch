def k_closest(points: list[tuple[float, float]], k: int) -> list[tuple[float, float]]:
    """The k points closest to the origin (e.g. detections nearest the gripper), sorted by distance
    (ties: original order). O(n log k) with a heap."""
    raise NotImplementedError
