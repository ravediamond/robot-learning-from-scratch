def topo_order(n: int, deps: list[tuple[int, int]]) -> list[int] | None:
    """Tasks 0..n-1; (a, b) means a must run before b (e.g. calibrate before record before train).
    Return an order (Kahn's algorithm, smallest available task first) or None if there is a cycle."""
    raise NotImplementedError
