def two_sum(readings: list[int], target: int) -> tuple[int, int] | None:
    seen: dict[int, int] = {}            # value -> first index
    for j, x in enumerate(readings):
        if target - x in seen:
            return seen[target - x], j
        seen.setdefault(x, j)            # keep the FIRST index: smallest i
    return None
