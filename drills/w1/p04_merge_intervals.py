def merge_segments(segments: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """A failure detector flags frame segments [start, end] (inclusive). Segments may overlap
    or touch (end + 1 == next start counts as touching). Merge them; return sorted by start.

    merge_segments([(8, 10), (1, 3), (2, 6), (11, 12)]) -> [(1, 6), (8, 12)]
    """
    # TODO
    raise NotImplementedError
