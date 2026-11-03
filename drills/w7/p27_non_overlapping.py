def min_removals(intervals):
    end, kept = float("-inf"), 0
    for s, e in sorted(intervals, key=lambda t: t[1]):
        if s >= end:
            kept += 1
            end = e
    return len(intervals) - kept
