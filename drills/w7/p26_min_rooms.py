import heapq


def min_robots(jobs):
    ends = []
    for s, e in sorted(jobs):
        if ends and ends[0] <= s:
            heapq.heapreplace(ends, e)
        else:
            heapq.heappush(ends, e)
    return len(ends)
