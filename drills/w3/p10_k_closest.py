import heapq


def k_closest(points, k):
    heap = []                                       # max-heap of size k via negative distance
    for i, (x, y) in enumerate(points):
        item = (-(x * x + y * y), -i, (x, y))
        if len(heap) < k:
            heapq.heappush(heap, item)
        elif item > heap[0]:
            heapq.heapreplace(heap, item)
    return [p for _, _, p in sorted(heap, key=lambda t: (-t[0], -t[1]))]
