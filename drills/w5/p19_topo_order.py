import heapq


def topo_order(n, deps):
    adj, indeg = [[] for _ in range(n)], [0] * n
    for a, b in deps:
        adj[a].append(b)
        indeg[b] += 1
    ready = [i for i in range(n) if indeg[i] == 0]
    heapq.heapify(ready)
    out = []
    while ready:
        u = heapq.heappop(ready)
        out.append(u)
        for v in adj[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                heapq.heappush(ready, v)
    return out if len(out) == n else None
