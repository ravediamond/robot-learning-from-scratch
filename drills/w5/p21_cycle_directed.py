def has_cycle(n, edges):
    adj = [[] for _ in range(n)]
    for a, b in edges:
        adj[a].append(b)
    color = [0] * n                                   # 0 white, 1 on stack, 2 done
    for s in range(n):
        if color[s]:
            continue
        stack = [(s, iter(adj[s]))]
        color[s] = 1
        while stack:
            u, it = stack[-1]
            nxt = next(it, None)
            if nxt is None:
                color[u] = 2
                stack.pop()
            elif color[nxt] == 1:
                return True
            elif color[nxt] == 0:
                color[nxt] = 1
                stack.append((nxt, iter(adj[nxt])))
    return False
