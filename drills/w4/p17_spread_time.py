from collections import deque


def spread_time(grid):
    R, C = len(grid), len(grid[0])
    q = deque((r, c, 0) for r in range(R) for c in range(C) if grid[r][c] == 2)
    fresh = sum(row.count(1) for row in grid)
    g = [row[:] for row in grid]
    t = 0
    while q:
        y, x, t = q.popleft()
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < R and 0 <= nx < C and g[ny][nx] == 1:
                g[ny][nx] = 2
                fresh -= 1
                q.append((ny, nx, t + 1))
    return -1 if fresh else t
