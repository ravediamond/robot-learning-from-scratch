from collections import deque


def grid_shortest_path(grid, start, goal):
    R, C = len(grid), len(grid[0])
    if grid[start[0]][start[1]] or grid[goal[0]][goal[1]]:
        return -1
    dist = {start: 0}
    q = deque([start])
    while q:
        y, x = q.popleft()
        if (y, x) == goal:
            return dist[(y, x)]
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            n = (y + dy, x + dx)
            if 0 <= n[0] < R and 0 <= n[1] < C and not grid[n[0]][n[1]] and n not in dist:
                dist[n] = dist[(y, x)] + 1
                q.append(n)
    return -1
