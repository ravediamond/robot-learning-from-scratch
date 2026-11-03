def flood_fill(img, r, c, new):
    out = [row[:] for row in img]
    old = out[r][c]
    if old == new:
        return out
    stack = [(r, c)]
    while stack:
        y, x = stack.pop()
        if 0 <= y < len(out) and 0 <= x < len(out[0]) and out[y][x] == old:
            out[y][x] = new
            stack += [(y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)]
    return out
