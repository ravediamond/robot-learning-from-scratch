from collections import deque


def sliding_max(torque: list[float], k: int) -> list[float]:
    dq: deque[int] = deque()             # indices, values decreasing front -> back
    out = []
    for i, x in enumerate(torque):
        while dq and torque[dq[-1]] <= x:
            dq.pop()
        dq.append(i)
        if dq[0] <= i - k:
            dq.popleft()
        if i >= k - 1:
            out.append(torque[dq[0]])
    return out
