def longest_under_budget(costs, budget):
    best = left = s = 0
    for right, c in enumerate(costs):
        s += c
        while s > budget and left <= right:
            s -= costs[left]
            left += 1
        best = max(best, right - left + 1)
    return best
