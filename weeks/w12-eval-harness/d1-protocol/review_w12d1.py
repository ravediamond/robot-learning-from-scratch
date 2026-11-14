"""REVIEW. SOLUTION.

BUG: each policy runs all its rollouts one after the other, so "which policy" is confounded with "when in
the session". Anything that drifts (battery, light, wear, operator skill) shows up as a policy effect, and
swapping the order swaps the winner. Interleave: blocks of one start with every policy once, the order
rotating from block to block.
"""


def schedule(policies: list[str], n_starts: int, repeats: int) -> list[dict]:
    rows, block = [], 0
    for r in range(repeats):
        for s in range(n_starts):
            k = block % len(policies)
            for p in policies[k:] + policies[:k]:
                rows.append({"trial": len(rows), "policy": p, "start_id": s})
            block += 1
    return rows
