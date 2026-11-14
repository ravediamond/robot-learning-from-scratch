"""REVIEW. Run sheet generator used for the "policy A beats policy B by 18 points" slide. Someone reran
the comparison with the two policies swapped in the config and B now wins by 18 points.
Find the bug, fix, replace:
BUG:
"""


def schedule(policies: list[str], n_starts: int, repeats: int) -> list[dict]:
    rows = []
    for p in policies:
        for r in range(repeats):
            for s in range(n_starts):
                rows.append({"trial": len(rows), "policy": p, "start_id": s})
    return rows
