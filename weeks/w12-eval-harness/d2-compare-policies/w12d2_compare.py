"""W12 D2: several policies, paired outcomes on the same start list. outcomes: {policy: [0/1, ...]}, aligned."""
import math
import random
from itertools import combinations
from statistics import NormalDist


def mcnemar_p(a: list[int], b: list[int]) -> float:
    """Exact two-sided McNemar test on paired 0/1 outcomes. n01 = a fails and b succeeds, n10 = the reverse,
    n = n01 + n10. p = min(1, 2 * sum_{k=0}^{min(n01, n10)} C(n, k) / 2^n); 1.0 when n == 0."""
    n01 = sum(1 for x, y in zip(a, b) if x == 0 and y == 1)
    n10 = sum(1 for x, y in zip(a, b) if x == 1 and y == 0)
    n = n01 + n10
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(n01, n10) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def holm(pvals: dict[str, float]) -> dict[str, float]:
    """Holm step-down adjusted p-values. Sort ascending (ties: by key); the i-th (from 0) is multiplied by
    (m - i); then take the running maximum so that adjusted values never decrease; cap at 1."""
    items = sorted(pvals.items(), key=lambda kv: (kv[1], kv[0]))
    m, running, out = len(items), 0.0, {}
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        out[k] = running
    return out


def pairwise_table(outcomes: dict[str, list[int]]) -> list[dict]:
    """One row per pair (a, b) of policy names, a < b alphabetically, pairs in sorted order:
    {"a", "b", "diff": mean(b) - mean(a), "n01", "n10", "p": mcnemar_p, "p_holm": Holm-adjusted over all pairs}.
    Keys for holm: f"{a}|{b}"."""
    names = sorted(outcomes)
    rows = []
    for a, b in combinations(names, 2):
        xa, xb = outcomes[a], outcomes[b]
        rows.append({"a": a, "b": b, "diff": sum(xb) / len(xb) - sum(xa) / len(xa),
                     "n01": sum(1 for x, y in zip(xa, xb) if x == 0 and y == 1),
                     "n10": sum(1 for x, y in zip(xa, xb) if x == 1 and y == 0),
                     "p": mcnemar_p(xa, xb)})
    adj = holm({f"{r['a']}|{r['b']}": r["p"] for r in rows})
    for r in rows:
        r["p_holm"] = adj[f"{r['a']}|{r['b']}"]
    return rows


def rank_probabilities(outcomes: dict[str, list[int]], n_boot: int, seed: int) -> dict[str, float]:
    """Probability that each policy has the highest success rate, by resampling START INDICES (shared by all
    policies) with replacement, rng = random.Random(seed), idx = [rng.randrange(n) for _ in range(n)] per
    resample. Ties for the top split the credit equally. Values sum to 1."""
    rng = random.Random(seed)
    names = list(outcomes)
    n = len(outcomes[names[0]])
    wins = {p: 0.0 for p in names}
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        scores = {p: sum(outcomes[p][i] for i in idx) for p in names}
        top = max(scores.values())
        best = [p for p in names if scores[p] == top]
        for p in best:
            wins[p] += 1 / len(best)
    return {p: w / n_boot for p, w in wins.items()}


def min_detectable_difference(n: int, p_base: float, alpha: float = 0.05, power: float = 0.8):
    """Smallest d in steps of 0.01 (d = 0.01, 0.02, ..., p_base + d <= 1) such that n rollouts per arm are enough
    to detect p_base vs p_base + d with a two-sided unpaired test:
      n_needed = (z_a * sqrt(2 * pbar * (1 - pbar)) + z_b * sqrt(p1 (1 - p1) + p2 (1 - p2)))^2 / d^2,
      pbar = (p1 + p2) / 2, z_a = NormalDist().inv_cdf(1 - alpha / 2), z_b = NormalDist().inv_cdf(power).
    Return round(d, 2), or None if no d qualifies."""
    za, zb = NormalDist().inv_cdf(1 - alpha / 2), NormalDist().inv_cdf(power)
    p1 = p_base
    for step in range(1, 101):
        d = step / 100
        p2 = p1 + d
        if p2 > 1 + 1e-12:
            return None
        pbar = (p1 + p2) / 2
        need = (za * math.sqrt(2 * pbar * (1 - pbar)) + zb * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2 / d ** 2
        if need <= n:
            return round(d, 2)
    return None
