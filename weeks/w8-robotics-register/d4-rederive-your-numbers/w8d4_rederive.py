"""W8 D4: SOLUTION."""
import csv
import json
import math
from pathlib import Path

import torch


def wilson(k, n, z):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def audit_counts(path, z, tol):
    out = []
    for name, v in json.loads(Path(path).read_text()).items():
        lo, hi = wilson(v["k"], v["n"], z)
        if abs(lo - v["wilson95"][0]) > tol or abs(hi - v["wilson95"][1]) > tol:
            out.append(name)
    return out


def load_anchor(path):
    rows = list(csv.DictReader(open(path)))
    out = {}
    for f in sorted({r["feature"] for r in rows}):
        rr = [r for r in rows if r["feature"] == f]
        eps = sorted({int(r["episode"]) for r in rr})
        seeds = sorted({int(r["seed"]) for r in rr})
        col = {e: i for i, e in enumerate(eps)}
        S = torch.zeros(len(seeds), len(eps), dtype=torch.float64)
        lab = torch.zeros(len(eps), dtype=torch.long)
        for r in rr:
            S[seeds.index(int(r["seed"])), col[int(r["episode"])]] = float(r["score"])
            lab[col[int(r["episode"])]] = int(r["label"])
        out[f] = {"episodes": torch.tensor(eps), "labels": lab, "scores": S}
    return out


def auroc(scores, labels):
    s = scores.double()
    order = torch.argsort(s)
    r = torch.empty(len(s), dtype=torch.float64)
    r[order] = torch.arange(1, len(s) + 1, dtype=torch.float64)
    for v in torch.unique(s):
        t = s == v
        r[t] = r[t].mean()
    n1 = int(labels.sum())
    n0 = len(labels) - n1
    return ((r[labels == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)).item()


def recompute_anchor(path):
    out = {}
    for f, d in load_anchor(path).items():
        per = [auroc(d["scores"][i], d["labels"]) for i in range(d["scores"].shape[0])]
        out[f] = {"per_seed": per, "mean_of_seeds": sum(per) / len(per), "seed_averaged": auroc(d["scores"].mean(0), d["labels"])}
    return out


def bootstrap_auroc_ci(scores, labels, B=2000, seed=0):
    g = torch.Generator().manual_seed(seed)
    n = len(scores)
    vals = []
    for _ in range(B):
        i = torch.randint(0, n, (n,), generator=g)
        if 0 < labels[i].sum() < n:
            vals.append(auroc(scores[i], labels[i]))
    q = torch.quantile(torch.tensor(vals, dtype=torch.float64), torch.tensor([0.025, 0.975], dtype=torch.float64))
    return q[0].item(), q[1].item()
