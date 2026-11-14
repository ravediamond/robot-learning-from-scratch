"""Turn a session's records into the report, with YOUR week-12 code.

    python flagship/analyze.py --records <records.csv> --aliases <aliases_SEALED.json> --latency <latency.csv> \
        [--settings <settings.json>] [--hz 30] --out report.md

records.csv: PROTOCOL.md schema (aliases). latency.csv: policy,ms (one row per timed call, after warm-up).
settings.json (optional): {policy: {"chunk", "exec_steps", "hz"}} for the async starvation check."""
import sys as _sys


def _unfinished_lab_hint(exc_type, exc, tb):
    if exc_type is NotImplementedError:
        print("\nThis script runs YOUR week-12 code, which is not finished yet (a TODO still raises NotImplementedError)."
              "\nMake `uv run pytest weeks/w12-eval-harness -k \"not review\"` green first, then run me again.", file=_sys.stderr)
    else:
        _sys.__excepthook__(exc_type, exc, tb)


_sys.excepthook = _unfinished_lab_hint

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for d in ("d1-protocol", "d2-compare-policies", "d3-failure-taxonomy", "d4-latency-target", "d5-repro-report"):
    sys.path.insert(0, str(ROOT / "weeks/w12-eval-harness" / d))

from evalkit import TAXONOMY, wilson  # noqa: E402
from w12d1_protocol import validate_record  # noqa: E402
from w12d2_compare import min_detectable_difference, pairwise_table, rank_probabilities  # noqa: E402
from w12d3_failures import biggest_drop, breakdown, stage_funnel  # noqa: E402
from w12d4_latency import deadline_miss_rate, latency_stats, max_async_latency, starvation_steps  # noqa: E402
from w12d5_report import results_table  # noqa: E402


def load_records(path: Path, aliases: dict) -> tuple[list[dict], list[str]]:
    recs, errors = [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            rec = {"trial": int(row["trial"]), "alias": row["alias"], "start_id": int(row["start_id"]),
                   "outcome": row["outcome"], "failure_tag": row["failure_tag"] or None,
                   "stage_reached": int(row["stage_reached"] or 0), "duration_s": float(row["duration_s"]),
                   "interventions": int(row["interventions"]), "operator": row["operator"]}
            errs = validate_record(rec)
            if errs:
                errors.append(f"trial {rec['trial']}: {'; '.join(errs)}")
                continue
            rec["policy"] = aliases[rec["alias"]]
            recs.append(rec)
    return recs, errors


def paired_outcomes(recs: list[dict], policies: list[str]) -> dict[str, list[int]]:
    """Pair rollouts by (start_id, k-th occurrence of that start for the policy); keep only complete pairs."""
    by = {p: {} for p in policies}
    for r in sorted(recs, key=lambda r: r["trial"]):
        if r["outcome"] == "aborted":
            continue
        slots = by[r["policy"]].setdefault(r["start_id"], [])
        slots.append(1 if r["outcome"] == "success" else 0)
    keys = []
    for s in sorted(set().union(*[set(v) for v in by.values()])):
        depth = min(len(by[p].get(s, [])) for p in policies)
        keys += [(s, k) for k in range(depth)]
    return {p: [by[p][s][k] for s, k in keys] for p in policies}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True)
    ap.add_argument("--aliases", required=True)
    ap.add_argument("--latency", required=True)
    ap.add_argument("--settings")
    ap.add_argument("--hz", type=float, default=30.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    aliases = json.loads(Path(args.aliases).read_text())
    recs, errors = load_records(Path(args.records), aliases)
    policies = sorted(set(aliases.values()))
    lat = {p: [] for p in policies}
    with open(args.latency) as f:
        for row in csv.DictReader(f):
            lat[row["policy"]].append(float(row["ms"]))
    settings = json.loads(Path(args.settings).read_text()) if args.settings else {}
    budget = 1000.0 / args.hz

    L = ["# Evaluation report", ""]
    if errors:
        L += ["## Invalid records (excluded, fix them)", ""] + [f"- {e}" for e in errors] + [""]
    n_abort = sum(1 for r in recs if r["outcome"] == "aborted")
    L += [f"Rollouts analysed: {sum(1 for r in recs if r['outcome'] != 'aborted')} · aborted: {n_abort}", ""]

    summary = []
    for p in policies:
        eps = [r for r in recs if r["policy"] == p and r["outcome"] != "aborted"]
        k = sum(1 for r in eps if r["outcome"] == "success")
        b = breakdown(recs, p)
        top = max(TAXONOMY, key=lambda t: (b[t]["count"], t))
        st = latency_stats(lat[p]) if lat[p] else {"p95": float("nan")}
        summary.append({"policy": p, "k": k, "n": len(eps), "top_failure": top if b[top]["count"] else None,
                        "p95_ms": st["p95"]})
    L += ["## Success", "", results_table(summary), ""]
    n_min = min(s["n"] for s in summary)
    base = sum(s["k"] for s in summary) / max(1, sum(s["n"] for s in summary))
    mde = min_detectable_difference(n_min, round(min(base, 0.99), 2))
    L += [f"Minimum detectable difference at n = {n_min}, base {base:.0%}: "
          f"{'none reachable' if mde is None else f'{mde:.0%}'} (80% power, α = 0.05, unpaired approximation). "
          "Smaller gaps are not resolved by this experiment.", ""]

    paired = paired_outcomes(recs, policies)
    L += [f"## Pairwise (paired on {len(paired[policies[0]])} start slots, exact McNemar, Holm)", "",
          "| A | B | B − A | A fails, B succeeds | A succeeds, B fails | p | p (Holm) |", "|---|---|---|---|---|---|---|"]
    for r in pairwise_table(paired):
        L.append(f"| {r['a']} | {r['b']} | {r['diff']:+.0%} | {r['n01']} | {r['n10']} | {r['p']:.3f} | {r['p_holm']:.3f} |")
    probs = rank_probabilities(paired, 2000, 0)
    L += ["", "P(best), bootstrap over start slots: " + ", ".join(f"{p} {probs[p]:.2f}" for p in policies), ""]

    L += ["## Failure causes (rate over all analysed rollouts, 95% CI)", "",
          "| Cause | " + " | ".join(policies) + " |", "|---|" + "---|" * len(policies)]
    bds = {p: breakdown(recs, p) for p in policies}
    for t in TAXONOMY:
        cells = [f"{bds[p][t]['rate']:.0%} ({bds[p][t]['lo']:.0%}–{bds[p][t]['hi']:.0%})" for p in policies]
        L.append(f"| {t} | " + " | ".join(cells) + " |")
    L += ["", "## Stage funnel (conditional success per stage)", "",
          "| Policy | " + " | ".join(r["stage"] for r in stage_funnel(recs, policies[0])) + " | biggest drop |",
          "|---|" + "---|" * 5]
    for p in policies:
        f = stage_funnel(recs, p)
        cells = ["-" if r["conditional"] is None else f"{r['conditional']:.0%}" for r in f]
        L.append(f"| {p} | " + " | ".join(cells) + f" | {biggest_drop(f)} |")

    L += ["", f"## Latency on target (ms, after warm-up; deadline {budget:.1f} ms)", "",
          "| Policy | n | p50 | p95 | p99 | max | deadline misses | async budget (ms) | starved steps |",
          "|---|---|---|---|---|---|---|---|---|"]
    for p in policies:
        if not lat[p]:
            L.append(f"| {p} | 0 | - | - | - | - | - | - | - |")
            continue
        st = latency_stats(lat[p])
        s = settings.get(p)
        if s:
            cap = f"{max_async_latency(s['chunk'], s['exec_steps'], s['hz']):.0f}"
            starve = str(starvation_steps(lat[p], s["exec_steps"], s["hz"]))
        else:
            cap = starve = "-"
        L.append(f"| {p} | {st['n']} | {st['p50']:.0f} | {st['p95']:.0f} | {st['p99']:.0f} | {st['max']:.0f} | "
                 f"{deadline_miss_rate(lat[p], budget):.0%} | {cap} | {starve} |")
    L += ["", "Deadline misses compare one inference with one control period: for chunked policies the async budget "
          "and starved steps are the numbers that matter.", ""]
    Path(args.out).write_text("\n".join(L) + "\n")
    print(f"report written to {args.out}" + (f" ({len(errors)} invalid records excluded)" if errors else ""))


if __name__ == "__main__":
    main()
