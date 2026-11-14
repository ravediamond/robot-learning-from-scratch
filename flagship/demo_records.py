"""Synthetic session data to dry-run analyze.py before touching the robot. NOT real results.

    python flagship/demo_records.py      # writes flagship/demo/{records.csv, aliases.json, latency.csv, settings.json}"""
import sys as _sys


def _unfinished_lab_hint(exc_type, exc, tb):
    if exc_type is NotImplementedError:
        print("\nThis script runs YOUR week-12 code, which is not finished yet (a TODO still raises NotImplementedError)."
              "\nMake `uv run pytest weeks/w12-eval-harness -k \"not review\"` green first, then run me again.", file=_sys.stderr)
    else:
        _sys.__excepthook__(exc_type, exc, tb)


_sys.excepthook = _unfinished_lab_hint

import csv
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "weeks/w12-eval-harness/d1-protocol"))

from w12d1_protocol import blind_aliases, make_schedule  # noqa: E402

# invented numbers for the dry run: success by policy, easier and harder starts, failure causes, latency profile
TRUE = {"act": 0.70, "smolvla": 0.55, "vla_jepa": 0.60, "mine": 0.78}
CAUSES = {"act": ["grasp", "placement", "timeout"], "smolvla": ["grasp", "grasp", "perception", "collision"],
          "vla_jepa": ["perception", "grasp", "placement"], "mine": ["placement", "grasp"]}
LAT = {"act": (18, 4), "smolvla": (150, 25), "vla_jepa": (95, 15), "mine": (40, 8)}
SETTINGS = {"act": {"chunk": 100, "exec_steps": 50, "hz": 30}, "smolvla": {"chunk": 50, "exec_steps": 25, "hz": 30},
            "vla_jepa": {"chunk": 50, "exec_steps": 25, "hz": 30}, "mine": {"chunk": 40, "exec_steps": 20, "hz": 30}}
STAGE_OF = {"perception": 0, "collision": 0, "grasp": 1, "transport": 3, "placement": 3, "timeout": 2}


def main():
    rng = random.Random(0)
    out = ROOT / "flagship/demo"
    out.mkdir(parents=True, exist_ok=True)
    pols = list(TRUE)
    aliases = blind_aliases(pols, 7)
    difficulty = [rng.uniform(-0.2, 0.2) for _ in range(10)]
    with open(out / "records.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["trial", "alias", "start_id", "outcome", "failure_tag", "stage_reached", "duration_s",
                    "interventions", "operator", "notes"])
        for r in make_schedule(pols, 10, 5, seed=7):
            p = r["policy"]
            if rng.random() < 0.02:
                w.writerow([r["trial"], aliases[p], r["start_id"], "aborted", "", 0, 5.0, 0, "rs", "cube bumped"])
                continue
            ok = rng.random() < min(0.98, max(0.02, TRUE[p] + difficulty[r["start_id"]]))
            tag = "" if ok else rng.choice(CAUSES[p])
            stage = 4 if ok else STAGE_OF[tag]
            dur = round(rng.uniform(15, 35) if ok else rng.uniform(20, 60), 1)
            w.writerow([r["trial"], aliases[p], r["start_id"], "success" if ok else "failure", tag, stage, dur, 0, "rs", ""])
    (out / "aliases.json").write_text(json.dumps({v: k for k, v in aliases.items()}, indent=2))
    with open(out / "latency.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["policy", "ms"])
        for p, (mu, sd) in LAT.items():
            for _ in range(500):
                ms = max(1.0, rng.gauss(mu, sd)) * (3 if rng.random() < 0.01 else 1)
                w.writerow([p, round(ms, 2)])
    (out / "settings.json").write_text(json.dumps(SETTINGS, indent=2))
    print(f"demo data written to {out} (synthetic, for the dry run only)")


if __name__ == "__main__":
    main()
