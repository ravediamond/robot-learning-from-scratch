"""Generate the blinded run sheet for a session from YOUR week-12 code.

    python flagship/make_sheet.py --policies act smolvla vla_jepa mine --starts 10 --repeats 5 --seed 42 --out flagship/session1

Writes run_sheet.csv (aliases only: give this to the operator) and aliases_SEALED.json (keep closed until analysis)."""
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
sys.path.insert(0, str(ROOT / "weeks/w12-eval-harness/d1-protocol"))

from w12d1_protocol import blind_aliases, make_schedule  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policies", nargs="+", required=True)
    ap.add_argument("--starts", type=int, default=10)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--seed", type=int, required=True, help="write it in the manifest")
    ap.add_argument("--out", default="flagship/session")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    aliases = blind_aliases(args.policies, args.seed)
    rows = make_schedule(args.policies, args.starts, args.repeats, args.seed)
    with open(out / "run_sheet.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["trial", "block", "start_id", "alias", "outcome", "failure_tag", "stage_reached", "duration_s",
                    "interventions", "operator", "notes"])
        for r in rows:
            w.writerow([r["trial"], r["block"], r["start_id"], aliases[r["policy"]], "", "", "", "", "", "", ""])
    (out / "aliases_SEALED.json").write_text(json.dumps({v: k for k, v in aliases.items()}, indent=2))
    print(f"{len(rows)} rollouts -> {out / 'run_sheet.csv'}; aliases sealed in {out / 'aliases_SEALED.json'}")


if __name__ == "__main__":
    main()
