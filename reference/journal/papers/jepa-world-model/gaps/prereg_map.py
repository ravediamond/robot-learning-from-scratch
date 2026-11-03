#!/usr/bin/env python3
"""WS-091 (owner 2026-11-02 14:45): map the paper's 20 register rows (FEASIBILITY.md §2, in order) to their rows in
journal/papers/reproducibility-study/prereg_check.json and keep the audited `class` (not `class_v1`).
Match is by exact `gate` string; a register row may span several prereg rows (REAP v1/v2, LeJEPA, SURVEYOR, RoboJEPA).
Writes the expected file for the paper repo: python3 prereg_map.py OUT.json"""
import hashlib, json, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
SRC = ROOT / "journal/papers/reproducibility-study/prereg_check.json"
MAP = [
    ("G0", ["G0"]),
    ("C1 probe", ["C1 probe"]),
    ("G1b W1", ["G1b W1"]),
    ("G1 probes", ["G1 probes"]),
    ("G1 W2 dynamics", ["G1 W2 dynamics"]),
    ("G6a", ["G6a"]),
    ("G10", ["G10"]),
    ("G6b", ["G6b"]),
    ("W1-A", ["W1-A (screening)"]),
    ("W1-B", ["W1-B (screening)"]),
    ("G11-B-cal", ["G11-B-cal (offline)"]),
    ("W1-C1", ["W1-C1 (offline)"]),
    ("W1-C2", ["W1-C2 (offline)"]),
    ("W1-R", ["W1-R (NEXT_ANGLES §1, D-JEPA operator on frozen W1)"]),
    ("REAP v1", ["REAP monitors (NEXT_ANGLES §2)", "REAP v2 (NEXT_ANGLES §2 amendment)"]),
    ("§30 inverse dynamics", ["§30 inverse dynamics (pre-registered 35f4600)"]),
    ("LeJEPA", ["LeJEPA identifiability, gate 0 (WS-047, arXiv 2605.26379)",
                "LeJEPA identifiability, gate 0 — correction (WS-047)"]),
    ("SURVEYOR", ["SURVEYOR τ, gate 0 (WS-050, NeurIPS 2026 workshop)", "SURVEYOR accept rule, gate 1 (WS-050)"]),
    ("HWM", ["HWM crossover, gate 0 (WS-051, CoRL 2026, arXiv 2604.03208)"]),
    ("RoboJEPA", ["RoboJEPA Gate A", "RoboJEPA Gate B", "RoboJEPA Gate B'"]),
]
KEEP = ("gate", "date", "bar_commit", "bar_date", "bar_how", "result_commit", "result_date", "class", "class_v1", "verdict")

raw = SRC.read_bytes(); d = json.loads(raw)
rows, unmatched = [], []
for reg, names in MAP:
    hits = [dict(index=i, **{k: x[k] for k in KEEP}) for i, x in enumerate(d) if x["gate"] in names]
    if len(hits) != len(names):
        unmatched.append(reg)
    cls = sorted({h["class"] for h in hits})
    rows.append({"register_gate": reg, "class": cls[0] if len(cls) == 1 else ("mixed: " + ", ".join(cls) if cls else None),
                 "prereg_rows": hits})
commit = subprocess.run(["git", "-C", str(ROOT), "log", "-1", "--format=%h", "--", str(SRC)], capture_output=True, text=True).stdout.strip()
out = {"about": "The paper's 20 register rows mapped to journal/papers/reproducibility-study/prereg_check.json (road-to-robotics); "
                "'class' = the audited class of the matched rows (WS-136 audit), 'class_v1' kept for reference. Built by "
                "journal/papers/jepa-world-model/gaps/prereg_map.py.",
       "source": "road-to-robotics journal/papers/reproducibility-study/prereg_check.json", "source_commit": commit,
       "source_sha256": hashlib.sha256(raw).hexdigest(), "unmatched": unmatched, "rows": rows}
Path(sys.argv[1]).write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
from collections import Counter
print(len(rows), "rows,", len(rows) - len(unmatched), "matched; classes", dict(Counter(r["class"] for r in rows)), "unmatched", unmatched)
