"""Run AFTER writing your guesses AND after your tests pass."""
import json
from pathlib import Path
from statistics import NormalDist

import w8d4_rederive as m

P = Path(__file__).resolve().parents[3] / "reference" / "journal" / "papers"
counts = P / "failure-monitors" / "results" / "counts_ci.json"
anchor = P / "jepa-world-model" / "gaps"
print("1 mismatches with z=1.96 at 1e-6:", len(m.audit_counts(counts, 1.96, 1e-6)), " at 1e-4:", len(m.audit_counts(counts, 1.96, 1e-4)),
      " with z=", NormalDist().inv_cdf(0.975), ":", len(m.audit_counts(counts, NormalDist().inv_cdf(0.975), 1e-6)))
rec = m.recompute_anchor(anchor / "anchor_predictions.csv")
stored = json.loads((anchor / "anchor_ci.json").read_text())["features"]
for f, r in rec.items():
    d = m.load_anchor(anchor / "anchor_predictions.csv")[f]
    ci = m.bootstrap_auroc_ci(d["scores"].mean(0), d["labels"], 2000, 0)
    print(f"2/3 {f:6s} mean-of-seeds {r['mean_of_seeds']:.4f} (stored {stored[f]['auroc_mean_of_seeds']})  "
          f"seed-averaged {r['seed_averaged']:.4f} (stored {stored[f]['auroc_seed_averaged']})  "
          f"CI {tuple(round(x, 4) for x in ci)} (stored {stored[f]['ci95_seed_averaged']})")
print("4 overlapping/non-overlapping marginal CIs is not the test: bootstrap the PAIRED difference on the same episodes")
