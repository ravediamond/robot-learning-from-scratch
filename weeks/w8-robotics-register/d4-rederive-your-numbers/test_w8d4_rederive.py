import json
from pathlib import Path
from statistics import NormalDist

import w8d4_rederive as m

P = Path(__file__).resolve().parents[3] / "reference" / "journal" / "papers"
COUNTS = P / "failure-monitors" / "results" / "counts_ci.json"
ANCHOR = P / "jepa-world-model" / "gaps"


def test_wilson_audit():
    z = NormalDist().inv_cdf(0.975)
    assert m.audit_counts(COUNTS, z, 1e-6) == [], "with the exact quantile every stored interval must match"
    loose = m.audit_counts(COUNTS, 1.96, 1e-6)
    assert len(loose) > 0 and m.audit_counts(COUNTS, 1.96, 1e-4) == [], "z=1.96 differs only at the 5th decimal"


def test_load_anchor():
    d = m.load_anchor(ANCHOR / "anchor_predictions.csv")
    assert set(d) == {"W1", "W2p", "joints"}
    w = d["W1"]
    assert w["scores"].shape == (5, 65) and int(w["labels"].sum()) == 21


def test_recompute_matches_published():
    stored = json.loads((ANCHOR / "anchor_ci.json").read_text())["features"]
    rec = m.recompute_anchor(ANCHOR / "anchor_predictions.csv")
    for f in ("W1", "W2p", "joints"):
        assert [round(x, 4) for x in rec[f]["per_seed"]] == stored[f]["auroc_oof_per_seed"]
        assert round(rec[f]["mean_of_seeds"], 4) == stored[f]["auroc_mean_of_seeds"]
        assert round(rec[f]["seed_averaged"], 4) == stored[f]["auroc_seed_averaged"]


def test_bootstrap_close_to_published():
    stored = json.loads((ANCHOR / "anchor_ci.json").read_text())["features"]["W1"]["ci95_seed_averaged"]
    d = m.load_anchor(ANCHOR / "anchor_predictions.csv")["W1"]
    lo, hi = m.bootstrap_auroc_ci(d["scores"].mean(0), d["labels"], 2000, 0)
    print(f"\nW1 seed-averaged CI ours ({lo:.4f}, {hi:.4f}) vs stored {stored}")
    assert abs(lo - stored[0]) < 0.03 and abs(hi - stored[1]) < 0.03
