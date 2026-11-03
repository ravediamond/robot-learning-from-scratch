"""WS-091 CI round (exploratory): per-episode out-of-fold predictions of Table 2's grasp-anchor column, and their CIs.

    python3 anchor_predictions.py <controls dir from extract_controls.py> <out dir>

Same rows, features and probe as analyze_controls.py gap 3 (run_w1c_offline.probe_cv on the current feature at the
W1-C1 anchor; label 1 = failure). Writes anchor_predictions.csv (episode, label, feature, seed, score) and
anchor_ci.json: per feature, the mean of the per-seed AUROCs (probe_cv's "auroc", the table's 0.8199 / 0.7067 / 0.5305),
the AUROC of the 5-seed-averaged predictions, and a 95 % episode-bootstrap CI on the latter (2,000 resamples, seed 0,
resamples with one class skipped, percentile) as analyze_controls.boot.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_controls import boot  # noqa: E402
from run_w1c_offline import K_IMAGINE, c1_anchor, probe_cv  # noqa: E402


def main():
    ctl, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    facts = json.loads((ROOT / "data/FACTS.json").read_text())["rollouts"]
    root = (Path.home() / ".cache/huggingface/hub" / ("datasets--" + facts["repo"].replace("/", "--"))
            / "snapshots" / facts["revision"])
    lab = pd.read_csv(root / "meta/rollout_labels.csv")
    lab = lab[lab.outcome.isin(["success", "failure"])].reset_index(drop=True)
    eps = lab.episode_index.astype(int).to_numpy()
    feats = {"W1": ROOT / "outputs/reap/latents", "W2p": ctl / "w2p", "joints": ctl / "joints"}
    Z = {k: {e: np.load(d / f"ep{e:03d}.npy") for e in eps} for k, d in feats.items()}
    rows = []
    for r in lab.itertuples():
        e = int(r.episode_index)
        acts = Z["joints"][e]
        T = len(acts)
        end = T if int(r.reset_start) < 0 else min(int(r.reset_start), T)
        a = c1_anchor(acts[:end, 5])
        if a is None or a < 2 or a + K_IMAGINE > T or np.isnan(Z["W1"][e][a - 2: a + 1]).any():
            continue
        rows.append((e, a, int(r.outcome == "failure")))
    yc = np.array([x[2] for x in rows])
    csv, res = [], {"labels": f"{facts['repo']}@{facts['revision']} meta/rollout_labels.csv", "scored": len(rows),
                    "failures": int(yc.sum()), "bootstrap": {"resamples": 2000, "seed": 0, "unit": "episode",
                                                             "interval": "percentile 95 %"}, "features": {}}
    for k in feats:
        X = np.stack([Z[k][e][a] for e, a, _ in rows]).astype(np.float64)
        cv = probe_cv(X, yc)
        for seed, oof in cv["oof"].items():
            csv += [{"episode": e, "label": int(lab_), "feature": k, "seed": seed, "score": repr(float(s))}
                    for (e, _, lab_), s in zip(rows, oof)]
        avg = np.mean(np.stack(list(cv["oof"].values())), axis=0)
        res["features"][k] = {"auroc_mean_of_seeds": cv["auroc"], "auroc_oof_per_seed": cv["auroc_oof_per_seed"],
                              "auroc_seed_averaged": round(float(roc_auc_score(yc, avg)), 4),
                              "ci95_seed_averaged": boot(yc, avg),
                              "auroc_seed0": round(float(roc_auc_score(yc, cv["oof"][0])), 4),
                              "ci95_seed0": boot(yc, cv["oof"][0])}
    pd.DataFrame(csv).to_csv(out / "anchor_predictions.csv", index=False)
    (out / "anchor_ci.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res["features"], indent=1))


if __name__ == "__main__":
    main()
