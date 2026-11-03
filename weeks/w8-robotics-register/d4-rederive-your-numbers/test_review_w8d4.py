import csv
import tempfile
from pathlib import Path

import torch

from review_w8d4 import seed_averaged_scores


def test_review_groups_by_episode():
    rows = [{"episode": e, "label": e % 2, "feature": "F", "seed": s, "score": 10 * e + s} for s in range(3) for e in range(4)]
    path = Path(tempfile.mkdtemp()) / "a.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["episode", "label", "feature", "seed", "score"])
        w.writeheader()
        w.writerows(rows)                                   # rows ordered seed-major, not episode-major
    scores, labels = seed_averaged_scores(path, "F")
    assert torch.allclose(scores, torch.tensor([1.0, 11.0, 21.0, 31.0])), f"averaged across episodes: {scores.tolist()}"
    assert labels.tolist() == [0, 1, 0, 1]
