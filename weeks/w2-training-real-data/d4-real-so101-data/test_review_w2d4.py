import numpy as np
import pandas as pd

from review_w2d4 import episode_actions


def test_review_time_order():
    rows = [{"episode_index": 0, "frame_index": t, "action": np.full(6, float(t), np.float32)} for t in range(20)]
    df = pd.DataFrame(rows).sample(frac=1.0, random_state=0)       # files read in another order
    a = episode_actions(df)[0]
    assert a[:, 0].tolist() == list(range(20)), "actions are not in time order"
