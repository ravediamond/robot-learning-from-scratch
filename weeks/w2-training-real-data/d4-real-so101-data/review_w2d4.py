"""REVIEW. SOLUTION.

BUG: assumes rows are already in time order. They are only if the parquet files and their rows were
read in order; another machine/file order gives shuffled frames. Sort each group by frame_index.
"""
import numpy as np


def episode_actions(df):
    return {int(e): np.stack(g.sort_values("frame_index")["action"].to_numpy())
            for e, g in df.groupby("episode_index")}
