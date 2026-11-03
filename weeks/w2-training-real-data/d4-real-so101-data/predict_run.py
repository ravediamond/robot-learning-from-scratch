"""Run AFTER writing your guesses in predict.md."""
import tempfile

import numpy as np
import pandas as pd

from fake_lerobot import make_fake_dataset

root = make_fake_dataset(tempfile.mkdtemp())
df = pd.read_parquet(root / "data" / "chunk-000" / "file-000.parquet")
print("1 dtype:", df["action"].dtype, type(df["action"].iloc[0]).__name__)
print("2 stacked:", np.stack(df["action"].to_numpy()[:100]).shape)
print("3 sorted:", sorted(["file-10.parquet", "file-9.parquet", "file-100.parquet"]), " <- zero-padded names avoid this")
shuf = df.sample(frac=1.0, random_state=0)
g = shuf.groupby("episode_index")["frame_index"].apply(list)
print("4 first group in time order:", g.iloc[0] == sorted(g.iloc[0]), g.iloc[0][:8])
print("5 lag:", round(2 / 30 * 1000, 1), "ms")
print("6 run inspect_real.py on your Mac")
