"""PROVIDED. Run on your Mac (needs internet): downloads the tabular part of your bowls dataset
(parquet + meta, no videos) to ~/data/so101_bowls_all_trimmed.

    uv run python download_bowls.py            # tables only, a few MB
    uv run python download_bowls.py --videos   # also the mp4s (for W2 D5, large)
"""
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

REPO = "ravediamond/so101_bowls_all_trimmed"
dest = Path.home() / "data" / "so101_bowls_all_trimmed"
patterns = ["meta/*", "meta/**", "data/**"] + (["videos/**"] if "--videos" in sys.argv else [])
snapshot_download(REPO, repo_type="dataset", local_dir=dest, allow_patterns=patterns)
print("downloaded to", dest)
