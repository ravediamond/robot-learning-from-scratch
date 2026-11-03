import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from review_w8d5 import load_pretrained


def test_review_loud_failure():
    src = nn.Linear(3, 3)
    path = str(Path(tempfile.mkdtemp()) / "ckpt.pt")
    torch.save({"model." + k: v for k, v in src.state_dict().items()}, path)        # wrapped keys
    with pytest.raises(Exception):
        load_pretrained(nn.Linear(3, 3), path)
