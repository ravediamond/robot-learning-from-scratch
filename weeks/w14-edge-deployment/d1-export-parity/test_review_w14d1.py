import warnings

import numpy as np
import onnxruntime as ort
import torch

from review_w14d1 import export

warnings.filterwarnings("ignore")


def test_review_batch_of_two(tmp_path):
    torch.manual_seed(0)
    model = torch.nn.Linear(4, 2)
    path = str(tmp_path / "m.onnx")
    export(model, 4, path)
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    out = sess.run(None, {"obs": np.ones((2, 4), dtype=np.float32)})[0]
    assert out.shape == (2, 2)
