import torch
import torch.nn as nn

from review_w3d2 import MHA


def test_review_matches_torch():
    torch.manual_seed(0)
    D, h = 16, 4
    ref = nn.MultiheadAttention(D, h, batch_first=True)
    ours = MHA(D, h)
    with torch.no_grad():
        for i, lin in enumerate([ours.q_proj, ours.k_proj, ours.v_proj]):
            lin.weight.copy_(ref.in_proj_weight[i * D:(i + 1) * D])
            lin.bias.copy_(ref.in_proj_bias[i * D:(i + 1) * D])
        ours.out_proj.load_state_dict(ref.out_proj.state_dict())
    x = torch.randn(2, 6, D)
    assert torch.allclose(ours(x), ref(x, x, x)[0], atol=1e-5)
