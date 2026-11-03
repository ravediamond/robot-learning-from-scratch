import pytest
import torch
import torch.nn as nn


import w8d5_forensics as m


class Policy(nn.Module):
    def __init__(self, act_dim=6):
        super().__init__()
        self.backbone = nn.Sequential(nn.Linear(12, 32), nn.ReLU())
        self.action_head = nn.Linear(32, act_dim)
        self.register_buffer("act_std", torch.ones(act_dim))

    def forward(self, x):
        return self.action_head(self.backbone(x)) * self.act_std


def _old_ckpt():
    torch.manual_seed(0)
    p = Policy()
    with torch.no_grad():
        p.act_std.fill_(3.0)
    sd = {}
    for k, v in p.state_dict().items():                       # saved from DDP, with the old head name
        sd["module." + k.replace("action_head", "head")] = v.clone()
    return p, sd


def test_report_finds_everything():
    trained, sd = _old_ckpt()
    r = m.load_report(Policy(), sd)
    assert r["expected"] == 5 and r["matched"] == 0 and len(r["missing"]) == 5 and len(r["unexpected"]) == 5
    r2 = m.load_report(Policy(act_dim=4), {k[len("module."):].replace("head", "action_head") if "head" in k else k[len("module."):]: v for k, v in sd.items()})
    assert "action_head.weight" in r2["shape_mismatch"] and r2["matched"] == 2


def test_remap_and_safe_load():
    trained, sd = _old_ckpt()
    fresh = Policy()
    with pytest.raises(RuntimeError):
        m.safe_load(fresh, sd)
    fixed = m.remap_keys(sd, [("module.head.", "action_head."), ("module.", "")])
    before = {k: v.clone() for k, v in fresh.state_dict().items()}
    assert m.safe_load(fresh, fixed) == 5
    assert m.fraction_changed(before, fresh) == 1.0
    x = torch.randn(4, 12)
    assert torch.allclose(fresh(x), trained(x)), "outputs must match the trained policy (buffers included)"


def test_remap_keeps_unmatched():
    assert m.remap_keys({"a.x": 1, "b.y": 2}, [("a.", "c.")]) == {"c.x": 1, "b.y": 2}
