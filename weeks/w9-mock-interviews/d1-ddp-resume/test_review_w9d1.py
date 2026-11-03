import torch
import torch.nn as nn

from review_w9d1 import resume, save


def test_review_adam_state_survives(tmp_path):
    torch.manual_seed(0)
    model = nn.Linear(4, 1)
    opt = torch.optim.Adam(model.parameters(), lr=0.1)
    for _ in range(5):
        model(torch.randn(8, 4)).pow(2).mean().backward()
        opt.step()
        opt.zero_grad()
    save(str(tmp_path / "c.pt"), model, opt, 5)
    m2 = nn.Linear(4, 1)
    o2 = torch.optim.Adam(m2.parameters(), lr=0.1)
    resume(str(tmp_path / "c.pt"), m2, o2)
    s1, s2 = opt.state_dict()["state"], o2.state_dict()["state"]
    assert s2 and torch.equal(s1[0]["exp_avg_sq"], s2[0]["exp_avg_sq"]), "Adam's moments were not restored"
