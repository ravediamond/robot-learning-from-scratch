"""Run AFTER writing your guesses AND after your tests pass."""
import torch
import torch.nn as nn

import w8d5_forensics as m

torch.manual_seed(0)
trained = nn.Sequential(nn.Linear(6, 32), nn.ReLU(), nn.Linear(32, 6))
ddp_sd = {"module." + k: v for k, v in trained.state_dict().items()}
fresh = nn.Sequential(nn.Linear(6, 32), nn.ReLU(), nn.Linear(32, 6))
res = fresh.load_state_dict(ddp_sd, strict=False)
print("1 strict=False: missing", len(res.missing_keys), "unexpected", len(res.unexpected_keys), "-> loaded", 4 - len(res.missing_keys))
try:
    fresh.load_state_dict(ddp_sd, strict=True)
except RuntimeError as e:
    print("2 strict=True raises:", str(e).splitlines()[0][:80])
x = torch.randn(3, 6)
print("3 same outputs:", torch.allclose(fresh(x), trained(x)))
print("4 report:", m.load_report(fresh, ddp_sd))
