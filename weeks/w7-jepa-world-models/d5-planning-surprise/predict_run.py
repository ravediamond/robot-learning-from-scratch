"""Run AFTER writing your guesses AND after your tests pass (~2 min)."""
import torch

import pixel_world as W
import w7d5_plan as m
from probes_ref import mlp_probe
from w7d4_lewm import train_lewm

tr = W.make_episodes(200, 20, seed=0)
enc, pred = train_lewm(tr, 0.1)
with torch.no_grad():
    probe = mlp_probe(enc(tr["obs"].flatten(0, 1)), tr["pos"].flatten(0, 1))
g = torch.Generator().manual_seed(5)
start, goal = torch.rand(100, 2, generator=g) * 1.6 - 0.8, torch.rand(100, 2, generator=g) * 1.6 - 0.8
rand = lambda z, gen: (torch.rand(len(z), 2, generator=gen) * 2 - 1) * W.AMAX
for name, mk in [("random", None), ("latent L2", lambda zg, gp: m.latent_cost(zg)),
                 ("probe-decoded", lambda zg, gp: m.probe_cost(probe, gp))]:
    final = m.run_planner(enc, pred, start, goal, mk, steps=25, seed=0) if mk else m.run_planner(enc, pred, start, goal, None, 25, 0, act_fn=rand)
    print(f"2 {name:14s} success {((final - goal).norm(dim=-1) < 0.15).float().mean():.2f}")
p = torch.rand(400, 2, generator=g) * 1.6 - 0.8
with torch.no_grad():
    z = enc(W.render(p))
i, j = torch.randint(0, 400, (3000,), generator=g), torch.randint(0, 400, (3000,), generator=g)
rk = lambda x: x.argsort().argsort().float()
print("3 spearman", round(torch.corrcoef(torch.stack([rk((z[i] - z[j]).norm(dim=-1)), rk((p[i] - p[j]).norm(dim=-1))]))[0, 1].item(), 3))
for jump in (0.6, 0.1):
    s, lab = m.surprise_scores(enc, pred, n_episodes=50, jump=jump, seed=3)
    pos, neg = s[lab], s[~lab]
    print(f"4 teleport {jump}: AUROC {(pos[:, None] > neg[None]).float().mean():.3f}")
