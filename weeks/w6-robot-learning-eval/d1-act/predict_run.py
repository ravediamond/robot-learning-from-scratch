"""Run AFTER writing your guesses AND after your tests pass (~6 min)."""
import torch

import policies_ref as P
import sim2d
import w6d1_act as m

d = sim2d.make_demos(200, 8, seed=0)
for beta in (0.01, 0.05, 1.0):
    for seed in (0, 1):
        pol = m.train_act(d, 8, beta=beta, seed=seed)
        with torch.no_grad():
            mu, _ = pol.encode(d["obs"][:2000], d["chunks"][:2000] / sim2d.MAX_STEP)
        r0 = P.rollout(lambda o, g: pol.act(o, g), 200, 4, seed=1)
        rz = P.rollout(lambda o, g: pol.act(o, g, sample_z=True), 200, 4, seed=1)
        print(f"beta {beta:<5} seed {seed}: z info (mu std max) {mu.std(0).max():.2f}   "
              f"collision z=0 {r0['collision'].float().mean():.2f}   z~N {rz['collision'].float().mean():.2f}")
