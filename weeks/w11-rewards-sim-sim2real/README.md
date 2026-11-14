# Week 11: Reward models, simulation, sim-to-real

Primers: `knowledge/11-reinforcement-learning.md` §7 (rewards), `knowledge/12-simulation-sim2real.md`.

RL needs a reward and a place to fail cheaply. This week you build both: learned rewards (the piece that
lets RL run on a real arm) and a MuJoCo model of an SO-101-like arm you can control, differentiate and invert.

| Day | Lab | Review bug |
|---|---|---|
| d1 | success classifier, precision-first threshold, pairwise progress reward | pairs built across episodes |
| d2 | MuJoCo: PD control with substeps, gravity compensation, Jacobian, IK | wrong number of substeps |

Days 3–5 (system identification, domain randomisation, sim-to-real analysis) are on the roadmap
(`ROADMAP.md`); until then, primer 12 covers what to say about them.

MuJoCo labs use numpy, not torch. `so101_like.xml` is a 5-joint arm with the SO-101's proportions, written
for this repo: it is NOT the official model. For real work use the official SO-101 MJCF from
`TheRobotStudio/SO-ARM100`.
