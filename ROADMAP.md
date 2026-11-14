# Part 2 roadmap

Weeks 1–9 are the core. A gap analysis of the repo against research-engineer job requirements and
interview loops (2025–2026) added the weeks below. No fixed end: complete beats short.

Status: ✅ in the repo · 📋 planned.

| Track | Where | Status |
|---|---|---|
| Reinforcement learning: returns, GAE, PPO, SAC, IQL, RECAP-style conditioning | `weeks/w10-reinforcement-learning`, `knowledge/11` | ✅ |
| Reward models; MuJoCo basics on an SO-101-like arm | `weeks/w11-rewards-sim-sim2real` d1–d2, `knowledge/12` | ✅ |
| Evaluation harness at scale (your flagship engine) | `weeks/w12-eval-harness`, `knowledge/13`, `flagship/` | ✅ |
| Robot data engineering | `weeks/w13-robot-data`, `knowledge/14` | ✅ |
| Deployment on the robot's computer | `weeks/w14-edge-deployment`, `knowledge/15` | ✅ |
| Live PyTorch rounds (10 timed prompts) | `drills/live` | ✅ |
| System identification, domain randomisation, sim-to-real analysis | W11 d3–d5 | 📋 |
| Profiling and memory: AMP, activation checkpointing, FSDP arithmetic | new week | 📋 |
| Fine-tuning a real pretrained VLA (param groups, freezing, norm-stat checks) + runbook | new week | 📋 |
| World-model diagnostics (action sensitivity, MPPI on a learned model) | new week | 📋 |
| Control and estimation: iLQR vs MPPI, Kalman filter, latency compensation | new week | 📋 |
| Middleware: fixed-rate loops, QoS, watchdogs, a ROS 2 bridge | new week | 📋 |
| JAX port of a flow-matching policy | new week | 📋 |
| Humanoid basics: LIPM, capture point, whole-body control vocabulary | new week | 📋 |
| Numerical drills (Welford, masked mean, SE(2), resampling, pairwise distances) | `drills/w9+` | 📋 |
| Mocks II: timed RL coding, open-problem session, fleet-data design, project talk | new week | 📋 |

For a data / evaluation / deployment research-engineer application, `RE_TRACK.md` gives the order.
