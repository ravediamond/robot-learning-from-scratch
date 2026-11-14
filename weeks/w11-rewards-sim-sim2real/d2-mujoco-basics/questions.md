# Questions (out loud)

1. `mj_step` vs `mj_forward`: what does each do, and when do you need the second one?
2. Why does a simulator run physics faster than the policy, and what goes wrong when the substep count is off by one?
3. Derive the steady-state error of a PD joint under gravity. Two ways to remove it, with their costs.
4. What does the damping term λ do in damped least squares IK? What happens at λ = 0 with the arm stretched out?
5. Your SO-101 has position servos, not torque motors. How do you model that in MuJoCo, and which servo parameters do you NOT know?
6. MuJoCo vs Isaac Lab vs MuJoCo MJX/Warp vs Genesis: pick one for (a) system identification of one arm, (b) PPO with 4,096 parallel humanoids, and justify.
