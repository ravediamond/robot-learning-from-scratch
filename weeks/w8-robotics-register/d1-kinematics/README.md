# W8 D1: Forward and inverse kinematics for an SO-101-like arm

Book: 10a.1–10a.2 (frames, kinematics), 10 (the hardware from first principles).

## Homogeneous transforms
A pose = rotation R (3×3) + translation p (3) packed in a 4×4 matrix `T = [[R, p], [0, 1]]`.
Compose by matrix product: `T_world_tool = T_world_base @ T_base_link1 @ ...`. Right-hand rule:
```
Rz(θ) = [[c, −s, 0], [s, c, 0], [0, 0, 1]]      Ry(θ) = [[c, 0, s], [0, 1, 0], [−s, 0, c]]
Rx(θ) = [[1, 0, 0], [0, c, −s], [0, s, c]]
```

## The arm (approximate SO-101 geometry, metres)
```
base: Rz(q0) shoulder pan, then up 0.10
shoulder: Ry(q1) lift, then along x 0.116 (upper arm)
elbow:    Ry(q2) flex, then along x 0.135 (forearm)
wrist:    Ry(q3) flex, then Rx(q4) roll, then along x 0.10 (to the gripper tip)
```
At q = 0 the arm points straight along +x: tip at (0.351, 0, 0.10).

**Jacobian** J (3×5): how the tip position moves per joint. Numerically: central differences.
**IK by damped least squares** (position only, 5 joints → 3D: redundant):
`Δq = Jᵀ (J Jᵀ + λ² I)⁻¹ e`, `e = target − tip(q)`, iterate. The damping λ keeps steps sane near
singularities (arm fully stretched, J loses rank).

## Today's build (`w8d1_kinematics.py`)
`rot_x`, `rot_y`, `rot_z`, `transform`, `fk`, `jacobian`, `ik_dls`.

Run: `uv run pytest weeks/w8-robotics-register/d1-kinematics`
