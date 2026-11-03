# Chapter 10a. Robotics fundamentals for learning engineers

Recruiter and hiring-manager guides for robot-learning roles name one red flag more often than any other: a deep-learning background with no grounding in the physical robot — someone who has trained a policy in a notebook but cannot say what a joint limit, a control loop, a camera calibration or a frame convention does to it. This chapter is that grounding, from zero, for the arm this book uses. It assumes chapter P1 and nothing else, and every number in it was computed with the code shown, run on 2026-10-24 (the SO-101 examples read the real URDF in `shared/robot/so101_new_calib.urdf`; run them from the repository root).

Chapter 10 describes this hardware; this chapter explains the ideas under it. Read 10a.1–10a.3 before any interview for a robot-learning role; the rest when a question touches it.

```python
import math
import numpy as np
np.set_printoptions(precision=4, suppress=True)
```

## 10a.1 Frames, positions and rotations

**A frame** is a coordinate system: an origin and three perpendicular axes. A robot has many: the base frame bolted to the table, one frame per link, the gripper frame, each camera's frame, the frame of the mat on the table. Every position is a position *in some frame*, and most robotics bugs are a vector expressed in one frame and used in another.

**A rotation** turns one frame's axes into another's. It is stored as a 3 × 3 **rotation matrix** R whose columns are the rotated frame's axes written in the original frame. Rotations do not commute: turning 90° about x then 90° about z is not the same as z then x. Other representations you will meet: **roll–pitch–yaw** angles (three rotations about fixed axes, the URDF convention; easy to read, but they have a singularity at pitch ±90°, "gimbal lock"); **axis–angle** (a unit axis and an angle; what a revolute joint does); **quaternions** (four numbers, no singularity, the usual storage format in software). A policy that outputs end-effector orientations needs one of these, and the choice matters for learning: angles wrap around at ±180°, which an MSE loss does not know.

**A homogeneous transform** packs a rotation R and a translation t into one 4 × 4 matrix, T = [[R, t], [0, 0, 0, 1]], so that "rotate then translate" becomes one matrix product on a point written as (x, y, z, 1). Transforms chain by multiplication: the pose of the gripper in the base frame is T_base→link1 · T_link1→link2 · … · T_linkN→gripper. Reading that product left to right, each factor is expressed in the frame produced by the previous one.

Conventions to state out loud, because they differ between libraries and are a classic silent bug: which axis is up (z for most robots, y for many graphics and camera libraries); whether a transform maps points *from* A *to* B or describes *B's pose in A* (the same matrix, two names, and opposite directions in different codebases); angles in degrees or radians (URDFs are in radians; LeRobot's SO-101 driver can report joints in degrees or in a normalised −100…100 range depending on its configuration, so check the dataset's `info.json` before mixing data from two setups).

## 10a.2 Kinematics: from joint angles to the gripper, and back

**Forward kinematics (FK)** answers: given the joint angles, where is the gripper? It is the chain of transforms above, one fixed offset per link (from the robot's geometry) and one rotation per joint (by the joint's current angle). FK always has exactly one answer.

**Inverse kinematics (IK)** answers the reverse: which joint angles put the gripper at a given position (and orientation)? It can have **no** answer (the target is out of reach or outside the joint limits), **exactly one**, **several** (an elbow-up and an elbow-down solution reach the same point), or **infinitely many** (when the arm has more joints than the task constrains, called **redundancy**).

The smallest example, a two-link arm in a plane with link lengths 12 and 14 cm:

```python
# ---- 1. two-link planar arm: forward and inverse kinematics ----
l1, l2 = 0.12, 0.14                       # link lengths in metres
def fk2(q1, q2):
    x = l1 * math.cos(q1) + l2 * math.cos(q1 + q2)
    y = l1 * math.sin(q1) + l2 * math.sin(q1 + q2)
    return x, y
print('fk2', [round(v, 4) for v in fk2(math.radians(30), math.radians(45))])

def ik2(x, y):
    c2 = (x * x + y * y - l1 * l1 - l2 * l2) / (2 * l1 * l2)   # law of cosines
    if abs(c2) > 1:
        return []                                             # out of reach: no solution
    sols = []
    for s in (+1, -1):                                        # elbow down / elbow up
        q2 = s * math.acos(c2)
        q1 = math.atan2(y, x) - math.atan2(l2 * math.sin(q2), l1 + l2 * math.cos(q2))
        sols.append((q1, q2))
    return sols
x, y = fk2(math.radians(30), math.radians(45))
for q1, q2 in ik2(x, y):
    print('ik2', round(math.degrees(q1), 2), round(math.degrees(q2), 2), [round(v, 4) for v in fk2(q1, q2)])
print('ik2 out of reach', ik2(0.3, 0.0))
```

Output: FK of (30°, 45°) puts the tip at (0.1402, 0.1952) m. IK returns two solutions, (30°, 45°) and (78.65°, −45°): elbow down and elbow up, both landing on the same point. A target 30 cm away (further than 12 + 14 = 26 cm) returns no solution. Those three outcomes are the whole IK story in miniature.

**The SO-101.** The same idea with the real geometry, read from the URDF. A **URDF** (Unified Robot Description Format) is an XML file listing links and joints; each joint has a parent and a child link, a fixed offset (`origin`: position and roll–pitch–yaw), a type (revolute, fixed…), a rotation axis and limits. FK walks the joints from the base to the gripper, multiplying the fixed offset and then the joint rotation (Rodrigues' formula turns an axis and an angle into a rotation matrix).

```python
# ---- 2. SO-101 forward kinematics from its URDF ----
import xml.etree.ElementTree as ET
def rpy_to_R(r, p, yw):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(yw), math.sin(yw)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx                                   # URDF convention: fixed-axis roll, pitch, yaw
def T_from(R, t):
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = t; return T
def axis_angle_R(axis, a):
    k = np.asarray(axis, float); k /= np.linalg.norm(k)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(a) * K + (1 - math.cos(a)) * K @ K   # Rodrigues

root = ET.parse('shared/robot/so101_new_calib.urdf').getroot()
joints = {j.find('child').get('link'): j for j in root.findall('joint')}
def chain(tip):                                     # joints from base to tip, in order
    out, link = [], tip
    while link in joints:
        j = joints[link]; out.append(j); link = j.find('parent').get('link')
    return out[::-1]
ARM = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll']
def fk_so101(q):                                    # q: dict joint name -> radians
    T = np.eye(4)
    for j in chain('gripper_frame_link'):
        o = j.find('origin')
        xyz = [float(v) for v in o.get('xyz').split()]
        rpy = [float(v) for v in o.get('rpy').split()]
        T = T @ T_from(rpy_to_R(*rpy), xyz)         # fixed offset parent -> joint frame
        if j.get('type') == 'revolute':
            ax = [float(v) for v in j.find('axis').get('xyz').split()]
            T = T @ T_from(axis_angle_R(ax, q.get(j.get('name'), 0.0)), [0, 0, 0])  # rotation about the joint axis
    return T
print('chain', [j.get('name') for j in chain('gripper_frame_link')])
p0 = fk_so101({})[:3, 3]
print('gripper at zero angles (m)', p0, 'distance from base', round(float(np.linalg.norm(p0)), 4))
q_test = dict(shoulder_pan=0.3, shoulder_lift=-0.5, elbow_flex=0.8, wrist_flex=0.4, wrist_roll=0.0)
print('gripper at q_test', fk_so101(q_test)[:3, 3])
```

Output: the chain is shoulder_pan → shoulder_lift → elbow_flex → wrist_flex → wrist_roll → the fixed gripper frame. With every joint at 0 the gripper frame sits 0.391 m in front of the base origin and 0.227 m above it; measured horizontally from the shoulder-pan axis (0.039 m in front of the base origin) that is about 0.35 m, the reach quoted in chapter 10. With (17°, −29°, 46°, 23°, 0°) it is at (0.276, −0.073, 0.085) m.

**Numerical IK.** For a real arm, IK is usually solved numerically. The **Jacobian** J is the matrix of partial derivatives of the gripper position with respect to each joint angle (3 rows for x, y, z; one column per joint): it says how the gripper moves for a small change of each joint. Repeat: compute the position error e; solve for the joint change that would remove it, Δq = Jᵀ(JJᵀ + λI)⁻¹ e (**damped least squares**; λ keeps the step finite near singularities); add it; clip to the joint limits; stop when the error is below a tolerance.

```python
# ---- 3. numerical IK: damped least squares on the position ----
def jac(q, eps=1e-6):
    p = fk_so101(q)[:3, 3]; J = np.zeros((3, len(ARM)))
    for i, n in enumerate(ARM):
        dq = dict(q); dq[n] = dq.get(n, 0.0) + eps
        J[:, i] = (fk_so101(dq)[:3, 3] - p) / eps
    return J
lims = {j.get('name'): (float(j.find('limit').get('lower')), float(j.find('limit').get('upper'))) for j in root.findall('joint') if j.find('limit') is not None}
def ik_so101(target, q=None, lam=0.01, iters=200):
    q = dict(q or {n: 0.0 for n in ARM})
    for it in range(iters):
        e = target - fk_so101(q)[:3, 3]
        if np.linalg.norm(e) < 1e-4:
            break
        J = jac(q)
        dq = J.T @ np.linalg.solve(J @ J.T + lam * np.eye(3), e)    # damped least squares
        for i, n in enumerate(ARM):
            q[n] = float(np.clip(q[n] + dq[i], *lims[n]))           # respect joint limits
    return q, float(np.linalg.norm(e)), it
target = fk_so101(q_test)[:3, 3]
q_sol, err, it = ik_so101(target)
print('ik err (m)', round(err, 6), 'iters', it, 'solution (deg)', {k: round(math.degrees(v), 1) for k, v in q_sol.items()})
print('true q (deg)', {k: round(math.degrees(v), 1) for k, v in q_test.items()})
far = p0 * 3
print('unreachable target: err', round(ik_so101(far)[1], 4))
print('cond(J) at zero', round(float(np.linalg.cond(jac({n: 0.0 for n in ARM}))), 1))
```

Output, and what each line teaches:

- The target was the FK of (17.2°, −28.6°, 45.8°, 22.9°, 0°). IK reached it to 0.08 mm in 10 iterations, but with (17.2°, −25.3°, 39.6°, 30.0°, 0.4°): **a different joint configuration, same gripper position.** Five joints, three position coordinates: the arm is redundant for position-only targets, and the solver returns whichever solution is nearest its starting point. This is why a demonstration dataset can contain two very different joint trajectories for the same task (a source of the multimodality of P1.14), and why "the IK solution" is not a well-defined label unless orientation is also constrained.
- An unreachable target (three times the zero-pose distance) ends with an error of 1.12 m: the solver stretches the arm towards it and stops. Always check the final error; IK libraries return *something* even when there is no solution.
- **Singularities** are configurations where the Jacobian loses rank (the arm fully stretched, or two joint axes aligned): some directions of gripper motion become impossible and the plain inverse blows up. The condition number of J (5.7 at the zero pose here) grows towards infinity near one. Damping (λ) is the standard defence.

In this repository IK is used in G16-lite's repair (hover above the target's centroid, IK, goal joints) and in the LEGO scripted pick. Policies trained by imitation do not use IK at all: they output joint angles directly, which is why their action space is joints (10a.6).

## 10a.3 Actuators: what a servo actually does

**A servo** is a motor, a gearbox, a position sensor (an **encoder**) and a small controller in one package. The STS3215 on the SO-101 has a 12-bit magnetic encoder (4 096 counts per turn, about 0.088°) and runs its own position control loop internally (chapter 10.1).

**Position, velocity or torque control.** What you send to an actuator decides what a policy can express:

- **Position control** (the SO-101): you send a goal angle; the servo's internal controller works out the motor effort. Simple and safe to learn on; but you cannot ask for a specific force, and contact is invisible except through indirect signals (position error, motor current, the gripper aperture). Most low-cost arms and most imitation-learning work use it.
- **Velocity control**: you send a goal speed. Common on mobile bases.
- **Torque (or current) control**: you send the effort directly; the controller loop runs on your computer. Necessary for compliant contact, force control and most reinforcement learning on legged robots; it needs torque sensing or accurate current-to-torque models and a fast, reliable control loop, because a bug sends raw force to the motor.

**Gearboxes** multiply torque and divide speed; they also add **backlash** (a small dead zone when the direction reverses) and friction, which is why a commanded angle and a measured angle differ by a fraction of a degree even at rest. **Torque limits** and **thermal limits** cap what the servo can do: under a sustained load (a long data-collection run, an arm held against an obstacle) a hobby servo heats, and its controller reduces torque or shuts down. That is why the harness logs servo temperature and pauses at a threshold.

**Calibration** of an arm means mapping each encoder's raw counts to a joint angle with a known zero and a known direction, and recording each joint's safe range. A calibration file from another arm, or from the same arm after a servo swap, produces a robot whose every joint angle is offset: policies trained on one calibration misbehave on the other without any error message (chapter 10.1).

## 10a.4 Control loops: P, PD, PID, and why gains and delay matter

**A feedback controller** compares where a joint is with where it should be and pushes towards the target. The simplest is **proportional (P)** control: effort = Kp · (target − measured). **PD** adds a damping term that opposes velocity: − Kd · velocity. **PID** adds an integral term that accumulates past error and removes a steady offset (caused by gravity or friction). The servo in each SO-101 joint runs such a loop thousands of times per second; a policy runs at 30 Hz on top of it.

A one-joint simulation shows every behaviour an interviewer asks about. The joint has inertia and friction; the controller applies a torque; the simulation steps physics every millisecond:

```python
# ---- 4. one joint under PD control ----
def simulate(kp, kd, delay_steps=0, target=1.0, dt=0.001, T=2.0, inertia=0.01, friction=0.02, torque_limit=None):
    """Joint angle q (rad), velocity v. Torque = kp*(target - q_measured) - kd*v_measured. Returns q history."""
    q, v = 0.0, 0.0
    hist = []
    buf = [(0.0, 0.0)] * (delay_steps + 1)      # measurements arrive delay_steps late
    for _ in range(int(T / dt)):
        buf.append((q, v)); qm, vm = buf.pop(0)
        tau = kp * (target - qm) - kd * vm
        if torque_limit is not None:
            tau = max(-torque_limit, min(torque_limit, tau))
        a = (tau - friction * v) / inertia
        v += a * dt; q += v * dt
        hist.append(q)
        if abs(q) > 1e3: break
    return np.array(hist)

def describe(h, target=1.0, dt=0.001):
    final = h[-1]
    over = max(0.0, h.max() - target)
    within = np.where(np.abs(h - target) > 0.02)[0]
    settle = (within[-1] + 1) * dt if len(within) else 0.0
    return f'final {final:.3f}  overshoot {over:.3f}  settles(2%) {settle:.2f} s'

print('P only, kp=1          ', describe(simulate(1.0, 0.0)))
print('PD kp=1 kd=0.2        ', describe(simulate(1.0, 0.2)))
print('PD kp=1 kd=0.05       ', describe(simulate(1.0, 0.05)))
h = simulate(-1.0, 0.2); print('wrong sign kp=-1      ', f'q after {len(h)*0.001:.2f} s = {h[-1]:.1f} rad (runs away)')
print('PD kp=1 kd=0.2 delay 30 ms', describe(simulate(1.0, 0.2, delay_steps=30)))
print('PD kp=5 kd=0.2 delay 30 ms', describe(simulate(5.0, 0.2, delay_steps=30)))
print('PD kp=1 kd=0.2 torque limit 0.1', describe(simulate(1.0, 0.2, torque_limit=0.1)))
```

The run of 2026-10-24, target 1 rad:

| Controller | Result | Lesson |
|---|---|---|
| P only, Kp = 1 | overshoot 0.74 rad, still oscillating after 2 s | P alone acts like a spring: it overshoots and rings |
| PD, Kp = 1, Kd = 0.2 | no overshoot, settles in 0.69 s | damping removes the oscillation |
| PD, Kp = 1, Kd = 0.05 | overshoot 0.31, settles in 1.10 s | too little damping |
| **Kp = −1 (wrong sign)** | runs away: −1 000 rad after 1.8 s | a sign error pushes *away* from the target, to the joint limit |
| PD, Kp = 1, Kd = 0.2, 30 ms measurement delay | settles in 0.67 s | a gentle controller tolerates delay |
| PD, Kp = 5, Kd = 0.2, 30 ms delay | overshoot 0.80, not settled after 2 s | **a stiff controller with delay oscillates**: it reacts to where the joint *was* |
| PD with a torque limit of 0.1 | settles in 0.88 s | saturation slows the response but here stays stable |

Three of these rows are the interview answers:

- **Wrong sign:** the symptom is a joint that drives hard to a limit as soon as the loop starts. On a real arm it strips gears or trips a current limit. First check in a simulation-debugging round: plot command against measurement for one joint and see whether the error grows.
- **Delay plus stiffness:** every learned policy adds delay (inference time, camera latency, the bus). The stiffer the loop around a delayed signal, the more it oscillates. This is the control-theory version of the chunk-boundary problem: a policy acting on a 238 ms-old observation is a delayed controller.
- **Integral term and steady-state error:** a P or PD controller holding an arm against gravity settles slightly below the target (the error needed to produce the holding torque); the I term removes it, at the cost of windup (accumulated error that causes overshoot after a saturation).

## 10a.5 Two clocks: the control loop and the policy

A learned robot runs at least two loops at different rates:

- **The servo loop** inside each actuator: kHz, position control, not yours to change beyond its gains and limits.
- **The control loop** on the computer: here 30 Hz (33.3 ms per tick, limited by the serial bus); reads joint positions and camera frames, writes goal positions.
- **The policy**: produces actions at its own pace. A chunked policy produces H actions per inference; the control loop consumes one per tick.

On the Orin, SmolVLA through vla.cpp takes 238 ms per chunk: about 7 control ticks. Two ways to deal with it (chapter 11.4): **synchronous** (the control loop waits; the arm pauses at every chunk) or **asynchronous** (the loop keeps executing the previous chunk while the next is computed; when the new chunk arrives, its first ~7 actions describe moments that have already passed). The asynchronous jump at chunk boundaries, and the fixes (prefix overwrite, real-time chunking), are the direct consequence of two clocks with a delay between them.

Other timing facts that matter: USB cameras deliver frames with independent, jittering timestamps (a few milliseconds between two cameras); a frame is already 30–100 ms old when the policy sees it (exposure, transfer, decoding); and every extra register read on the servo bus (temperature, current) costs bus time and lowers the achievable control rate.

## 10a.6 Action and observation spaces

What a policy outputs is a design choice with consequences:

| Action space | What it is | Pros | Cons |
|---|---|---|---|
| **Absolute joint positions** (this repository, ACT, SmolVLA on SO-101) | the goal angle of each joint | matches the servo interface; no IK; easy to record from a leader arm | embodiment-specific; redundancy makes demonstrations multimodal |
| **Delta joint positions** | change from the current angle | smaller numbers, invariant to start pose | drift accumulates; needs care at chunk boundaries |
| **End-effector pose (absolute or delta)** | where the gripper should be, plus orientation | transfers across arms; matches human video and language ("move 2 cm left") | needs IK at run time, with its no-solution and multiple-solution cases; orientation representation issues (10a.1) |
| **Gripper** | one number: aperture or open/close | | a binary gripper cannot express a gentle grasp; the SO-101's is continuous |

**Observations** are typically the camera images plus **proprioception** (the robot's own joint positions, sometimes velocities and gripper state). Proprioception is cheap and precise; it is also a shortcut a policy can over-rely on (chapter 8's pose-predicts-outcome finding is a version of this).

**Teleoperation** with a leader arm records the leader's angles as the *action* and the follower's measured angles as the *state*. The two differ by the follower's tracking lag and by any calibration mismatch between the arms; a leader that leads the follower by more than a few degrees on a fraction of ticks (chapter 10's LEGO session) shows up as a systematic action-state offset in the data.

## 10a.7 Cameras: the pinhole model, calibration, and why the oblique view distorts

**The pinhole model.** A camera maps a 3-D point in its own frame (x right, y down, z forward, in metres) to a pixel (u, v) by dividing by depth and scaling: u = fx · x/z + cx, v = fy · y/z + cy. The four numbers fx, fy (focal lengths in pixels) and cx, cy (the principal point, near the image centre) form the **intrinsic matrix** K. **Extrinsics** are the camera's pose in the world: a rotation R and translation t that take world points into the camera frame. **Distortion** (straight lines bending near the image edge) is a third, separate set of parameters.

```python
# ---- 5. a pinhole camera ----
fx = fy = 600.0; cx, cy = 320.0, 240.0            # intrinsics for a 640x480 image, in pixels
K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
def project(P_cam):                                # point in camera coordinates (metres, z forward)
    p = K @ P_cam
    return p[:2] / p[2]
print('point 0.1 m right, 0.5 m ahead ->', project(np.array([0.1, 0.0, 0.5])))
print('same point 1.0 m ahead        ->', project(np.array([0.1, 0.0, 1.0])))
# extrinsics: where the camera is and where it looks (world: x right, y away from the camera, z up)
cam_pos = np.array([0.0, -0.30, 0.45])             # 30 cm behind the table origin, 45 cm above it
look_at = np.array([0.0, 0.15, 0.0])               # aims at a point 15 cm into the work area
z_c = look_at - cam_pos; z_c /= np.linalg.norm(z_c)            # camera z: forward
x_c = np.cross(z_c, [0, 0, 1.0]); x_c /= np.linalg.norm(x_c)   # camera x: right in the image
y_c = np.cross(z_c, x_c)                                       # camera y: down in the image
R = np.stack([x_c, y_c, z_c])                      # rows: camera axes in world coordinates
t = -R @ cam_pos
def world_to_pixel(Pw):
    return project(R @ Pw + t)
print('aim point ->', world_to_pixel(look_at).round(1), ' table origin ->', world_to_pixel(np.zeros(3)).round(1))
print('1 cm along y, near (y=0) vs far (y=0.30):',
      np.linalg.norm(world_to_pixel(np.array([0, 0.01, 0])) - world_to_pixel(np.zeros(3))).round(2),
      np.linalg.norm(world_to_pixel(np.array([0, 0.31, 0])) - world_to_pixel(np.array([0, 0.30, 0]))).round(2))
# homography from four table points: pixel <-> table plane
import itertools
tab = np.array([[-0.1, 0.0], [0.1, 0.0], [0.1, 0.2], [-0.1, 0.2]])
pix = np.array([world_to_pixel(np.array([x, y, 0.0])) for x, y in tab])
A = []
for (X, Y), (u, v) in zip(tab, pix):
    A.append([-X, -Y, -1, 0, 0, 0, u * X, u * Y, u])
    A.append([0, 0, 0, -X, -Y, -1, v * X, v * Y, v])
Hm = np.linalg.svd(np.array(A))[2][-1].reshape(3, 3)       # direct linear transform
q = Hm @ np.array([0.05, 0.1, 1.0]); print('homography maps (0.05, 0.10) ->', (q[:2] / q[2]).round(2), 'true', world_to_pixel(np.array([0.05, 0.1, 0.0])).round(2))
```

Output, and what it shows:

- A point 10 cm to the right at 0.5 m lands at pixel (440, 240); the same point at 1.0 m lands at (380, 240). **Image size scales with 1/depth.** A single camera cannot tell a small near object from a large far one without extra information.
- A camera 45 cm above and 30 cm behind the table, aimed at a point 15 cm into the work area: the aim point lands at the image centre (320, 240), the table origin at (320, 360). **One centimetre on the table is 9.5 pixels near the camera and 4.9 pixels 30 cm further away.** That is the oblique-view distortion of chapter 10 and the LEGO calibration (parts "look larger than their footprint", growth different along and across the view): measurements in pixels are not measurements on the table until you correct for it.
- **A homography** is a 3 × 3 matrix that maps one plane to another under perspective, here the table plane to the image. Four known points on the table and their pixels determine it (the direct linear transform above, solved with an SVD); it then maps any table point to its pixel and back, exactly, as long as the point is on the plane. The LEGO cell uses the 1 cm grid of the cutting mat for this. Objects with height are *not* on the plane, which is one source of the residual growth errors.

**Calibration in practice.** Intrinsics and distortion: photograph a checkerboard (or a ChArUco board) from 15–30 angles and run OpenCV's `calibrateCamera`; check the reprojection error (well under a pixel is good). Extrinsics: a board at a known position, or a homography from known table points, or the hand–eye procedure for a wrist camera (move the arm to known poses and solve for the camera's fixed offset on the gripper). Re-check after any knock to a mount: a camera that moved by a degree silently changes every learned policy's input distribution.

**Exposure, white balance and light.** Automatic exposure and white balance change the image when the scene changes (a hand enters, a cloud passes). Lock them for data collection and log the setting; this repository closes the curtains and locks white balance for that reason.

## 10a.8 Simulation and the reality gap

A **simulator** (MuJoCo, Isaac Sim / Isaac Lab, PyBullet, Drake) computes physics from a model of the robot (often the same URDF) and the scene. It gives unlimited, resettable, perfectly labelled episodes and ground-truth state (exact object poses), which no real rig can. The **sim-to-real gap** is every way the simulator differs from the real robot: contact and friction models, actuator dynamics (backlash, delay, saturation), camera appearance, latency. Standard tools against it: **domain randomisation** (vary masses, friction, delays, textures and lighting in simulation so the real world looks like one more variation), **system identification** (measure the real robot's parameters and put them in the simulator), and **fine-tuning on real data**.

When an interviewer asks about it, the answer that reads as experience is specific: which parameters you would randomise and over what ranges, how you would check that the policy is not overfitting to the randomisation, and which real-world number you would compare first. For this rig, a simulated SO-101 would mainly serve as an **oracle** for world-model questions (true object state, unlimited counterfactual rollouts; chapter 12b's Planning Limits uses the real simulator exactly this way), not as a training source, because the cheap servos' backlash and delay are hard to model.

### The reality gap, decomposed (Aljalbout et al., "The Reality Gap in Robotics", Annual Review, arXiv 2510.20808)

This survey is the reference to cite when the question is "how would you go from simulation to the real robot". Its useful contributions are a vocabulary and a checklist, not experiments.

**Two gaps, not one.** The **reality gap** is the difference between the simulated and real worlds themselves: a *dynamics gap* (the transition model) and a *perception gap* (the observation model). The **performance gap** is the difference in a given policy's return between simulation and reality. The goal is to close the second, not the first: a policy that is robust to the differences transfers even when the simulator is wrong. The same distinction is chapter 12c's for learned world models: prediction fidelity is the reality gap, planning success is the performance gap, and the two are only loosely related.

**Where gaps come from**, with the symptom the survey attaches to each, mapped to this rig:

| Source | Example in the survey | On the SO-101 |
|---|---|---|
| Dynamics: modelling | rigid bodies, contact simplified, chaotic and stochastic effects | grasp slip and contact are what a simulated SO-101 would get wrong first |
| Dynamics: parameters, integrators | wrong friction or mass; integration error accumulating over long tasks | servo friction and backlash vary per joint and with wear |
| Dynamics: unmodelled effects | wear, fatigue, thermal changes | STS3215 heating on long runs (the harness's thermal pause) |
| Perception: sensor model, noise | idealised pinhole cameras; Gaussian noise where real noise is correlated and state-dependent | USB webcam jitter, auto-exposure, the oblique-view scale change (10a.7) |
| Actuation: actuator models | first-order motors vs real dead zones, slew limits, hysteresis | the servo's internal PID and gearbox: commanded ≠ measured by a fraction of a degree |
| Actuation: low-level control | hidden vendor filters, rate limits, anti-windup | the STS3215 firmware loop you cannot see or change beyond its gains |
| System design: communication | delays and packet loss, absent in simulation | serial-bus time limits, Wi-Fi in remote inference (14b.10) |
| System design: safety mechanisms | virtual walls and clamps that exist only on the real robot | the System 0 joint clamp, the 8°/tick limit, the gripper floor: a trained policy never saw them |
| System design: POMDP | privileged rewards or resets impossible in reality | harness resets that avoid hard states (SEAR), scripted-pick resets in the data engine |
| System design: implementation | different control frequencies or discretisation in sim and real | training at one fps and deploying at another; interpolation multipliers |

The last three rows apply **even without a simulator**: training on teleoperation and deploying through a harness creates the same gaps between the data's world and the deployment's world. The survey's term for the fix is **software-stack alignment**: the policy should meet the same clamps, filters and control rate in training data as at deployment.

**The recipe** the survey proposes: design a simulation with every variable the task needs; reduce each gap as far as possible (system identification, learned residual models, real-to-sim reconstruction, better low-level controllers, modalities with smaller gaps such as depth or keypoints instead of RGB, simpler or slower dynamics); train methods that overcome the rest (domain randomisation, adaptation, privileged-information teachers distilled into students, policy regularisation such as smoothness penalties); train in simulation; evaluate in reality; adjust the simulator from the real results and repeat.

**How to measure it** (its section 5):

- **Sim-to-real correlation coefficient (SRCC):** the Pearson correlation, across several policies, between their success in simulation and in reality. A simulator with high average success but low SRCC cannot be used to choose between policies. This is the right statistic for *any* offline evaluator, not just simulators: see 12c.6a.
- **Offline replay error:** replay the real robot's recorded actions open-loop in the simulator and average the squared state difference. For a learned world model this is exactly the open-loop rollout error against copy-last of G0.
- **Visual fidelity metrics** (FID, KID, SSIM, PSNR) for the perception gap; and for transfer itself, success rate, cumulative reward and task-specific metrics, with the survey's warning that a single success rate hides *where* and *why* transfer failed.

Two open problems it raises connect to Part II: "wrong models, better controllers" (a model accurate near high-return states beats a uniformly accurate one, the decision-relevant-error argument of 12c.4), and video and world models as learned simulators with a potentially smaller reality gap but compounding errors and high data needs (12d).

## 10a.9 Safety on a real arm

Learned policies are unpredictable by construction; the safety layer is not learned. On this rig: a joint-limit clamp in software, never wider than the URDF limits (the harness's "System 0" layer and the `STS3215SafetyGuard` of chapter 10.1); a cap on the change of goal per tick (8° here, architecture v1.4 §7) so a bad action cannot snap the arm; a gripper floor; torque and temperature monitoring; a stillness check before any judgement; soft objects only for blocking tests; a human with a kill switch. In an interview, name the layer the policy cannot override, and say what triggers it.

## 10a.10 Questions to be able to answer in one minute each

1. What is the difference between forward and inverse kinematics, and when does IK have zero, one, several or infinitely many solutions?
2. Your SO-101 has five arm joints and the task only specifies the gripper position. What does that imply for the demonstrations, and for a policy trained on them?
3. Position control vs torque control: what can a position-controlled arm not do, and what contact signals does it still have?
4. A joint drives straight to its limit when the controller starts. First hypothesis, first check?
5. Why does a stiff controller oscillate when the measurement is delayed, and how is that related to a policy's inference time?
6. The control loop runs at 30 Hz and the policy takes 238 ms per chunk. What happens in synchronous and asynchronous execution?
7. Joint-space or end-effector actions for a dataset that mixes two different arms? Why?
8. Why does 1 cm on the table cover more pixels near the camera than far from it, and how does a homography help?
9. What would you randomise to train a policy in simulation for this arm, and what would you measure first on the real one?
10. Name the safety layer a learned policy cannot override on your rig.

## 10a.11 Exercises

1. Run every code block of this chapter from the repository root. Then change the two-link lengths to 0.10 and 0.10 m and predict before running which targets lose their solutions.
2. Extend `ik_so101` to also match a gripper orientation (use the rotation part of the FK transform and a 3-element orientation error); check that the redundancy example now returns the original joint angles or say why not.
3. In the PD simulation, find the largest Kp that settles within 1 s with a 30 ms delay and Kd = 0.2. Then repeat with a 240 ms delay (a policy's latency) and write one paragraph on what that means for a policy that outputs goals at 4 Hz.
4. Measure your top camera's intrinsics with a printed checkerboard and OpenCV; report the reprojection error; compare the pixels-per-centimetre near and far on the mat with the prediction of 10a.7.
