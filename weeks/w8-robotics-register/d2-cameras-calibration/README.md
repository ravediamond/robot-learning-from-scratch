# W8 D2: Cameras, projection, and hand-eye calibration

Book: 10a.7 (the pinhole model, calibration, why the oblique view distorts).

## Pinhole camera
```
camera frame: x right, y down, z forward (OpenCV convention)
p_cam = T_cam_world @ [p_world, 1]                  extrinsics (4x4)
u = fx * X / Z + cx,   v = fy * Y / Z + cy           intrinsics K = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
back-projection with depth Z:  X = (u − cx) Z / fx,  Y = (v − cy) Z / fy,  then T_world_cam @ p_cam
```
Divide by Z (depth along the optical axis), not by the distance to the point.

## Hand-eye as a rigid fit
Move the gripper to N poses; you know each tip position in the robot frame (FK, W8 D1) and you measure
it in the camera frame (a marker + depth). Find R, t with `B ≈ R A + t` (A robot, B camera):
**Kabsch**: centre both sets, H = A_cᵀ B_c, SVD H = U S Vᵀ, R = V diag(1, 1, d) Uᵀ with
d = sign(det(V Uᵀ)) (prevents returning a reflection), t = mean(B) − R mean(A).

## Today's build (`w8d2_camera.py`)
`intrinsics`, `project`, `backproject`, `kabsch`, `rigid_rmse`.

Run: `uv run pytest weeks/w8-robotics-register/d2-cameras-calibration`
