"""Provided: synthetic SO-101 episodes with KNOWN defects, for QA labs.

An episode: {"id", "station", "operator", "session", "state" (T, 6), "action" (T, 6), "frame_hash" (T,) ints}."""
import numpy as np

LIMITS = np.array([[-1.9, 1.9], [-1.75, 1.75], [-1.7, 1.7], [-1.7, 1.7], [-2.7, 2.7], [0.0, 1.5]])


def _episode(rng: np.random.Generator, eid: int, T: int) -> dict:
    t = np.linspace(0, 1, T)[:, None]
    start = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.6]])                    # rest pose, gripper half open
    goal = rng.uniform(-0.8, 0.8, size=(1, 6))
    goal[0, 5] = rng.uniform(0.2, 1.2)
    state = start + (goal - start) * (1 - np.cos(np.pi * t)) / 2 + 0.01 * rng.standard_normal((T, 6))
    action = np.vstack([state[2:], np.repeat(state[-1:], 2, axis=0)])
    return {"id": eid, "station": f"st{eid % 4}", "operator": f"op{eid % 3}", "session": eid // 10,
            "state": state, "action": action, "frame_hash": rng.integers(0, 2**31, size=T)}


def make_fleet(n: int = 200, seed: int = 0) -> tuple[list[dict], dict]:
    """Episodes plus the ground truth of injected defects: {"frozen", "dead", "limits", "long", "short", "dup"}: sets of ids."""
    rng = np.random.default_rng(seed)
    eps = [_episode(rng, i, int(rng.integers(240, 420))) for i in range(n)]
    truth = {"frozen": {3, 77}, "dead": {12, 140}, "limits": {45}, "long": {60}, "short": {99}, "dup": {150}}
    for i in truth["frozen"]:
        eps[i]["frame_hash"][50:350] = eps[i]["frame_hash"][50]          # 10 s of the same frame
    for i in truth["dead"]:
        eps[i]["state"][:, 2] = 0.37                                       # elbow encoder stuck
    eps[45]["state"][100:200, 1] = LIMITS[1, 1] + 0.05                     # driven into the limit
    eps[60] = _episode(rng, 60, 6000)                                      # forgot to stop recording
    eps[99] = _episode(rng, 99, 45)                                        # mis-click
    eps[150] = {**eps[20], "id": 150, "frame_hash": eps[20]["frame_hash"].copy(),
                "state": eps[20]["state"].copy(), "action": eps[20]["action"].copy()}  # uploaded twice
    return eps, truth
