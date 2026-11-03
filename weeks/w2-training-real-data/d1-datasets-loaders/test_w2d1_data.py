import torch

import w2d1_data as m


def _eps():
    out = []
    for e, T in enumerate([5, 3, 7]):
        obs = torch.arange(T, dtype=torch.float32)[:, None].repeat(1, 4) + 100 * e
        act = torch.arange(T, dtype=torch.float32)[:, None].repeat(1, 2) + 100 * e
        out.append({"obs": obs, "action": act})
    return out


def test_len_and_mapping():
    ds = m.EpisodeWindowDataset(_eps(), H=4)
    assert len(ds) == 15
    s = ds[5]                                     # first frame of episode 1
    assert s["episode"] == 1 and s["obs"][0].item() == 100.0


def test_window_and_padding():
    ds = m.EpisodeWindowDataset(_eps(), H=4)
    s = ds[3]                                     # episode 0, t=3, T=5
    assert s["actions"].shape == (4, 2) and s["action_mask"].dtype == torch.bool
    assert s["actions"][:, 0].tolist() == [3.0, 4.0, 4.0, 4.0], "pad by repeating the last action"
    assert s["action_mask"].tolist() == [True, True, False, False]
    last = ds[14]                                 # last frame of episode 2
    assert last["actions"][:, 0].tolist() == [206.0] * 4 and last["action_mask"].sum() == 1


def test_never_crosses_episodes():
    ds = m.EpisodeWindowDataset(_eps(), H=6)
    for i in range(len(ds)):
        s = ds[i]
        assert (s["actions"] // 100 == s["episode"]).all(), f"sample {i} reads another episode"


def test_collate():
    ds = m.EpisodeWindowDataset(_eps(), H=4)
    b = m.collate_windows([ds[0], ds[6], ds[9]])
    assert b["obs"].shape == (3, 4) and b["actions"].shape == (3, 4, 2) and b["action_mask"].shape == (3, 4)
    assert b["episode"].tolist() == [0, 1, 2] and b["episode"].dtype == torch.long
    import inspect
    assert "default_collate" not in inspect.getsource(m.collate_windows)


def test_loader_reproducible():
    ds = m.EpisodeWindowDataset(_eps(), H=4)
    a = [b["obs"][:, 0].tolist() for b in m.make_loader(ds, 4, seed=1)]
    b = [b["obs"][:, 0].tolist() for b in m.make_loader(ds, 4, seed=1)]
    c = [b["obs"][:, 0].tolist() for b in m.make_loader(ds, 4, seed=2)]
    assert a == b and a != c
    assert sum(len(x) for x in a) == 15


def test_task_balanced_weights():
    tasks = torch.tensor([0] * 90 + [1] * 10)
    w = m.task_balanced_weights(tasks)
    assert w.shape == (100,) and torch.isclose(w.sum(), torch.tensor(1.0))
    assert torch.isclose(w[tasks == 0].sum(), w[tasks == 1].sum())
