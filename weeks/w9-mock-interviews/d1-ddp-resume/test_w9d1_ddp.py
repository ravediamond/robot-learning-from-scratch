import socket
import tempfile
from pathlib import Path

import torch
import torch.multiprocessing as mp

import w9d1_ddp as m


def _port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def test_ddp_equals_single_process():
    out = str(Path(tempfile.mkdtemp()) / "ddp.pt")
    mp.spawn(m.ddp_worker, args=(2, _port(), 30, 32, 0.05, out), nprocs=2, join=True)
    ddp_sd = torch.load(out, weights_only=True)
    ref = m.single_process_reference(30, 32, 0.05)
    for k in ref:
        assert torch.allclose(ddp_sd[k], ref[k], atol=1e-5), f"{k} differs: gradients were not averaged like a full batch"


def test_exact_resume(tmp_path):
    data = m.make_data()

    def fresh():
        model = m.make_model(0)
        opt = torch.optim.Adam(model.parameters(), lr=1e-2)
        sched = torch.optim.lr_scheduler.StepLR(opt, step_size=7, gamma=0.5)
        return model, opt, sched, torch.Generator().manual_seed(3)

    a = fresh()
    m.train_steps(*a, 0, 20, data)
    b = fresh()
    m.train_steps(*b, 0, 10, data)
    path = str(tmp_path / "ck.pt")
    m.save_checkpoint(path, b[0], b[1], b[2], 10, b[3])
    c = fresh()                                                    # "a new process after a crash"
    start = m.load_checkpoint(path, c[0], c[1], c[2], c[3])
    assert start == 10
    m.train_steps(*c, start, 20, data)
    for (k, va), vc in zip(a[0].state_dict().items(), c[0].state_dict().values()):
        assert torch.equal(va, vc), f"{k}: resumed run diverged from the straight run"
