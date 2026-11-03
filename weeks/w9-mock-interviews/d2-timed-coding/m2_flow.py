import torch


def flow_matching_loss(v_net, x1, cond, generator):
    x0 = torch.randn(x1.shape, generator=generator)
    t = torch.rand(x1.shape[0], generator=generator)
    tt = t.view(-1, *([1] * (x1.dim() - 1)))
    xt = (1 - tt) * x0 + tt * x1
    return ((v_net(xt, t, cond) - (x1 - x0)) ** 2).mean()


@torch.no_grad()
def sample(v_net, cond, shape, steps, generator):
    x = torch.randn(shape, generator=generator)
    for i in range(steps):
        x = x + v_net(x, torch.full((shape[0],), i / steps), cond) / steps
    return x
