import math

import torch


def causal_attention(q, k, v):
    T = q.shape[-2]
    s = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
    s = s.masked_fill(torch.triu(torch.ones(T, T, dtype=torch.bool), 1), float("-inf"))
    return torch.softmax(s, -1) @ v


def multi_head_self_attention(x, Wqkv, Wo, h):
    B, T, D = x.shape
    q, k, v = (x @ Wqkv.T).split(D, -1)
    q, k, v = (t.view(B, T, h, D // h).transpose(1, 2) for t in (q, k, v))
    return causal_attention(q, k, v).transpose(1, 2).reshape(B, T, D) @ Wo.T
