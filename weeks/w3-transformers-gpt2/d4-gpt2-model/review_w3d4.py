"""REVIEW. A GPT wrapper meant to tie input and output embeddings. The parameter count is 38.6M
higher than the paper's, and the checkpoint is 31% bigger. Find the bug, fix, replace:
BUG:
"""
import torch.nn as nn


class TiedLM(nn.Module):
    def __init__(self, V: int, D: int):
        super().__init__()
        self.wte = nn.Embedding(V, D)
        self.body = nn.Linear(D, D)
        self.lm_head = nn.Linear(D, V, bias=False)
        self.lm_head.weight = nn.Parameter(self.wte.weight.clone())     # tie

    def forward(self, idx):
        return self.lm_head(self.body(self.wte(idx)))
