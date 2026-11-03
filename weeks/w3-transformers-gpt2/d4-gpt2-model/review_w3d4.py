"""REVIEW. SOLUTION.

BUG: clone() copies the data into a NEW Parameter: two independent matrices that start equal and drift
apart after the first update. Tie by assigning the same Parameter: self.lm_head.weight = self.wte.weight.
"""
import torch.nn as nn


class TiedLM(nn.Module):
    def __init__(self, V, D):
        super().__init__()
        self.wte = nn.Embedding(V, D)
        self.body = nn.Linear(D, D)
        self.lm_head = nn.Linear(D, V, bias=False)
        self.lm_head.weight = self.wte.weight

    def forward(self, idx):
        return self.lm_head(self.body(self.wte(idx)))
