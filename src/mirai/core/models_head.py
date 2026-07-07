"""
models_heads.py — shared definitions for head experiment models.
Import from here so torch.load can always find the classes.
"""

import torch
import torch.nn as nn


class FixedMLP(nn.Module):
    def __init__(self, d_in: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, 32), nn.LeakyReLU(0.01),
            nn.Linear(32, 32),   nn.LeakyReLU(0.01),
            nn.Linear(32, 32),   nn.LeakyReLU(0.01),
            nn.Linear(32, 32),   nn.LeakyReLU(0.01),
            nn.Linear(32, 32),   nn.LeakyReLU(0.01),
            nn.Linear(32, 32),   nn.LeakyReLU(0.01),
            nn.Linear(32, 32),   nn.LeakyReLU(0.01),
            nn.Linear(32, 32),   nn.LeakyReLU(0.01),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x)


class HeadProjection(nn.Module):
    def __init__(self, d_in: int, head_dim: int):
        super().__init__()
        self.W   = nn.Linear(d_in, head_dim)
        self.act = nn.Tanh()

    def forward(self, x):
        return self.act(self.W(x))


class TransformerWithNHeads(nn.Module):
    def __init__(self, n_heads: int, d_in: int = 6, head_dim: int = 4):
        super().__init__()
        self.heads = nn.ModuleList([
            HeadProjection(d_in, head_dim)
            for _ in range(n_heads)
        ])
        self.mlp = FixedMLP(n_heads * head_dim)

    def forward(self, x):
        head_outs = [h(x) for h in self.heads]
        combined  = torch.cat(head_outs, dim=1)
        return self.mlp(combined)