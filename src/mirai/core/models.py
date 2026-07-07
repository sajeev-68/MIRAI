"""
models.py

Shared model definitions — import from here in both training scripts
and app.py so torch.load can always find the classes.
"""

import torch
import torch.nn as nn


class ResBlock(nn.Module):
    """
    Single residual block.
    output = LeakyReLU(Wx + b) + x

    Skip connection guarantees gradient = 1.0 through skip path.
    Each block learns one correction to the previous approximation —
    structurally analogous to one Taylor term.
    """
    def __init__(self, width: int):
        super().__init__()
        self.linear = nn.Linear(width, width)
        self.act    = nn.LeakyReLU(0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.linear(x)) + x


def make_resnet(n_blocks: int, width: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(1, width),
        nn.LeakyReLU(0.01),
        *[ResBlock(width) for _ in range(n_blocks)],
        nn.Linear(width, 1),
    )


def make_wide_shallow(width: int = 62) -> nn.Module:
    return nn.Sequential(
        nn.Linear(1,     width), nn.LeakyReLU(0.01),
        nn.Linear(width, width), nn.LeakyReLU(0.01),
        nn.Linear(width, 1),
    )


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())