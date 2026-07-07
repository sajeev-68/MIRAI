"""
train_heads_dim.py — standalone, no external imports except torch.

Tests: minimum heads = intrinsic dimensionality of data.

Data lives on a 3D manifold (3 latent angles, 6D ambient).
Theory predicts saturation at 3 heads.

Saturation metrics (loss alone is not enough):
    effective_rank  — how many independent dims do heads use?
    head_corr       — do extra heads become redundant/correlated?

Run:
    python train_heads_dim.py
"""

import torch
import torch.nn as nn
import math
import numpy as np
from pathlib import Path

torch.manual_seed(42)
Path("checkpoints").mkdir(exist_ok=True)


# ------------------------------------------------------------------ #
#  Model definitions (inline so torch.save/load works)               #
# ------------------------------------------------------------------ #

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


def count_params(model):
    return sum(p.numel() for p in model.parameters())


# ------------------------------------------------------------------ #
#  Data — 3D manifold embedded in 6D                                 #
# ------------------------------------------------------------------ #

def generate_data(n: int):
    t1 = torch.rand(n) * 2 * math.pi
    t2 = torch.rand(n) * 2 * math.pi
    t3 = torch.rand(n) * 2 * math.pi

    X = torch.stack([
        torch.cos(t1), torch.sin(t1),
        torch.cos(t2), torch.sin(t2),
        torch.cos(t3), torch.sin(t3),
    ], dim=1)

    y = (torch.sin(t1 + t2) * torch.cos(t3)).unsqueeze(1)
    return X, y


X_train, y_train = generate_data(2000)
X_val,   y_val   = generate_data(500)
X_probe, _       = generate_data(1000)


# ------------------------------------------------------------------ #
#  Saturation metrics                                                 #
# ------------------------------------------------------------------ #

def head_effective_rank(model, X, threshold=0.01):
    """SVD of combined head outputs — how many independent dims?"""
    model.eval()
    with torch.no_grad():
        outs     = [h(X) for h in model.heads]
        combined = torch.cat(outs, dim=1).float()
    sv     = torch.linalg.svdvals(combined)
    cutoff = threshold * sv[0].item()
    return int((sv > cutoff).sum().item())


def head_correlation(model, X):
    """Average off-diagonal correlation between heads."""
    model.eval()
    with torch.no_grad():
        scalars = [h(X).mean(dim=1) for h in model.heads]
    H        = torch.stack(scalars, dim=0).float()
    H_c      = H - H.mean(dim=1, keepdim=True)
    H_n      = H_c / (H_c.norm(dim=1, keepdim=True) + 1e-8)
    corr     = (H_n @ H_n.T).abs()
    n        = corr.shape[0]
    if n == 1:
        return 0.0
    return ((corr.sum() - corr.trace()) / (n * (n - 1))).item()


# ------------------------------------------------------------------ #
#  Training                                                           #
# ------------------------------------------------------------------ #

def train(model, name, epochs=50_000, lr=1e-3, weight_decay=1e-2):
    opt     = torch.optim.AdamW(model.parameters(),
                                lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    print(f"\n{'='*55}")
    print(f"  {name}  ({count_params(model):,} params)")
    print(f"{'='*55}")
    print(f"  {'epoch':>8}  {'train':>10}  {'val':>10}")
    print(f"  {'-'*32}")

    for epoch in range(epochs + 1):
        model.train()
        loss = loss_fn(model(X_train), y_train)
        opt.zero_grad()
        loss.backward()
        opt.step()

        if epoch % 10_000 == 0:
            model.eval()
            with torch.no_grad():
                val_loss = loss_fn(model(X_val), y_val).item()
            print(f"  {epoch:>8}  {loss.item():>10.6f}  {val_loss:>10.6f}")

    model.eval()
    with torch.no_grad():
        final_val = loss_fn(model(X_val), y_val).item()

    eff_rank = head_effective_rank(model, X_probe)
    corr     = head_correlation(model, X_probe)

    torch.save(model, f"checkpoints/heads_{name}.pt")
    print(f"\n  val loss      : {final_val:.8f}")
    print(f"  effective rank: {eff_rank}")
    print(f"  head corr     : {corr:.4f}")

    return {
        "name":      name,
        "n_heads":   len(model.heads),
        "params":    count_params(model),
        "val_loss":  final_val,
        "eff_rank":  eff_rank,
        "head_corr": corr,
    }


# ------------------------------------------------------------------ #
#  Run                                                                #
# ------------------------------------------------------------------ #

HEAD_COUNTS = [2, 3, 4, 5, 6]
results     = []

for n_heads in HEAD_COUNTS:
    torch.manual_seed(42)
    model = TransformerWithNHeads(n_heads=n_heads)
    results.append(train(model, f"{n_heads}heads"))


# ------------------------------------------------------------------ #
#  Summary                                                            #
# ------------------------------------------------------------------ #

print(f"\n{'='*60}")
print(f"  Results — intrinsic dimensionality = 3")
print(f"{'='*60}")
print(f"  {'heads':>6}  {'val':>10}  {'eff_rank':>10}  {'corr':>8}")
print(f"  {'-'*40}")
for r in results:
    marker = " ←" if r["n_heads"] == 3 else ""
    print(
        f"  {r['n_heads']:>6}  {r['val_loss']:>10.6f}"
        f"  {r['eff_rank']:>10}  {r['head_corr']:>8.4f}{marker}"
    )

print(f"""
  What to look for:
  effective_rank should plateau at ~3 from h=3 onward
  head_corr should rise beyond h=3 (redundant heads)

  Load checkpoints into MIRAI Heads tab
  to see each head's output dimensions.
""")