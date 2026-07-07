"""
train_model.py

Trains a small MLP to approximate e^x.
Uses weight decay + long training to encourage grokking.
Saves model checkpoints so you can compare pre/post grokking in MIRAI.

Usage:
    python train_model.py
"""

import torch
import torch.nn as nn
from pathlib import Path

# ------------------------------------------------------------------ #
#  Target function                                                     #
# ------------------------------------------------------------------ #

def target(x: torch.Tensor) -> torch.Tensor:
    return torch.exp(x)


# ------------------------------------------------------------------ #
#  Model                                                              #
# ------------------------------------------------------------------ #

torch.manual_seed(42)

model = nn.Sequential(
    nn.Linear(1, 32),
    nn.ReLU(),
    nn.Linear(32, 32),
    nn.ReLU(),
    nn.Linear(32, 1),
)


# ------------------------------------------------------------------ #
#  Data — small dataset encourages grokking                           #
# ------------------------------------------------------------------ #

# training: 80 points on [-2, 2]
x_train = torch.linspace(-2.0, 2.0, 80).unsqueeze(1)
y_train = target(x_train)

# validation: 500 points on same range (held out)
x_val = torch.linspace(-2.0, 2.0, 500).unsqueeze(1)
y_val = target(x_val)

# normalise targets so loss scale is reasonable
y_mean = y_train.mean()
y_std  = y_train.std()
y_train_n = (y_train - y_mean) / y_std
y_val_n   = (y_val   - y_mean) / y_std


# ------------------------------------------------------------------ #
#  Optimiser — weight decay is the key grokking ingredient            #
# ------------------------------------------------------------------ #

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-3,
    weight_decay=1e-2,   # increase to 1e-1 if grokking doesn't appear
)
loss_fn = nn.MSELoss()

# save directory
Path("checkpoints").mkdir(exist_ok=True)


# ------------------------------------------------------------------ #
#  Training loop                                                       #
# ------------------------------------------------------------------ #

print("Training e^x approximator\n")
print(f"{'epoch':>8}  {'train loss':>12}  {'val loss':>12}  {'gap':>10}")
print("-" * 50)

best_val    = float("inf")
grok_epoch  = None

for epoch in range(100_001):
    # ── train step ────────────────────────────────────────────────
    model.train()
    y_pred = model(x_train)
    loss   = loss_fn(y_pred, y_train_n)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # ── validation ────────────────────────────────────────────────
    if epoch % 1000 == 0:
        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(x_val), y_val_n)

        train_l = loss.item()
        val_l   = val_loss.item()
        gap     = val_l - train_l

        print(f"{epoch:>8}  {train_l:>12.6f}  {val_l:>12.6f}  {gap:>10.6f}")

        # detect grokking — val loss suddenly drops close to train loss
        if val_l < best_val * 0.5 and epoch > 5000:
            print(f"\n  *** possible grokking at epoch {epoch} ***\n")
            grok_epoch = epoch
            torch.save(model, f"checkpoints/model_grok_{epoch}.pt")

        if val_l < best_val:
            best_val = val_l

    # ── periodic checkpoints to compare in MIRAI ──────────────────
    if epoch in (1000, 5000, 10000, 50000, 100000):
        torch.save(model, f"checkpoints/model_epoch_{epoch}.pt")
        print(f"  → saved checkpoint at epoch {epoch}")


# ------------------------------------------------------------------ #
#  Save final model                                                    #
# ------------------------------------------------------------------ #

torch.save(model, "model.pt")
print("\nSaved final model → model.pt")

if grok_epoch:
    print(f"Grokking detected at epoch {grok_epoch}")
    print(f"Compare checkpoints/model_epoch_1000.pt  (memorizing)")
    print(f"     vs checkpoints/model_grok_{grok_epoch}.pt  (grokked)")
else:
    print("\nNo clear grokking detected — try:")
    print("  1. increase weight_decay to 1e-1")
    print("  2. reduce training points (change 80 → 40)")
    print("  3. run longer (change 100_001 → 300_001)")

print("\nProbe any checkpoint in MIRAI with input like [0.5]")