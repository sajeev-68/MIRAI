"""
train_compare.py

Trains two models on clean e^x (no noise, no normalisation) and saves both.

Tests the core assumption:
    wide shallow → many narrow brittle pockets (stitching)
    deep narrow  → few wide robust pockets (structural)

Models are parameter-matched:
    wide shallow: depth 2,  width 62 — ~4093 params
    deep narrow:  depth 16, width 16 — ~4129 params

Load both into MIRAI Function view and compare:
    - pocket count        (wide should have more)
    - boundary distance   (deep should be larger)
    - effective rank      (deep should use more of its capacity)

Run:
    python train_compare.py
"""

import torch
import torch.nn as nn
from pathlib import Path

torch.manual_seed(42)
Path("checkpoints").mkdir(exist_ok=True)


# ------------------------------------------------------------------ #
#  Data — clean, no noise, no normalisation                           #
# ------------------------------------------------------------------ #

x_train = torch.linspace(-2.0, 2.0, 100).unsqueeze(1)
y_train = torch.exp(x_train)   # raw e^x, range ~[0.14, 7.39]

x_val   = torch.linspace(-2.0, 2.0, 500).unsqueeze(1)
y_val   = torch.exp(x_val)


# ------------------------------------------------------------------ #
#  Models                                                             #
# ------------------------------------------------------------------ #

def wide_shallow() -> nn.Module:
    """
    depth 2, width 62 — 4093 params
    Classic stitcher — many linear pieces fitted locally.
    """
    return nn.Sequential(
        nn.Linear(1,  62), nn.ReLU(),
        nn.Linear(62, 62), nn.ReLU(),
        nn.Linear(62, 1),
    )


def deep_narrow() -> nn.Module:
    """
    depth 16, width 16 — 4129 params
    Structural — depth matches ~12 Taylor terms needed for e^x on [-2,2].
    Width 16 avoids dying ReLU while keeping capacity constrained.
    """
    layers = [nn.Linear(1, 16), nn.ReLU()]
    for _ in range(14):
        layers += [nn.Linear(16, 16), nn.ReLU()]
    layers += [nn.Linear(16, 1)]
    return nn.Sequential(*layers)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


# ------------------------------------------------------------------ #
#  Training                                                           #
# ------------------------------------------------------------------ #

def train(model: nn.Module, name: str,
          epochs: int = 200000,
          lr: float = 1e-3,
          weight_decay: float = 5e-2) -> None:

    opt     = torch.optim.AdamW(model.parameters(),
                                lr=lr,
                                weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"  params       : {count_params(model)}")
    print(f"  epochs       : {epochs}")
    print(f"  lr           : {lr}")
    print(f"  weight_decay : {weight_decay}")
    print(f"{'='*50}")
    print(f"  {'epoch':>8}  {'train loss':>12}  {'val loss':>12}")
    print(f"  {'-'*36}")

    for epoch in range(epochs + 1):
        model.train()
        pred = model(x_train)
        loss = loss_fn(pred, y_train)

        opt.zero_grad()
        loss.backward()
        opt.step()

        if epoch % 5000 == 0:
            model.eval()
            with torch.no_grad():
                val_loss = loss_fn(model(x_val), y_val).item()
            print(f"  {epoch:>8}  {loss.item():>12.6f}  {val_loss:>12.6f}")

    # final eval
    model.eval()
    with torch.no_grad():
        final_val = loss_fn(model(x_val), y_val).item()
    print(f"\n  Final val loss: {final_val:.8f}")

    torch.save(model, f"checkpoints/{name}.pt")
    print(f"  Saved → checkpoints/{name}.pt")


# ------------------------------------------------------------------ #
#  Run                                                                #
# ------------------------------------------------------------------ #

print("\nTraining wide shallow model...")
torch.manual_seed(42)
model_a = wide_shallow()
train(model_a, "wide_shallow")

print("\nTraining deep narrow model...")
torch.manual_seed(42)
model_b = deep_narrow()
train(model_b, "deep_narrow")


# ------------------------------------------------------------------ #
#  Summary                                                            #
# ------------------------------------------------------------------ #

print(f"\n{'='*50}")
print(f"  Parameter comparison")
print(f"{'='*50}")
print(f"  wide_shallow (d=2,  w=62): {count_params(wide_shallow())} params")
print(f"  deep_narrow  (d=16, w=16): {count_params(deep_narrow())} params")
print(f"  difference              : "
      f"{abs(count_params(wide_shallow()) - count_params(deep_narrow()))} params")

print(f"""
{'='*50}
  What to look for in MIRAI Function view
{'='*50}

  Load checkpoints/wide_shallow.pt
  Load checkpoints/deep_narrow.pt

  Compare on the same input range [-2, 2]:

  1. Pocket count (yellow lines in Function view)
     wide shallow → expect more pockets (~25-40)
     deep narrow  → expect fewer pockets (~8-15)

  2. Boundary distance (Debugger tab)
     wide shallow → small  (brittle, close to boundaries)
     deep narrow  → large  (robust, far from boundaries)

  3. Residual error shape (Function view panel 2)
     wide shallow → flat with small wiggles (pure stitching)
     deep narrow  → structured residual if not converged,
                    flat if it found the Taylor structure

  4. Local slope panel (Function view panel 3)
     wide shallow → many slope jumps (many piece transitions)
     deep narrow  → fewer smoother slope transitions
""")