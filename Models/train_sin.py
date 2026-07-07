"""
train_sin.py

Fair comparison of wide shallow vs deep narrow on sin(x).

sin(x) is the right test for depth vs width because:
  - it has curvature reversals (oscillates)
  - Taylor series needs alternating terms: x - x³/6 + x⁵/120 - ...
  - depth should help because composition can represent sign changes
  - width alone struggles because all pieces slope in one direction

Training range:  [-2π, 2π]  — two full cycles
OOD test range:  [-3π, 3π]  — one extra cycle each side

Parameter matched:
  wide shallow: depth 2,  width 62 — ~4093 params
  deep narrow:  depth 16, width 16 — ~4129 params

Falsifiable predictions:
  1. Deep narrow achieves fewer pockets than wide shallow
  2. Deep narrow extrapolates better to [-3π, 3π]
  3. Deep narrow residual is more symmetric (sin is odd function)
     wide shallow residual will be asymmetric (no inductive bias)

Run:
    python train_sin.py
"""

import torch
import torch.nn as nn
from pathlib import Path

torch.manual_seed(42)
Path("checkpoints").mkdir(exist_ok=True)

PI = torch.pi


# ------------------------------------------------------------------ #
#  Data                                                               #
# ------------------------------------------------------------------ #

# training: 2 full cycles, 200 points
x_train = torch.linspace(-2 * PI, 2 * PI, 200).unsqueeze(1)
y_train = torch.sin(x_train)

# validation: same range, denser
x_val = torch.linspace(-2 * PI, 2 * PI, 1000).unsqueeze(1)
y_val = torch.sin(x_val)

# OOD: one extra cycle each side — never seen during training
x_ood = torch.linspace(-3 * PI, 3 * PI, 1000).unsqueeze(1)
y_ood = torch.sin(x_ood)


# ------------------------------------------------------------------ #
#  Models — parameter matched ~4100 params each                      #
# ------------------------------------------------------------------ #

def wide_shallow() -> nn.Module:
    """depth 2, width 62 — 4093 params"""
    return nn.Sequential(
        nn.Linear(1,  62), nn.ReLU(),
        nn.Linear(62, 62), nn.ReLU(),
        nn.Linear(62, 1),
    )


def deep_narrow() -> nn.Module:
    """depth 16, width 16 — 4129 params"""
    layers = [nn.Linear(1, 16), nn.ReLU()]
    for _ in range(14):
        layers += [nn.Linear(16, 16), nn.ReLU()]
    layers += [nn.Linear(16, 1)]
    return nn.Sequential(*layers)


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


# ------------------------------------------------------------------ #
#  Training                                                           #
# ------------------------------------------------------------------ #

def train(model: nn.Module, name: str,
          epochs: int = 100_000,
          lr: float = 1e-3,
          weight_decay: float = 5e-2) -> dict:

    opt     = torch.optim.AdamW(model.parameters(),
                                lr=lr,
                                weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    print(f"\n{'='*55}")
    print(f"  {name}  ({count_params(model)} params)")
    print(f"  epochs={epochs}  lr={lr}  wd={weight_decay}")
    print(f"{'='*55}")
    print(f"  {'epoch':>8}  {'train':>10}  {'val':>10}  {'ood':>10}")
    print(f"  {'-'*44}")

    best_val = float("inf")

    for epoch in range(epochs + 1):
        model.train()
        loss = loss_fn(model(x_train), y_train)
        opt.zero_grad()
        loss.backward()
        opt.step()

        if epoch % 10_000 == 0:
            model.eval()
            with torch.no_grad():
                val_loss = loss_fn(model(x_val), y_val).item()
                ood_loss = loss_fn(model(x_ood), y_ood).item()

            if val_loss < best_val:
                best_val = val_loss
                torch.save(model, f"checkpoints/{name}_best.pt")

            print(f"  {epoch:>8}  {loss.item():>10.6f}"
                  f"  {val_loss:>10.6f}  {ood_loss:>10.6f}")

    # final save
    torch.save(model, f"checkpoints/{name}_final.pt")

    model.eval()
    with torch.no_grad():
        final_val = loss_fn(model(x_val), y_val).item()
        final_ood = loss_fn(model(x_ood), y_ood).item()

    print(f"\n  Final val loss : {final_val:.8f}")
    print(f"  Final OOD loss : {final_ood:.8f}")
    print(f"  OOD / val ratio: {final_ood / final_val:.2f}x")
    print(f"  (lower ratio = better generalization)")
    print(f"  Saved → checkpoints/{name}_final.pt")

    return {
        "name":      name,
        "val_loss":  final_val,
        "ood_loss":  final_ood,
        "ood_ratio": final_ood / final_val,
    }


# ------------------------------------------------------------------ #
#  Run both                                                           #
# ------------------------------------------------------------------ #

torch.manual_seed(42)
results_wide = train(wide_shallow(), "sin_wide_shallow")

torch.manual_seed(42)
results_deep = train(deep_narrow(), "sin_deep_narrow")


# ------------------------------------------------------------------ #
#  Summary                                                            #
# ------------------------------------------------------------------ #

print(f"""
{'='*55}
  Results summary
{'='*55}

  {'Model':<20} {'Val loss':>10} {'OOD loss':>10} {'OOD ratio':>10}
  {'-'*52}
  {'wide_shallow':<20} {results_wide['val_loss']:>10.6f} \
{results_wide['ood_loss']:>10.6f} {results_wide['ood_ratio']:>10.2f}x
  {'deep_narrow':<20} {results_deep['val_loss']:>10.6f} \
{results_deep['ood_loss']:>10.6f} {results_deep['ood_ratio']:>10.2f}x

  Predictions:
  1. deep_narrow OOD ratio should be lower (better generalization)
  2. deep_narrow pocket count should be lower in MIRAI
  3. deep_narrow residual should be symmetric around x=0

  What to do next:
  Load both _final.pt files into MIRAI Function view.
  Set function to sin(x).

  In-distribution test:  x range [-6.28, 6.28]   (2π ≈ 6.28)
  OOD test:              x range [-9.42, 9.42]    (3π ≈ 9.42)

  The OOD slope panel is the key — does either model
  continue oscillating beyond the training range?
  A model that learned sin's structure should show
  continued oscillation. A stitcher will go flat.
""")