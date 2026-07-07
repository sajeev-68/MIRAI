"""
train_resnet_sin.py

Standalone script — no external dependencies except torch.
Tests the Taylor depth hypothesis using ResNet architecture.

Run:
    python train_resnet_sin.py
"""

import torch
import torch.nn as nn
from pathlib import Path

torch.manual_seed(42)
Path("checkpoints").mkdir(exist_ok=True)

PI = torch.pi


# ------------------------------------------------------------------ #
#  Model definitions                                                  #
# ------------------------------------------------------------------ #

class ResBlock(nn.Module):
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


# ------------------------------------------------------------------ #
#  Data                                                               #
# ------------------------------------------------------------------ #

x_train = torch.linspace(-2 * PI, 2 * PI, 200).unsqueeze(1)
y_train = torch.sin(x_train)

x_val   = torch.linspace(-2 * PI, 2 * PI, 1000).unsqueeze(1)
y_val   = torch.sin(x_val)

x_ood   = torch.linspace(-4 * PI, 4 * PI, 1000).unsqueeze(1)
y_ood   = torch.sin(x_ood)


# ------------------------------------------------------------------ #
#  Models                                                             #
# ------------------------------------------------------------------ #

FIXED_WIDTH = 8

MODELS = {
    "wide_shallow" : make_wide_shallow(62),
    "resnet_4"     : make_resnet(4,  FIXED_WIDTH),
    "resnet_8"     : make_resnet(8,  FIXED_WIDTH),
    "resnet_16"    : make_resnet(16, FIXED_WIDTH),
}

print(f"\n{'='*55}")
print(f"  Model parameter counts")
print(f"  wide_shallow : width=62 depth=2  (stitcher baseline)")
print(f"  resnets      : width=8  depth varies (forced structure)")
print(f"{'='*55}")
for name, model in MODELS.items():
    print(f"  {name:<20} {count_params(model):>6} params")


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

    model.eval()
    with torch.no_grad():
        final_val = loss_fn(model(x_val), y_val).item()
        final_ood = loss_fn(model(x_ood), y_ood).item()

    torch.save(model, f"checkpoints/{name}_final.pt")

    print(f"\n  val loss : {final_val:.8f}")
    print(f"  ood loss : {final_ood:.8f}")

    return {
        "name":     name,
        "params":   count_params(model),
        "val_loss": final_val,
        "ood_loss": final_ood,
    }


# ------------------------------------------------------------------ #
#  Run                                                                #
# ------------------------------------------------------------------ #

results = []
for name, model in MODELS.items():
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            nn.init.zeros_(m.bias)
    torch.manual_seed(42)
    results.append(train(model, name))


# ------------------------------------------------------------------ #
#  Summary                                                            #
# ------------------------------------------------------------------ #

print(f"\n{'='*55}")
print(f"  Final results")
print(f"{'='*55}")
print(f"  {'model':<20} {'params':>7} {'val':>10} {'ood':>10}")
print(f"  {'-'*50}")
for r in results:
    print(f"  {r['name']:<20} {r['params']:>7}"
          f"  {r['val_loss']:>10.6f}  {r['ood_loss']:>10.6f}")

print(f"""
  Predictions:
  1. resnet_8  ood < wide_shallow ood   (depth beats width OOD)
  2. resnet_4  ood > resnet_8 ood       (too few Taylor terms)
  3. resnet_16 ood ≈ resnet_8 ood       (saturation at depth 8)

  Load checkpoints into MIRAI Function view:
  In-distribution : x = [-6.28,   6.28]
  OOD             : x = [-12.56, 12.56]
""")