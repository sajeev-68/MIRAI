"""
function_viz.py

Plots the true mathematical function against the model's learned
approximation over a user-specified range.

Pocket detection:
    For each input point, collect the raw collapsed linear equation
    (e.g. "y_0 = 0.8273x_0") directly from the model's Jacobian.
    Group consecutive inputs that share the same equation.
    Each unique equation = one pocket, colored distinctly.
    No slope comparison, no thresholds — just raw equation collection.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from .probe_engine import ProbeEngine
from .equationformatter import EquationFormatter


# known functions the user can select
FUNCTIONS = {
    "e^x":        lambda x: torch.exp(x),
    "sin(x)":     lambda x: torch.sin(x),
    "cos(x)":     lambda x: torch.cos(x),
    "x²":         lambda x: x ** 2,
    "x³":         lambda x: x ** 3,
    "x³ - x":     lambda x: x ** 3 - x,
    "tanh(x)":    lambda x: torch.tanh(x),
    "1/(1+e^-x)": lambda x: torch.sigmoid(x),
    "none":       None,
}

# distinct colors for pockets — cycles if more pockets than colors
POCKET_COLORS = [
    "#3b82f6", "#f59e0b", "#10b981", "#ef4444", "#8b5cf6",
    "#06b6d4", "#f97316", "#84cc16", "#ec4899", "#14b8a6",
    "#a855f7", "#eab308", "#22c55e", "#f43f5e", "#0ea5e9",
    "#d946ef", "#fb923c", "#4ade80", "#fb7185", "#38bdf8",
]


def _eq_key(jacobian: torch.Tensor, precision: int = 3) -> str:
    """
    Convert a Jacobian tensor to a rounded string key.
    Two inputs are in the same pocket if their keys match.
    precision=3 means slopes must agree to 3 decimal places.
    """
    j = jacobian.float()
    if j.ndim == 0:
        j = j.reshape(1, 1)
    elif j.ndim == 1:
        j = j.unsqueeze(0)
    # round each entry to `precision` decimal places
    rounded = torch.round(j * (10 ** precision)) / (10 ** precision)
    return str(rounded.tolist())


def plot_function_vs_model(
    model: nn.Module,
    fn_name: str,
    x_min: float,
    x_max: float,
    n_points: int = 300,
) -> plt.Figure:
    """
    Plot true function vs model output over [x_min, x_max].
    Pocket regions colored by their unique linear equation.
    """
    engine    = ProbeEngine()
    formatter = EquationFormatter(precision=3)

    x    = torch.linspace(x_min, x_max, n_points).unsqueeze(1)
    x_np = x.squeeze().numpy()

    # ── model predictions ─────────────────────────────────────────
    model.eval()
    with torch.no_grad():
        y_model = model(x).squeeze().numpy()

    # ── true function ─────────────────────────────────────────────
    fn = FUNCTIONS.get(fn_name)
    if fn is not None:
        with torch.no_grad():
            y_true = fn(x.squeeze()).numpy()
        has_true = True
    else:
        y_true   = None
        has_true = False

    # ── collect equation for every sampled point ──────────────────
    # raw collection — no slope comparison, no threshold
    # just: what is the linear equation at this input?
    print(f"Collecting equations for {n_points} points...")

    point_equations = []   # list of (x_val, eq_key, eq_latex)
    for i in range(n_points):
        result   = engine.probe(model, x[i])
        key      = _eq_key(result.full_jacobian)
        latex    = formatter.format_jacobian_eq(result.full_jacobian, x[i])
        point_equations.append((x_np[i], key, latex))

    # ── group consecutive points with the same equation ───────────
    # a pocket = maximal run of consecutive points with the same key
    pockets = []   # list of {x_start, x_end, key, latex, indices}
    current_key    = None
    current_latex  = None
    current_start  = None
    current_start_idx = 0

    for i, (xv, key, latex) in enumerate(point_equations):
        if key != current_key:
            if current_key is not None:
                pockets.append({
                    "x_start": current_start,
                    "x_end":   xv,
                    "key":     current_key,
                    "latex":   current_latex,
                    "i_start": current_start_idx,
                    "i_end":   i,
                })
            current_key       = key
            current_latex     = latex
            current_start     = xv
            current_start_idx = i

    # close last pocket
    if current_key is not None:
        pockets.append({
            "x_start": current_start,
            "x_end":   x_np[-1],
            "key":     current_key,
            "latex":   current_latex,
            "i_start": current_start_idx,
            "i_end":   n_points,
        })

    # assign a color to each unique equation
    unique_keys   = list(dict.fromkeys(p["key"] for p in pockets))
    key_to_color  = {
        k: POCKET_COLORS[i % len(POCKET_COLORS)]
        for i, k in enumerate(unique_keys)
    }
    key_to_latex  = {p["key"]: p["latex"] for p in pockets}

    n_unique = len(unique_keys)
    print(f"Found {len(pockets)} pocket regions, {n_unique} unique equations")

    # ── figure ────────────────────────────────────────────────────
    n_panels = 3 if has_true else 2
    fig, axes = plt.subplots(n_panels, 1, figsize=(12, 4 * n_panels))
    if n_panels == 1:
        axes = [axes]

    fig.patch.set_facecolor("#0f0f0f")
    for ax in axes:
        ax.set_facecolor("#111111")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#333333")

    def shade_pockets(ax):
        """Shade each pocket region with its equation color."""
        for p in pockets:
            color = key_to_color[p["key"]]
            ax.axvspan(
                p["x_start"], p["x_end"],
                alpha=0.15, color=color, linewidth=0,
            )
            # boundary line at pocket start
            ax.axvline(
                p["x_start"], color=color,
                linewidth=0.8, alpha=0.6, linestyle="-",
            )

    # ── panel 1: function vs model, pockets colored ───────────────
    ax = axes[0]
    shade_pockets(ax)
    ax.plot(x_np, y_model, color="white", linewidth=2,
            label="model output", zorder=4)
    if has_true:
        ax.plot(x_np, y_true, color="#34d399", linewidth=2,
                linestyle="--", label=f"true {fn_name}", zorder=3)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    reuse_ratio = len(pockets) / max(n_unique, 1)
    ax.set_title(
        f"Model vs {fn_name}   |   "
        f"{len(pockets)} regions   |   "
        f"{n_unique} unique equations   |   "
        f"reuse ratio {reuse_ratio:.2f}x"
    )
    ax.legend(fontsize=8, facecolor="#2a2a2a",
              edgecolor="#444444", labelcolor="white")

    # ── panel 2: equation map ─────────────────────────────────────
    # shows which equation is active at each x
    ax = axes[1]
    shade_pockets(ax)

    # plot equation index as a step function
    eq_indices = [unique_keys.index(p["key"])
                  for p in pockets
                  for _ in range(p["i_end"] - p["i_start"])]
    # pad to n_points
    while len(eq_indices) < n_points:
        eq_indices.append(eq_indices[-1] if eq_indices else 0)
    eq_indices = eq_indices[:n_points]

    ax.step(x_np, eq_indices, color="white", linewidth=1.5,
            where="post", zorder=4)
    ax.set_xlabel("x")
    ax.set_ylabel("equation index")
    ax.set_title(
        f"Active equation map   |   "
        f"{n_unique} unique equations reused across {len(pockets)} regions"
    )
    ax.set_yticks(range(n_unique))
    ax.tick_params(axis="y", labelsize=6, colors="white")

    if has_true:
        # ── panel 3: residual ─────────────────────────────────────
        ax = axes[2]
        shade_pockets(ax)
        residual = y_true - y_model
        ax.plot(x_np, residual, color="#f87171", linewidth=1.5,
                label="residual (true − model)", zorder=4)
        ax.axhline(0, color="#555555", linewidth=0.8)
        ax.fill_between(x_np, residual, 0,
                        alpha=0.15, color="#f87171")
        ax.set_xlabel("x")
        ax.set_ylabel("error")
        ax.set_title(
            f"Residual   |   "
            f"max={np.abs(residual).max():.4f}   "
            f"mean={np.abs(residual).mean():.4f}"
        )
        ax.legend(fontsize=8, facecolor="#2a2a2a",
                  edgecolor="#444444", labelcolor="white")

    # ── legend: unique equations ──────────────────────────────────
    # show up to 10 equations in a text box
    eq_lines = []
    for i, key in enumerate(unique_keys[:10]):
        latex = key_to_latex[key]
        eq_lines.append(f"[{i}] {latex}")
    if n_unique > 10:
        eq_lines.append(f"... +{n_unique - 10} more")

    # count how many regions each unique equation covers
    eq_usage = {}
    for p in pockets:
        eq_usage[p["key"]] = eq_usage.get(p["key"], 0) + 1

    eq_lines_with_usage = []
    for i, key in enumerate(unique_keys[:10]):
        latex   = key_to_latex[key]
        n_uses  = eq_usage[key]
        eq_lines_with_usage.append(
            f"[{i}] used {n_uses}x — {latex}"
        )
    if n_unique > 10:
        eq_lines_with_usage.append(f"... +{n_unique - 10} more")

    eq_lines_with_usage.insert(0,
        f"Total: {len(pockets)} regions, "
        f"{n_unique} unique equations, "
        f"reuse {reuse_ratio:.2f}x"
    )

    fig.text(
        0.01, 0.01,
        "\n".join(eq_lines_with_usage),
        fontsize=6, color="#aaaaaa",
        va="bottom", family="monospace",
        bbox=dict(facecolor="#1a1a1a", edgecolor="#333333",
                  boxstyle="round,pad=0.4"),
    )

    plt.tight_layout(rect=[0, 0.08, 1, 1])
    return fig