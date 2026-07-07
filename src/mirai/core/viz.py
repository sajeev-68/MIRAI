"""
visualizer.py

Draws the neural network as a layered graph using matplotlib.
Active neurons are colored by activation value.
Dead neurons (ReLU killed) are shown in dark gray.
"""

import torch
import matplotlib
matplotlib.use("Agg")   # no display needed
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import numpy as np

from .types import ProbeResult


# max neurons to draw per layer before switching to summary mode
MAX_DRAW = 24


def draw_network(result: ProbeResult) -> plt.Figure:
    """
    Given a ProbeResult, return a matplotlib Figure of the network graph.
    Neurons are colored by post-activation value.
    Dead neurons are gray.
    """

    layer_names  = result.layer_order
    layer_records = result.layer_records

    # collect per-layer data in forward order
    layers = []
    for name in layer_names:
        if name not in layer_records:
            continue
        record = layer_records[name]
        post   = record.post_activation.float().flatten()
        mask   = record.relu_mask.flatten()
        layers.append({
            "name":   name,
            "post":   post,
            "mask":   mask,
            "n":      len(post),
        })

    # add input layer
    x_in = result.input_tensor.float().flatten()
    layers.insert(0, {
        "name": "input",
        "post": x_in,
        "mask": torch.ones(len(x_in), dtype=torch.bool),
        "n":    len(x_in),
    })

    # add output layer
    out = result.output_tensor.float().flatten()
    layers.append({
        "name": "output",
        "post": out,
        "mask": torch.ones(len(out), dtype=torch.bool),
        "n":    len(out),
    })

    n_layers = len(layers)
    fig_w    = max(10, n_layers * 2.5)
    fig_h    = 8
    fig, ax  = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_facecolor("#0f0f0f")
    fig.patch.set_facecolor("#0f0f0f")
    ax.axis("off")

    # colormap for activation values
    cmap     = plt.cm.plasma
    all_vals = torch.cat([l["post"] for l in layers]).numpy()
    vmin, vmax = float(all_vals.min()), float(all_vals.max())
    if vmin == vmax:
        vmin -= 1e-6
        vmax += 1e-6
    norm     = Normalize(vmin=vmin, vmax=vmax)
    sm       = ScalarMappable(cmap=cmap, norm=norm)

    # x positions — evenly spaced
    xs = np.linspace(0.08, 0.92, n_layers)

    # store neuron centres for drawing edges
    centres = []   # list of list of (x, y)

    for col, (lx, layer) in enumerate(zip(xs, layers)):
        n       = layer["n"]
        draw_n  = min(n, MAX_DRAW)
        post    = layer["post"]
        mask    = layer["mask"]

        # y positions — centred vertically
        if draw_n == 1:
            ys = np.array([0.5])
        else:
            ys = np.linspace(0.1, 0.9, draw_n)

        layer_centres = []

        for row in range(draw_n):
            cy = ys[row]

            if n > MAX_DRAW:
                # map row to actual neuron index
                idx = int(row * (n - 1) / (draw_n - 1))
            else:
                idx = row

            val     = post[idx].item()
            is_live = mask[idx].item() if idx < len(mask) else True

            if is_live:
                rgba = sm.to_rgba(val)
            else:
                rgba = (0.18, 0.18, 0.18, 1.0)   # dead — dark gray

            circle = mpatches.Circle(
                (lx, cy),
                radius=0.018,
                color=rgba,
                zorder=3,
            )
            ax.add_patch(circle)

            # value label for small layers
            if n <= 8:
                ax.text(
                    lx, cy, f"{val:.2f}",
                    ha="center", va="center",
                    fontsize=5.5, color="white",
                    zorder=4,
                )

            layer_centres.append((lx, cy))

        # collapsed indicator if layer was truncated
        if n > MAX_DRAW:
            ax.text(
                lx, 0.04,
                f"({n} neurons,\nshowing {draw_n})",
                ha="center", va="bottom",
                fontsize=6, color="#888888",
            )

        centres.append(layer_centres)

        # layer name
        short = layer["name"].split(".")[-1]   # trim module path
        ax.text(
            lx, 0.97, short,
            ha="center", va="top",
            fontsize=7, color="#cccccc",
            rotation=30,
        )

        # active / dead count (skip input/output)
        if layer["name"] not in ("input", "output"):
            n_active = int(mask.sum().item())
            ax.text(
                lx, 0.01,
                f"{n_active}/{n} active",
                ha="center", va="bottom",
                fontsize=6, color="#aaaaaa",
            )

    # draw edges between adjacent layers
    for col in range(len(centres) - 1):
        src = centres[col]
        dst = centres[col + 1]

        # draw every edge for small layers, sample for large
        max_edges = 60
        pairs = [(s, d) for s in src for d in dst]
        if len(pairs) > max_edges:
            step  = max(1, len(pairs) // max_edges)
            pairs = pairs[::step]

        for (sx, sy), (dx, dy) in pairs:
            ax.plot(
                [sx, dx], [sy, dy],
                color="#333333", linewidth=0.4,
                alpha=0.6, zorder=1,
            )

    # colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.012, 0.7])
    fig.colorbar(sm, cax=cbar_ax)
    cbar_ax.yaxis.label.set_color("white")
    cbar_ax.tick_params(colors="white", labelsize=6)
    cbar_ax.set_ylabel("activation", color="white", fontsize=7)

    # legend
    live_patch = mpatches.Patch(color=cmap(0.8), label="active neuron")
    dead_patch = mpatches.Patch(color="#2e2e2e",  label="dead (ReLU)")
    ax.legend(
        handles=[live_patch, dead_patch],
        loc="lower right",
        fontsize=7,
        facecolor="#1a1a1a",
        edgecolor="#444444",
        labelcolor="white",
    )

    # title
    sig_short = result.region_signature[:20] + ("…" if len(result.region_signature) > 20 else "")
    ax.set_title(
        f"region: {sig_short}   |   boundary dist: {result.global_boundary_dist:.4f}",
        color="#cccccc", fontsize=8, pad=4,
    )

    plt.tight_layout()
    return fig