"""
app.py — MIRAI Neural Network Debugger
Three tabs:
    1. Equations   — end-to-end and per-neuron linear equations
    2. Function    — model output vs true function with pocket coloring
    3. Heads       — transformer attention head analysis
"""

import torch
import torch.nn as nn
import gradio as gr
import json
import datetime
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mirai.core.probe_engine import ProbeEngine
from mirai.core.types import ProbeResult
from mirai.core.fviz import plot_function_vs_model, FUNCTIONS

# ------------------------------------------------------------------ #
#  Global state                                                        #
# ------------------------------------------------------------------ #

engine           = ProbeEngine()
_last_result     : ProbeResult | None = None
_loaded_model    : nn.Module   | None = None
_loaded_model_name: str               = ""


# ------------------------------------------------------------------ #
#  Model loading                                                       #
# ------------------------------------------------------------------ #

def load_model(file) -> str:
    global _loaded_model, _loaded_model_name
    if file is None:
        return "No file uploaded."
    try:
        import __main__
        try:
            from mirai.core.models import ResBlock, make_resnet, make_wide_shallow
            __main__.ResBlock          = ResBlock
            __main__.make_resnet       = make_resnet
            __main__.make_wide_shallow = make_wide_shallow
        except ImportError:
            pass

        import __main__

        # ResNet classes
        try:
            from mirai.core.models import ResBlock, make_resnet, make_wide_shallow
            __main__.ResBlock          = ResBlock
            __main__.make_resnet       = make_resnet
            __main__.make_wide_shallow = make_wide_shallow
        except ImportError:
            pass

        # Head experiment classes
        try:
            from mirai.core.models_head import (
                FixedMLP, HeadProjection, TransformerWithNHeads
            )
            __main__.FixedMLP              = FixedMLP
            __main__.HeadProjection        = HeadProjection
            __main__.TransformerWithNHeads = TransformerWithNHeads
        except ImportError:
            pass

        model = torch.load(file.name, map_location="cpu", weights_only=False)
        model.eval()
        _loaded_model      = model
        _loaded_model_name = file.name.split("/")[-1].replace(".pt","").replace(".pth","")

        layers  = [(n, type(l).__name__)
                   for n, l in model.named_modules() if n != ""]
        summary = "\n".join(f"  {n}: {k}" for n, k in layers)
        total   = sum(p.numel() for p in model.parameters())
        return f"✓ {type(model).__name__}  ({total:,} params)\n\n{summary}"
    except Exception as e:
        return f"✗ {e}"


# ------------------------------------------------------------------ #
#  Tab 1 — Equations                                                   #
# ------------------------------------------------------------------ #

def run_probe(input_str: str):
    global _last_result
    if _loaded_model is None:
        return ("No model loaded.", "", gr.update(choices=[]))

    try:
        x = torch.tensor(json.loads(input_str), dtype=torch.float32)
    except Exception as e:
        return (f"✗ Invalid input: {e}", "", gr.update(choices=[]))

    try:
        result        = engine.probe(_loaded_model, x)
        _last_result  = result
    except Exception as e:
        return (f"✗ Probe failed: {e}", "", gr.update(choices=[]))

    status = f"✓ Probed  |  region: {result.region_signature[:20]}...  |  boundary dist: {result.global_boundary_dist:.4f}"

    # end-to-end equation
    eq = f"$${result.output_equation_latex}$$"

    # neuron dropdown
    choices = []
    for layer_name, eqs in result.neuron_equations.items():
        for e in eqs:
            choices.append(f"{layer_name} / neuron {e.neuron_index}")

    return status, eq, gr.update(choices=choices, value=choices[0] if choices else None)


def show_neuron_eq(selection: str):
    if _last_result is None or not selection:
        return "", "", ""
    try:
        layer_name, neuron_part = selection.split(" / ")
        neuron_idx = int(neuron_part.split(" ")[1])
    except Exception:
        return "Could not parse.", "", ""

    eqs = _last_result.neuron_equations.get(layer_name, [])
    eq  = next((e for e in eqs if e.neuron_index == neuron_idx), None)
    if eq is None:
        return "Not found.", "", ""

    status   = "🟢 ACTIVE" if eq.is_active else "🔴 DEAD (ReLU)"
    expanded = f"$${eq.latex_expanded}$$"
    return status, expanded, f"$${eq.latex_collapsed}$$"


# ------------------------------------------------------------------ #
#  Tab 2 — Function view                                               #
# ------------------------------------------------------------------ #

def run_function_plot(fn_name, x_min, x_max, n_pts):
    if _loaded_model is None:
        return None, "No model loaded."
    try:
        fig = plot_function_vs_model(
            _loaded_model, fn_name,
            float(x_min), float(x_max), int(n_pts)
        )
        return fig, "✓ Done"
    except Exception as e:
        return None, f"✗ {e}"


def save_function_plot(fn_name, x_min, x_max, n_pts):
    if _loaded_model is None:
        return "No model loaded."
    try:
        fig  = plot_function_vs_model(
            _loaded_model, fn_name,
            float(x_min), float(x_max), int(n_pts)
        )
        ts   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = f"{_loaded_model_name}_{fn_name}_{ts}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        return f"✓ Saved → {path}"
    except Exception as e:
        return f"✗ {e}"


# ------------------------------------------------------------------ #
#  Tab 3 — Transformer heads                                           #
# ------------------------------------------------------------------ #

def _get_head_layers(model):
    heads = []
    for name, module in model.named_modules():
        cls = type(module).__name__
        if cls == "HeadProjection" or (
            name != "" and (
                hasattr(module, "num_heads") or
                "attention" in cls.lower()
            )
        ):
            heads.append((name, module))
    return heads


def analyze_heads(input_str: str, x_range_str: str):
    if _loaded_model is None:
        return None, "No model loaded."

    # parse input
    try:
        raw_list = json.loads(input_str) if input_str.strip() else [1,0,1,0,1,0]
    except Exception:
        raw_list = [1,0,1,0,1,0]

    inp = torch.tensor(raw_list, dtype=torch.float32).unsqueeze(0)  # (1, d)

    head_layers = _get_head_layers(_loaded_model)
    if not head_layers:
        return None, "No head layers found."

    # feed input through each head, capture output
    results = []   # list of (name, input_vec, output_vec)
    for layer_name, module in head_layers:
        captured = {}

        def hook(mod, inp_h, out, d=captured):
            o = out[0] if isinstance(out, tuple) else out
            d["out"] = o.detach().cpu().squeeze(0).float()

        handle = module.register_forward_hook(hook)
        _loaded_model.eval()
        with torch.no_grad():
            try:
                _loaded_model(inp)
            except Exception:
                pass
        handle.remove()

        out_vec = captured.get("out", torch.zeros(4))
        results.append((layer_name, inp.squeeze(0), out_vec))

    # ── plot: one graph per head ──────────────────────────────────
    n_heads  = len(results)
    fig, axes = plt.subplots(1, n_heads,
                             figsize=(4 * n_heads, 4),
                             squeeze=False)
    fig.patch.set_facecolor("#0f0f0f")

    for i, (name, in_vec, out_vec) in enumerate(results):
        ax = axes[0][i]
        ax.set_facecolor("#111111")
        ax.tick_params(colors="white", labelsize=7)
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        for sp in ax.spines.values():
            sp.set_edgecolor("#333333")

        in_np  = in_vec.numpy()
        out_np = out_vec.numpy()

        # plot input as bars
        x_in  = np.arange(len(in_np))
        x_out = np.arange(len(out_np))

        ax.bar(x_in  - 0.2, in_np,  0.35,
               color="#60a5fa", alpha=0.8, label="input")
        ax.bar(x_out + 0.2, out_np, 0.35,
               color="#e879f9", alpha=0.8, label="output")

        ax.axhline(0, color="#444", linewidth=0.8)
        ax.set_xlabel("dimension")
        ax.set_ylabel("value")
        ax.set_title(f"head {i}")
        ax.legend(fontsize=7, facecolor="#1a1a1a",
                  edgecolor="#333", labelcolor="white")

    fig.suptitle(
        f"{n_heads} heads  |  blue=input  purple=output",
        color="white", fontsize=9
    )
    plt.tight_layout()

    # text summary
    lines = []
    for i, (name, in_vec, out_vec) in enumerate(results):
        lines.append(f"head {i}: {in_vec.tolist()} → {[round(v,3) for v in out_vec.tolist()]}")

    return fig, "".join(lines)


def get_layer_count():
    if _loaded_model is None:
        return 0
    return sum(1 for n, m in _loaded_model.named_modules()
               if hasattr(m, 'num_heads') or
               'attention' in type(m).__name__.lower())


# ------------------------------------------------------------------ #
#  Build UI                                                            #
# ------------------------------------------------------------------ #

def build_ui():
    with gr.Blocks(title="MIRAI — NN Debugger") as demo:

        gr.Markdown("# MIRAI — Neural Network Debugger")

        # ── model loader (shared across all tabs) ─────────────────
        with gr.Row():
            with gr.Column(scale=1):
                model_file   = gr.File(label="Model (.pt / .pth)",
                                       file_types=[".pt", ".pth"])
                load_btn     = gr.Button("Load", variant="primary")
            with gr.Column(scale=3):
                model_status = gr.Textbox(label="Model", lines=4,
                                          interactive=False)

        load_btn.click(fn=load_model,
                       inputs=model_file,
                       outputs=model_status)

        gr.Markdown("---")

        with gr.Tabs():

            # ── Tab 1: Equations ──────────────────────────────────
            with gr.Tab("Equations"):
                gr.Markdown("### Local linear equation for a given input")

                with gr.Row():
                    input_box  = gr.Textbox(
                        label="Input tensor (JSON)",
                        placeholder="e.g. [0.5]",
                        scale=3
                    )
                    probe_btn  = gr.Button("▶ Probe", variant="primary",
                                           scale=1)

                probe_status = gr.Textbox(label="Status", lines=1,
                                          interactive=False)

                gr.Markdown("#### End-to-end equation")
                ete_eq = gr.Markdown("_probe an input to see equation_")

                gr.Markdown("#### Per-neuron equations")
                with gr.Row():
                    neuron_dd     = gr.Dropdown(label="Select neuron",
                                                choices=[], scale=2)
                    neuron_status = gr.Textbox(label="Status", lines=1,
                                               interactive=False, scale=1)

                neuron_expanded  = gr.Markdown("_select neuron_")
                neuron_collapsed = gr.Markdown("")

                probe_btn.click(
                    fn=run_probe,
                    inputs=input_box,
                    outputs=[probe_status, ete_eq, neuron_dd]
                )
                neuron_dd.change(
                    fn=show_neuron_eq,
                    inputs=neuron_dd,
                    outputs=[neuron_status, neuron_expanded, neuron_collapsed]
                )

            # ── Tab 2: Function view ──────────────────────────────
            with gr.Tab("Function view"):
                gr.Markdown("### Model output vs true function — pockets colored by equation")

                with gr.Row():
                    with gr.Column(scale=1):
                        fn_dd    = gr.Dropdown(label="True function",
                                               choices=list(FUNCTIONS.keys()),
                                               value="sin(x)")
                        with gr.Row():
                            xmin_box = gr.Number(label="x min", value=-6.28)
                            xmax_box = gr.Number(label="x max", value=6.28)
                        npts_sl  = gr.Slider(label="Points", minimum=50,
                                             maximum=500, value=300, step=50)
                        with gr.Row():
                            plot_btn = gr.Button("▶ Plot", variant="primary")
                            save_btn = gr.Button("💾 Save")
                        fn_status = gr.Textbox(label="Status", lines=1,
                                               interactive=False)
                    with gr.Column(scale=3):
                        fn_plot = gr.Plot(label="Function comparison")

                plot_btn.click(
                    fn=run_function_plot,
                    inputs=[fn_dd, xmin_box, xmax_box, npts_sl],
                    outputs=[fn_plot, fn_status]
                )
                save_btn.click(
                    fn=save_function_plot,
                    inputs=[fn_dd, xmin_box, xmax_box, npts_sl],
                    outputs=fn_status
                )

            # ── Tab 3: Heads ─────────────────────────────────
            with gr.Tab("Heads"):
                gr.Markdown("### Head input → output mapping")
                gr.Markdown(
                    "Sweeps input across a range. "
                    "For each attention/projection head, plots what "
                    "that head outputs. One graph per head."
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        head_input = gr.Textbox(
                            label="Template input (JSON)",
                            placeholder="e.g. [1,0,1,0,1,0]",
                            lines=2
                        )
                        xrange_box = gr.Textbox(
                            label="Sweep range (min, max)",
                            value="-3.14, 3.14"
                        )
                        head_btn  = gr.Button("▶ Plot heads",
                                              variant="primary")
                        head_info = gr.Textbox(label="Status", lines=2,
                                               interactive=False)
                    with gr.Column(scale=3):
                        head_plot = gr.Plot(label="Head mappings")

                head_btn.click(
                    fn=analyze_heads,
                    inputs=[head_input, xrange_box],
                    outputs=[head_plot, head_info]
                )

        return demo


# ------------------------------------------------------------------ #
#  Entry point                                                         #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    build_ui().launch(share=False)