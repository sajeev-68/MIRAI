from dataclasses import dataclass, field
from typing import Optional
import torch
from torch import Tensor


@dataclass
class LayerRecord:
    """Raw data captured by hooks for a single layer."""
    pre_activation: Tensor          # z = Wx + b, before ReLU
    post_activation: Tensor         # a = ReLU(z), after activation
    relu_mask: Tensor               # bool — which neurons fired (pre > 0)
    weight: Optional[Tensor]        # layer weight matrix if available
    bias: Optional[Tensor]          # layer bias if available


@dataclass
class LayerAnalysis:
    """Math computed by probe_engine for a single layer."""
    jacobian: Tensor                # d(layer_output) / d(model_input)
    effective_rank: float           # how many singular values are non-negligible
    singular_values: Tensor         # full SVD spectrum
    boundary_distances: Tensor      # per-neuron distance to nearest ReLU boundary
    nearest_boundary: float         # min across all neurons — how close to a region edge


@dataclass
class NeuronEquation:
    """Human-readable equation for a single neuron."""
    layer_name: str
    neuron_index: int
    is_active: bool                 # did ReLU fire for this input?
    coefficients: Tensor            # one coefficient per input dimension
    bias: float
    latex_expanded: str             # e.g. "0.43x_0 - 0.21x_1 + 0.07"
    latex_collapsed: str            # e.g. "0.31x - 0.18" (if 1D input)


@dataclass
class ProbeResult:
    """
    Single object returned by probe_engine.probe(model, x).
    Contains everything the Gradio frontend needs to render.
    """

    # --- input ---
    input_tensor: Tensor

    # --- per-layer raw data (from hook_manager) ---
    layer_records: dict[str, LayerRecord]           # keyed by layer name

    # --- per-layer analysis (from probe_engine) ---
    layer_analyses: dict[str, LayerAnalysis]        # keyed by layer name

    # --- end-to-end ---
    full_jacobian: Tensor                           # d(output) / d(input), full network
    output_tensor: Tensor                           # final model output

    # --- region ---
    region_signature: str                           # e.g. "101101" — binary relu mask flattened
    global_boundary_dist: float                     # nearest boundary across all layers

    # --- equations (from equation_formatter) ---
    neuron_equations: dict[str, list[NeuronEquation]]   # layer_name → list of neuron eqs
    output_equation_latex: str                          # collapsed end-to-end eq as LaTeX

    # --- metadata ---
    model_name: str = ""
    layer_order: list[str] = field(default_factory=list)   # layers in forward-pass order