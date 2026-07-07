import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional
from functools import partial

from mirai.core.types import LayerRecord, LayerAnalysis, NeuronEquation, ProbeResult
from mirai.core.hookmanager import HookManager
from mirai.core.equationformatter import EquationFormatter


class ProbeEngine:
    def __init__(self, rank_threshold: float = 0.01):
        """
        :param rank_threshold: singular values below this fraction of the
                               largest singular value are considered zero
                               when computing effective rank.
        """
        self.rank_threshold = rank_threshold
        self.formatter = EquationFormatter()

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def probe(self, model: nn.Module, x: Tensor) -> ProbeResult:
        """
        Single entry point. Run one forward pass and return a fully
        populated ProbeResult.
        """
        model.eval()

        # ── 1. run forward pass through hook manager ──────────────────
        hm = HookManager(model, x)
        layer_records: dict[str, LayerRecord] = hm._hook_layer()

        # ── 2. full end-to-end jacobian ───────────────────────────────
        full_jacobian = self._full_jacobian(model, x)

        # ── 3. actual model output ────────────────────────────────────
        with torch.no_grad():
            output_tensor = model(x)

        # ── 4. per-layer analysis ─────────────────────────────────────
        layer_analyses: dict[str, LayerAnalysis] = {}
        neuron_equations: dict[str, list[NeuronEquation]] = {}
        layer_order = list(layer_records.keys())

        for layer_name, record in layer_records.items():
            analysis = self._analyse_layer(model, x, layer_name, record)
            layer_analyses[layer_name] = analysis

            equations = self._build_neuron_equations(layer_name, record, x)
            neuron_equations[layer_name] = equations

        # ── 5. region tracking ────────────────────────────────────────
        region_signature = self._region_signature(layer_records)
        global_boundary_dist = self._global_boundary_dist(layer_records)

        # ── 6. end-to-end collapsed equation as latex ─────────────────
        output_equation_latex = self.formatter.format_jacobian_eq(
            full_jacobian, x
        )

        return ProbeResult(
            input_tensor=x,
            layer_records=layer_records,
            layer_analyses=layer_analyses,
            full_jacobian=full_jacobian,
            output_tensor=output_tensor,
            region_signature=region_signature,
            global_boundary_dist=global_boundary_dist,
            neuron_equations=neuron_equations,
            output_equation_latex=output_equation_latex,
            model_name=type(model).__name__,
            layer_order=layer_order,
        )

    # ------------------------------------------------------------------ #
    #  Jacobian                                                            #
    # ------------------------------------------------------------------ #

    def _full_jacobian(self, model: nn.Module, x: Tensor) -> Tensor:
        """
        Full d(output) / d(input) via autograd.
        Shape: [output_dim, input_dim]
        """
        J = torch.autograd.functional.jacobian(model, x)
        if J.ndim == 0:   J = J.reshape(1, 1)
        elif J.ndim == 1: J = J.unsqueeze(0)
        return J

    def _partial_jacobian(
        self, model: nn.Module, x: Tensor, up_to_layer: str
    ) -> Tensor:
        """
        Jacobian of the network truncated just after `up_to_layer`.
        Builds a partial Sequential from layer_order up to that point.
        """
        layers = []
        for name, layer in model.named_modules():
            if name == "":
                continue
            layers.append(layer)
            if name == up_to_layer:
                break

        partial_model = nn.Sequential(*layers)
        J = torch.autograd.functional.jacobian(partial_model, x)
        if J.ndim == 0:   J = J.reshape(1, 1)
        elif J.ndim == 1: J = J.unsqueeze(0)
        return J

    # ------------------------------------------------------------------ #
    #  Per-layer analysis                                                  #
    # ------------------------------------------------------------------ #

    def _analyse_layer(
        self,
        model: nn.Module,
        x: Tensor,
        layer_name: str,
        record: LayerRecord,
    ) -> LayerAnalysis:

        # jacobian up to this layer
        jacobian = self._partial_jacobian(model, x, layer_name)

        # effective rank from weight SVD (only for layers with weights)
        effective_rank, singular_values = self._effective_rank(record.weight)

        # per-neuron distance to nearest ReLU boundary
        # dist_j = |z_j| / ||w_j||  where z_j is pre-activation
        boundary_distances, nearest_boundary = self._boundary_distances(record)

        return LayerAnalysis(
            jacobian=jacobian,
            effective_rank=effective_rank,
            singular_values=singular_values,
            boundary_distances=boundary_distances,
            nearest_boundary=nearest_boundary,
        )

    # ------------------------------------------------------------------ #
    #  Effective rank                                                      #
    # ------------------------------------------------------------------ #

    def _effective_rank(
        self, weight: Optional[Tensor]
    ) -> tuple[float, Tensor]:
        """
        Compute effective rank of a weight matrix via SVD.
        Effective rank = number of singular values above threshold * max_sv.
        Returns (effective_rank, all_singular_values).
        """
        if weight is None or weight.ndim < 2:
            empty = torch.tensor([])
            return 0.0, empty

        _, S, _ = torch.linalg.svd(weight, full_matrices=False)
        threshold = self.rank_threshold * S[0].item()   # fraction of largest sv
        effective_rank = float((S > threshold).sum().item())
        return effective_rank, S

    # ------------------------------------------------------------------ #
    #  Boundary distance                                                   #
    # ------------------------------------------------------------------ #

    def _boundary_distances(
        self, record: LayerRecord
    ) -> tuple[Tensor, float]:
        """
        For each neuron j, distance to the nearest ReLU boundary is:
            dist_j = |z_j| / ||w_j||
        where z_j is the pre-activation value and w_j is the weight row.
        Minimum across all neurons = distance to nearest boundary overall.
        """
        z = record.post_activation.float().flatten()

        if record.weight is None:
            inf_tensor = torch.full_like(z, float("inf"))
            return inf_tensor, float("inf")

        W = record.weight.float()
        # ||w_j|| for each neuron j
        w_norms = torch.linalg.norm(W, dim=1).clamp(min=1e-8)
        distances = z.abs() / w_norms

        return distances, distances.min().item()

    # ------------------------------------------------------------------ #
    #  Local linear equations per neuron                                  #
    # ------------------------------------------------------------------ #

    def _build_neuron_equations(
        self,
        layer_name: str,
        record: LayerRecord,
        x: Tensor,
    ) -> list[NeuronEquation]:
        """
        For each neuron j in this layer, compute its local linear equation:
            output_j = weight[j] @ x + bias[j]   if active
            output_j = 0                           if dead (ReLU killed it)
        Then pass to EquationFormatter for LaTeX rendering.
        """
        equations = []

        if record.weight is None:
            return equations   # activation layers (ReLU) have no weights

        W = record.weight.float()          # [num_neurons, input_dim]
        b = record.bias.float() if record.bias is not None else torch.zeros(W.shape[0])
        mask = record.relu_mask.flatten()  # [num_neurons]
        x_flat = x.float().flatten()

        for j in range(W.shape[0]):
            is_active = bool(mask[j].item())
            coeffs = W[j] if is_active else torch.zeros_like(W[j])
            bias_val = b[j].item() if is_active else 0.0

            latex_expanded, latex_collapsed = self.formatter.format_neuron_eq(
                coeffs, bias_val, is_active
            )

            equations.append(NeuronEquation(
                layer_name=layer_name,
                neuron_index=j,
                is_active=is_active,
                coefficients=coeffs,
                bias=bias_val,
                latex_expanded=latex_expanded,
                latex_collapsed=latex_collapsed,
            ))

        return equations

    # ------------------------------------------------------------------ #
    #  Region tracking                                                     #
    # ------------------------------------------------------------------ #

    def _region_signature(self, layer_records: dict[str, LayerRecord]) -> str:
        """
        Flatten all ReLU masks across all layers into a single binary string.
        e.g. "10110010" — uniquely identifies the linear region for this input.
        Each character is one neuron: 1 = active, 0 = dead.
        """
        bits = []
        for record in layer_records.values():
            bits.extend(record.relu_mask.flatten().int().tolist())
        return "".join(str(b) for b in bits)

    def _global_boundary_dist(
        self, layer_records: dict[str, LayerRecord]
    ) -> float:
        """
        Minimum boundary distance across all neurons in all layers.
        How much do we need to perturb x before ANY ReLU gate flips?
        """
        min_dist = float("inf")
        for record in layer_records.values():
            _, nearest = self._boundary_distances(record)
            if nearest < min_dist:
                min_dist = nearest
        return min_dist