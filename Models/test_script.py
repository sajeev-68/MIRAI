"""
test_debugger.py

Run this from the project root:
    python test_debugger.py

Tests each component in isolation then end-to-end.
No external test framework needed — just plain asserts.
"""

import torch
import torch.nn as nn

# ── import package ────────────────────────────────────────────────────────────
from mirai.core import (
    HookManager,
    EquationFormatter,
    ProbeEngine,
    ProbeResult,
)


# ------------------------------------------------------------------ #
#  Shared fixture — small MLP                                         #
# ------------------------------------------------------------------ #

def make_model() -> nn.Module:
    """
    Simple 3-layer MLP:
        Linear(4 → 8) → ReLU → Linear(8 → 4) → ReLU → Linear(4 → 2)
    Small enough to inspect by hand.
    """
    torch.manual_seed(42)
    return nn.Sequential(
        nn.Linear(4, 8),
        nn.ReLU(),
        nn.Linear(8, 4),
        nn.ReLU(),
        nn.Linear(4, 2),
    )


def make_input() -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(4)


# ------------------------------------------------------------------ #
#  Test 1 — HookManager                                               #
# ------------------------------------------------------------------ #

def test_hook_manager():
    print("── test_hook_manager ──────────────────────────────────")
    model = make_model()
    x = make_input()

    hm = HookManager(model, x)
    records = hm._hook_layer()

    # should have one record per named module (excluding top-level)
    assert len(records) > 0, "No LayerRecords captured"

    for name, record in records.items():
        print(f"  {name}")
        print(f"    pre_activation : {record.pre_activation.shape}")
        print(f"    post_activation: {record.post_activation.shape}")
        print(f"    relu_mask      : {record.relu_mask.shape}")
        print(f"    weight         : {record.weight.shape if record.weight is not None else None}")
        print(f"    bias           : {record.bias.shape if record.bias is not None else None}")

        # pre/post shapes differ for Linear layers (in_dim vs out_dim) — expected
        # relu_mask must match post_activation shape (one bool per output neuron)
        assert record.relu_mask.shape == record.post_activation.shape, \
            f"Mask shape mismatch in {name}: " \
            f"mask {record.relu_mask.shape} vs post {record.post_activation.shape}"

    print("  ✓ passed\n")


# ------------------------------------------------------------------ #
#  Test 2 — EquationFormatter                                         #
# ------------------------------------------------------------------ #

def test_equation_formatter():
    print("── test_equation_formatter ────────────────────────────")
    fmt = EquationFormatter()

    # active neuron — 1D input
    coeffs_1d = torch.tensor([0.5])
    exp, col = fmt.format_neuron_eq(coeffs_1d, bias=0.25, is_active=True)
    print(f"  1D active   expanded : {exp}")
    print(f"  1D active   collapsed: {col}")
    assert exp != "", "Expanded eq should not be empty"
    assert col != "", "Collapsed eq should not be empty"

    # active neuron — multi-dim input
    coeffs_nd = torch.tensor([0.43, -0.21, 0.07])
    exp, col = fmt.format_neuron_eq(coeffs_nd, bias=0.1, is_active=True)
    print(f"  ND active   expanded : {exp}")
    print(f"  ND active   collapsed: {col}")

    # dead neuron
    exp, col = fmt.format_neuron_eq(coeffs_nd, bias=0.1, is_active=False)
    print(f"  dead neuron expanded : {exp}")
    assert "dead" in exp.lower() or "0" in exp, "Dead neuron should show 0"

    print("  ✓ passed\n")


# ------------------------------------------------------------------ #
#  Test 3 — ProbeEngine end-to-end                                    #
# ------------------------------------------------------------------ #

def test_probe_engine():
    print("── test_probe_engine ──────────────────────────────────")
    model = make_model()
    x = make_input()

    engine = ProbeEngine()
    result: ProbeResult = engine.probe(model, x)

    # basic structure
    assert result.input_tensor is not None
    assert result.output_tensor is not None
    assert len(result.layer_records) > 0
    assert len(result.layer_analyses) > 0
    assert len(result.neuron_equations) > 0
    assert len(result.region_signature) > 0
    assert result.global_boundary_dist >= 0.0

    print(f"  model name       : {result.model_name}")
    print(f"  input shape      : {result.input_tensor.shape}")
    print(f"  output shape     : {result.output_tensor.shape}")
    print(f"  layers captured  : {len(result.layer_records)}")
    print(f"  region signature : {result.region_signature}")
    print(f"  boundary dist    : {result.global_boundary_dist:.6f}")
    print(f"  full jacobian    : {result.full_jacobian.shape}")
    print(f"  output eq latex  :\n    {result.output_equation_latex}")

    # per-layer
    print("\n  Per-layer analysis:")
    for name, analysis in result.layer_analyses.items():
        print(f"    {name}")
        print(f"      effective_rank   : {analysis.effective_rank}")
        print(f"      nearest_boundary : {analysis.nearest_boundary:.6f}")
        print(f"      jacobian shape   : {analysis.jacobian.shape}")

    # neuron equations — print first active and first dead
    print("\n  Sample neuron equations:")
    for layer_name, eqs in result.neuron_equations.items():
        active = [e for e in eqs if e.is_active]
        dead   = [e for e in eqs if not e.is_active]
        if active:
            e = active[0]
            print(f"    [ACTIVE] {layer_name}/neuron {e.neuron_index}")
            print(f"      expanded : {e.latex_expanded}")
            print(f"      collapsed: {e.latex_collapsed}")
        if dead:
            e = dead[0]
            print(f"    [DEAD]   {layer_name}/neuron {e.neuron_index}")
            print(f"      expanded : {e.latex_expanded}")

    print("  ✓ passed\n")


# ------------------------------------------------------------------ #
#  Test 4 — region changes with different inputs                      #
# ------------------------------------------------------------------ #

def test_region_changes():
    print("── test_region_changes ────────────────────────────────")
    model = make_model()
    engine = ProbeEngine()

    signatures = set()
    for i in range(10):
        x = torch.randn(4)
        result = engine.probe(model, x)
        signatures.add(result.region_signature)

    print(f"  10 random inputs → {len(signatures)} distinct linear regions")
    assert len(signatures) > 1, \
        "All inputs landed in same region — unlikely for random inputs"
    print("  ✓ passed\n")


# ------------------------------------------------------------------ #
#  Run all                                                            #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    print("\n" + "═" * 55)
    print("  NN Debugger — test suite")
    print("═" * 55 + "\n")

    test_hook_manager()
    test_equation_formatter()
    test_probe_engine()
    test_region_changes()

    print("═" * 55)
    print("  All tests passed ✓")
    print("═" * 55 + "\n")