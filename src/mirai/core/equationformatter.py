import torch
from torch import Tensor
import sympy as sp


class EquationFormatter:
    """
    Takes raw coefficient tensors from ProbeEngine and formats them
    into human-readable LaTeX strings using sympy.

    Two outputs per neuron:
      - latex_expanded : shows every input term explicitly
                         e.g. "0.43x_0 - 0.21x_1 + 0.07"
      - latex_collapsed: single Ax + b form (only meaningful for 1D input)
                         e.g. "0.31x - 0.18"
    """

    def __init__(self, precision: int = 4):
        """
        :param precision: decimal places to round coefficients to
        """
        self.precision = precision

    # ------------------------------------------------------------------ #
    #  Neuron equation                                                     #
    # ------------------------------------------------------------------ #

    def format_neuron_eq(
        self,
        coefficients: Tensor,   # shape [input_dim] — one coeff per input feature
        bias: float,
        is_active: bool,
    ) -> tuple[str, str]:
        """
        Format the local linear equation for one neuron.
        Returns (latex_expanded, latex_collapsed).
        """
        if not is_active:
            return r"0 \quad \text{(dead — ReLU)}", r"0"

        coeffs = coefficients.float().flatten().tolist()
        coeffs_r = [round(c, self.precision) for c in coeffs]
        bias_r = round(bias, self.precision)

        # ── expanded: sum of coefficient * x_i terms ──────────────────
        expanded = self._build_expanded(coeffs_r, bias_r)

        # ── collapsed: if 1D input just write ax + b ──────────────────
        if len(coeffs_r) == 1:
            collapsed = self._build_collapsed_1d(coeffs_r[0], bias_r)
        else:
            # for multi-dim input, collapsed = sympy simplification
            collapsed = self._sympy_simplify(coeffs_r, bias_r)

        return expanded, collapsed

    # ------------------------------------------------------------------ #
    #  Full jacobian equation (end-to-end)                                #
    # ------------------------------------------------------------------ #

    def format_jacobian_eq(
        self,
        jacobian: Tensor,   # shape [output_dim, input_dim]
        x: Tensor,
    ) -> str:
        """
        Format the end-to-end linear equation from the full Jacobian.
        For each output dimension i:
            output_i = J[i,0]*x_0 + J[i,1]*x_1 + ... + c_i
        Returns a LaTeX string with one equation per output.
        """
        J = jacobian.float()

        # normalize to 2D [output_dim, input_dim]
        if J.ndim == 0:   J = J.reshape(1, 1)
        elif J.ndim == 1: J = J.unsqueeze(0)

        lines = []
        for i, row in enumerate(J):
            coeffs = [round(c, self.precision) for c in row.tolist()]
            # no bias available from jacobian alone — just the linear part
            eq = self._build_expanded(coeffs, bias=0.0, var_prefix="x")
            lines.append(rf"y_{{{i}}} = {eq}")

        return r" \\ ".join(lines)

    # ------------------------------------------------------------------ #
    #  Layer weight matrix as latex                                       #
    # ------------------------------------------------------------------ #

    def format_weight_matrix(self, weight: Tensor) -> str:
        """
        Format a weight matrix as a LaTeX bmatrix.
        Useful for displaying a full layer's transformation.
        """
        W = weight.float()
        rows = []
        for row in W:
            entries = " & ".join(
                str(round(v, self.precision)) for v in row.tolist()
            )
            rows.append(entries)
        inner = r" \\ ".join(rows)
        return rf"\begin{{bmatrix}} {inner} \end{{bmatrix}}"

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _build_expanded(
        self,
        coeffs: list[float],
        bias: float,
        var_prefix: str = "x",
    ) -> str:
        """
        Build a LaTeX sum string:
            c_0 x_0 + c_1 x_1 + ... + bias
        Skips zero terms. Handles sign formatting cleanly.
        """
        terms = []

        for i, c in enumerate(coeffs):
            if c == 0.0:
                continue
            var = rf"{var_prefix}_{{{i}}}"
            terms.append(self._format_term(c, var, first=len(terms) == 0))

        # bias term
        if bias != 0.0 or len(terms) == 0:
            terms.append(self._format_const(bias, first=len(terms) == 0))

        return "".join(terms)

    def _build_collapsed_1d(self, coeff: float, bias: float) -> str:
        """
        For scalar input: format as ax + b.
        """
        x = sp.Symbol("x")
        expr = sp.Rational(coeff).limit_denominator(1000) * x + \
               sp.Rational(bias).limit_denominator(1000)
        return sp.latex(expr)

    def _sympy_simplify(self, coeffs: list[float], bias: float) -> str:
        """
        For multi-dim input, build a sympy expression and simplify.
        Variables are x0, x1, x2, ...
        """
        syms = [sp.Symbol(f"x{i}") for i in range(len(coeffs))]
        expr = sum(
            sp.Rational(c).limit_denominator(1000) * s
            for c, s in zip(coeffs, syms)
        ) + sp.Rational(bias).limit_denominator(1000)
        return sp.latex(sp.simplify(expr))

    def _format_term(self, coeff: float, var: str, first: bool) -> str:
        """
        Format one coefficient * variable term with correct sign spacing.
        """
        c = round(coeff, self.precision)
        if c == 1.0:
            coeff_str = ""
        elif c == -1.0:
            coeff_str = "-"
        else:
            coeff_str = str(abs(c))

        term = rf"{coeff_str}{var}"

        if first:
            return f"-{term}" if c < 0 else term
        else:
            return rf" - {term}" if c < 0 else rf" + {term}"

    def _format_const(self, val: float, first: bool) -> str:
        """
        Format a constant (bias) term with correct sign spacing.
        """
        v = round(val, self.precision)
        if first:
            return str(v)
        return rf" - {abs(v)}" if v < 0 else rf" + {v}"