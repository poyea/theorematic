"""SMT encoding of integer ReLU nets via z3.

The same inversion problem as `theorematic.ilp`, encoded differently. Where
ILP linearises ReLU with a binary indicator + big-M, z3 handles the
non-linearity natively:

    y == If(z >= 0, z, 0)

No bound propagation, no big-M, no risk of cutting off feasible regions with
an undersized constant. The trade is that z3's integer theory is more
powerful but in general slower per call; the practical pictures of the two
solvers on the same fixtures is the lesson.

Interface deliberately mirrors `ilp.invert`: same arguments, same return
shape, same post-solve verification.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import z3

from theorematic.errors import VerificationError
from theorematic.net import Layer, evaluate

__all__ = ["InvertResult", "VerificationError", "invert"]


@dataclass(frozen=True)
class InvertResult:
    x: np.ndarray
    status: str  # "sat", "unsat", or "unknown"

    @property
    def feasible(self) -> bool:
        return self.status == "sat"


def _linear_expr(W_row: np.ndarray, b_j: int, h: list) -> object:
    """Build the integer expression `sum(W_row[k] * h[k]) + b_j`.

    Skips zero coefficients to keep the SMT formula sparse.
    """
    terms = [int(c) * h[k] for k, c in enumerate(W_row) if c != 0]
    if not terms:
        return z3.IntVal(int(b_j))
    expr = terms[0]
    for t in terms[1:]:
        expr = expr + t
    return expr + int(b_j)


def invert(
    layers: list[Layer],
    target: list[float | None],
    *,
    input_lo: int = 0,
    input_hi: int = 1,
    timeout_ms: int | None = None,
) -> InvertResult:
    """Find an input `x` such that `evaluate(layers, x) == target` using z3.

    `target` is one entry per output; `None` means "don't care". `input_lo`
    and `input_hi` bound every input coordinate. Inputs are integer (z3 has
    a native integer theory; the float case from ILP has no analogue here
    and isn't worth bolting on).

    `timeout_ms` is forwarded to the z3 solver; on timeout the result has
    status `"unknown"` and an empty `x`.
    """
    if not layers:
        raise ValueError("layers must be non-empty")
    if input_lo > input_hi:
        raise ValueError(f"input_lo ({input_lo}) > input_hi ({input_hi})")
    n_out = layers[-1].out_features
    if len(target) != n_out:
        raise ValueError(f"target has {len(target)} entries, net emits {n_out}")

    solver = z3.Solver()
    if timeout_ms is not None:
        solver.set("timeout", int(timeout_ms))

    n_in = layers[0].in_features
    x = [z3.Int(f"x_{i}") for i in range(n_in)]
    for xi in x:
        solver.add(xi >= int(input_lo), xi <= int(input_hi))

    h: list = list(x)
    last = len(layers) - 1
    for li, layer in enumerate(layers):
        z_exprs = [_linear_expr(layer.W[j], int(layer.b[j]), h) for j in range(layer.out_features)]
        if li == last:
            h = z_exprs
            continue
        new_h: list = []
        for j, z_j in enumerate(z_exprs):
            y = z3.Int(f"y_{li}_{j}")
            solver.add(y == z3.If(z_j >= 0, z_j, 0))
            new_h.append(y)
        h = new_h

    for j, t in enumerate(target):
        if t is not None:
            solver.add(h[j] == int(t))

    result = solver.check()
    status = str(result)  # "sat", "unsat", or "unknown"
    if result != z3.sat:
        return InvertResult(x=np.array([], dtype=int), status=status)

    model = solver.model()
    values = np.array([model.eval(xi, model_completion=True).as_long() for xi in x], dtype=int)

    # Post-solve verification: same safety net as ilp.invert. A z3 "sat"
    # answer is only meaningful if the assignment actually drives the net to
    # the target — a mismatch means the encoding is wrong.
    actual = evaluate(layers, values)
    for j, t in enumerate(target):
        if t is None:
            continue
        if int(actual[j]) != int(t):
            raise VerificationError(
                f"z3 returned x={values.tolist()} but evaluate yields "
                f"output[{j}]={int(actual[j])} (target={int(t)}). The SMT "
                f"encoding is inconsistent with the network."
            )

    return InvertResult(x=values, status=status)
