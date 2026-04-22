"""MILP encoding of integer ReLU nets.

A ReLU unit `y = relu(z)` becomes four linear constraints plus one binary
indicator `a`:

    y >= 0
    y >= z
    y <= z + M * (1 - a)
    y <= M * a

where `M` is an upper bound on `|z|` on the feasible region. With the
indicator, any MILP solver can invert the network — find an input that drives
the output to a chosen value.

Big-M is not a magic constant. Too small → you cut off real solutions. Too
large → numerically flabby and slow. We propagate interval bounds through
the network given input bounds and derive a tight M per neuron.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pulp

from theorematic.net import Layer


@dataclass(frozen=True)
class InvertResult:
    x: np.ndarray
    status: str

    @property
    def feasible(self) -> bool:
        return self.status == "Optimal"


def preact_bounds(
    layers: list[Layer], input_lo: np.ndarray, input_hi: np.ndarray
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Interval-propagate `[lo, hi]` through the net; return per-layer preact bounds.

    After each hidden layer the post-activation bounds are clipped at zero
    before propagating onward.
    """
    lo, hi = input_lo.astype(float), input_hi.astype(float)
    out: list[tuple[np.ndarray, np.ndarray]] = []
    last = len(layers) - 1
    for i, layer in enumerate(layers):
        W = layer.W.astype(float)
        b = layer.b.astype(float)
        Wp, Wn = np.maximum(W, 0), np.minimum(W, 0)
        z_lo = Wp @ lo + Wn @ hi + b
        z_hi = Wp @ hi + Wn @ lo + b
        out.append((z_lo, z_hi))
        if i != last:
            lo, hi = np.maximum(z_lo, 0), np.maximum(z_hi, 0)
    return out


def invert(
    layers: list[Layer],
    target: list[float | None],
    *,
    input_lo: int = 0,
    input_hi: int = 1,
    input_integer: bool = True,
    solver_msg: bool = False,
) -> InvertResult:
    """Find an input `x` such that `evaluate(layers, x) == target`.

    `target` is a list with one entry per output; `None` means "don't care".
    `input_lo` / `input_hi` bound every input coordinate; `input_integer`
    restricts them to integers (the usual case for reverse engineering).
    """
    n_in = layers[0].in_features
    n_out = layers[-1].out_features
    if len(target) != n_out:
        raise ValueError(f"target has {len(target)} entries, net emits {n_out}")

    lo_vec = np.full(n_in, input_lo, dtype=float)
    hi_vec = np.full(n_in, input_hi, dtype=float)
    bounds = preact_bounds(layers, lo_vec, hi_vec)

    prob = pulp.LpProblem("invert", pulp.LpMinimize)
    cat = "Integer" if input_integer else "Continuous"
    x = [
        pulp.LpVariable(f"x_{i}", lowBound=input_lo, upBound=input_hi, cat=cat)
        for i in range(n_in)
    ]

    h: list = list(x)
    last = len(layers) - 1
    for li, layer in enumerate(layers):
        W = layer.W
        b = layer.b
        z = [
            pulp.lpSum(int(W[j, k]) * h[k] for k in range(layer.in_features)) + int(b[j])
            for j in range(layer.out_features)
        ]
        if li == last:
            h = z
            continue
        z_lo, z_hi = bounds[li]
        new_h = []
        for j in range(layer.out_features):
            # M must upper-bound both |z_lo| and |z_hi|; +1 for slack.
            M = float(max(abs(z_lo[j]), abs(z_hi[j]))) + 1.0
            y = pulp.LpVariable(f"y_{li}_{j}", lowBound=0)
            a = pulp.LpVariable(f"a_{li}_{j}", cat="Binary")
            prob += y >= z[j]
            prob += y <= z[j] + M * (1 - a)
            prob += y <= M * a
            new_h.append(y)
        h = new_h

    for j, t in enumerate(target):
        if t is not None:
            prob += h[j] == t

    prob += 0  # feasibility only — any solution is fine
    status_code = prob.solve(pulp.PULP_CBC_CMD(msg=solver_msg))
    status = pulp.LpStatus[status_code]

    if status != "Optimal":
        return InvertResult(x=np.array([]), status=status)

    dtype = int if input_integer else float

    def _extract(v: pulp.LpVariable) -> float:
        # CBC can return None for variables it didn't bind; default to the lower bound.
        raw = v.value()
        return float(input_lo) if raw is None else float(raw)

    values = np.array(
        [int(round(_extract(v))) if input_integer else _extract(v) for v in x], dtype=dtype
    )
    return InvertResult(x=values, status=status)
