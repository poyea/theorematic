"""Evaluate-preserving rewrites that shrink a circuit.

Two families of rewrite, each with its own driver.

**Structural** — `reduce(layers)` applies these to fixed point. They read
only the weights, so they hold for every input:

- `remove_dead_neurons`        — drop hidden units whose column in the next
                                  layer's W is all zero (their activation
                                  never reaches the output).
- `remove_always_zero_neurons` — drop hidden units whose row in W is all zero
                                  and whose bias is <= 0 (ReLU clamps them
                                  to 0 for every input).
- `drop_identity_layers`       — drop layers in the interior that are
                                  W=I, b=0 (a no-op when the input has
                                  already passed through a ReLU, which it has
                                  for any non-input layer).

**Bound-driven** — `reduce_with_bounds(layers, input_lo, input_hi)` applies
the structural passes *plus* these, which need declared input bounds and
hold only for inputs inside them:

- `remove_unreachable_neurons` — drop hidden units whose pre-activation
                                  upper bound is <= 0 (they never fire).
- `fuse_linear_layers`         — merge a layer into its successor when the
                                  ReLU between them is provably the identity.

Bounds come from `net.preact_bounds`, a sound over-approximation. Sound
means no false positives: a neuron it calls unreachable really is. The
converse fails — interval arithmetic misses correlations, so some genuinely
dead neurons survive.

The equivalence guarantee is point-wise. For `reduce` it holds for every
legal `x`; for `reduce_with_bounds` it holds for `x` within the declared
bounds, and outside them the reduced net may differ. Both are asserted
exhaustively in `tests/test_reduce.py`.

The bound-driven passes carry one further precondition: the net's weights
must be small enough that neither its own forward pass nor a fused weight
product overflows int64. `evaluate` wraps silently on overflow, which would
make its ReLU decisions disagree with what these bounds describe. Both
passes detect that regime and become no-ops in it rather than emitting a
net that is not equivalent. See `_bounds_are_exact`.
"""

from __future__ import annotations

import numpy as np

from theorematic.net import Layer, preact_bounds

# float64 represents integers exactly only up to 2**53, and `preact_bounds`
# works in float64. Past this magnitude a bound is no longer a reliable
# statement about an integer pre-activation, so the bound-driven passes
# decline to act on it rather than acting on noise.
_EXACT_FLOAT_INT = 2**53


def _require_nonempty(layers: list[Layer]) -> None:
    if not layers:
        raise ValueError("layers must be non-empty")


def _bounds_are_exact(bounds: list[tuple[np.ndarray, np.ndarray]]) -> bool:
    """Are the propagated bounds trustworthy statements about integer preacts?

    Checked across the *whole* net, not per layer. A net whose bounds run past
    this magnitude is one whose own forward pass can overflow int64, and
    `evaluate` wraps silently when it does. Its ReLU decisions are then made
    on wrapped values while these bounds describe the unwrapped ones, so no
    bound-driven rewrite can be sound anywhere in it — including at layers
    whose own bounds look small.
    """
    return all(
        bool(np.all(np.abs(z_lo) <= _EXACT_FLOAT_INT) and np.all(np.abs(z_hi) <= _EXACT_FLOAT_INT))
        for z_lo, z_hi in bounds
    )


def _fits_int64(values: np.ndarray) -> bool:
    info = np.iinfo(np.int64)
    return bool(np.all(values >= int(info.min)) and np.all(values <= int(info.max)))


def remove_dead_neurons(layers: list[Layer]) -> list[Layer]:
    """Drop hidden units that the next layer ignores.

    For each hidden layer i (i < len-1), a unit k is dead iff column k of
    layer[i+1].W is all zeros — the next layer's pre-activation does not
    depend on it. Removing it preserves the forward pass exactly.
    """
    _require_nonempty(layers)
    out = list(layers)
    for i in range(len(out) - 1):
        keep = ~np.all(out[i + 1].W == 0, axis=0)
        if keep.all():
            continue
        out[i] = Layer(W=out[i].W[keep, :], b=out[i].b[keep])
        out[i + 1] = Layer(W=out[i + 1].W[:, keep], b=out[i + 1].b)
    return out


def remove_always_zero_neurons(layers: list[Layer]) -> list[Layer]:
    """Drop hidden units that emit 0 for every input.

    A unit with W row all zero and bias <= 0 has pre-activation <= 0, so
    ReLU emits 0 unconditionally. Such a unit contributes nothing to the
    next layer and can be dropped along with its column downstream.

    Only applied to hidden layers (i < len-1); the final layer is linear
    so a zero-row there is a real (constant) output, not a discard target.
    """
    _require_nonempty(layers)
    out = list(layers)
    for i in range(len(out) - 1):
        W, b = out[i].W, out[i].b
        dead = np.all(W == 0, axis=1) & (b <= 0)
        if not dead.any():
            continue
        keep = ~dead
        out[i] = Layer(W=W[keep, :], b=b[keep])
        out[i + 1] = Layer(W=out[i + 1].W[:, keep], b=out[i + 1].b)
    return out


def _is_identity_layer(layer: Layer) -> bool:
    n_out, n_in = layer.W.shape
    if n_out != n_in:
        return False
    return np.array_equal(layer.W, np.eye(n_in, dtype=layer.W.dtype)) and not layer.b.any()


def drop_identity_layers(layers: list[Layer]) -> list[Layer]:
    """Drop interior identity layers (W=I, b=0).

    Safe in the interior because the input to a non-zero-indexed layer is
    the ReLU output of the previous layer and is therefore non-negative;
    `relu(I @ h + 0) == h` exactly. Not safe at position 0 (input may be
    negative) or at the last position (final layer is linear, so an
    identity there would otherwise contribute a ReLU that gets dropped).
    """
    _require_nonempty(layers)
    out: list[Layer] = []
    last = len(layers) - 1
    for i, layer in enumerate(layers):
        if 0 < i < last and _is_identity_layer(layer):
            continue
        out.append(layer)
    return out


def remove_unreachable_neurons(
    layers: list[Layer],
    input_lo: int | np.ndarray,
    input_hi: int | np.ndarray,
) -> list[Layer]:
    """Drop hidden units that never fire for inputs within the given bounds.

    A unit whose pre-activation upper bound is <= 0 emits 0 from its ReLU for
    every input in `[input_lo, input_hi]`, so it and its downstream column
    can go. This strictly subsumes `remove_always_zero_neurons`: a zero row
    with bias <= 0 has `z_hi = b <= 0` under any bounds.

    Equivalence holds only inside the declared bounds, and only for nets whose
    bounds stay in float64's exact-integer range — see `_bounds_are_exact`.
    Outside that range this is a no-op rather than a guess.
    """
    _require_nonempty(layers)
    bounds = preact_bounds(layers, input_lo, input_hi)
    if not _bounds_are_exact(bounds):
        return list(layers)
    out = list(layers)
    for i in range(len(out) - 1):
        keep = bounds[i][1] > 0
        if keep.all():
            continue
        out[i] = Layer(W=out[i].W[keep, :], b=out[i].b[keep])
        out[i + 1] = Layer(W=out[i + 1].W[:, keep], b=out[i + 1].b)
    return out


def fuse_linear_layers(
    layers: list[Layer],
    input_lo: int | np.ndarray,
    input_hi: int | np.ndarray,
) -> list[Layer]:
    """Merge a layer into its successor where the ReLU between them is identity.

    If every pre-activation of layer `i` is provably >= 0 under the declared
    bounds, its ReLU is a no-op and the two affine maps compose exactly:

        W' = W[i+1] @ W[i]        b' = W[i+1] @ b[i] + b[i+1]

    Integer weights are closed under this product, so the result is still a
    valid `Layer`. Only the first eligible pair is fused per call — the
    composed layer changes the bounds downstream of it, so the driver
    recomputes and calls again.

    Two magnitude guards, because fusion *multiplies* weights and so grows
    them far faster than a forward pass does:

    - The net's bounds must be exact in float64 (see `_bounds_are_exact`).
      Past `2**53` a bound is noise, not a proof that the ReLU is identity.
    - The composed weights must fit in int64. The product is computed in
      exact Python integers and the fusion is declined if it would not fit,
      because `numpy` would wrap it silently and the rewrite would stop
      preserving `evaluate`.

    Equivalence holds only inside the declared bounds.
    """
    _require_nonempty(layers)
    bounds = preact_bounds(layers, input_lo, input_hi)
    if not _bounds_are_exact(bounds):
        return list(layers)
    for i in range(len(layers) - 1):
        if not np.all(bounds[i][0] >= 0):
            continue
        cur, nxt = layers[i], layers[i + 1]
        # object dtype gives arbitrary-precision Python ints, so the overflow
        # check happens before any wraparound can occur.
        W = nxt.W.astype(object) @ cur.W.astype(object)
        b = nxt.W.astype(object) @ cur.b.astype(object) + nxt.b.astype(object)
        if not (_fits_int64(W) and _fits_int64(b)):
            continue
        fused = Layer(W=W.astype(np.int64), b=b.astype(np.int64))
        return layers[:i] + [fused] + layers[i + 2 :]
    return list(layers)


def _signature(layers: list[Layer]) -> tuple:
    return tuple((l.W.shape, l.W.tobytes(), l.b.tobytes()) for l in layers)


def _fixed_point(layers: list[Layer], passes: list) -> list[Layer]:
    cur = list(layers)
    while True:
        before = _signature(cur)
        for apply_pass in passes:
            cur = apply_pass(cur)
        if _signature(cur) == before:
            return cur


_STRUCTURAL = [drop_identity_layers, remove_always_zero_neurons, remove_dead_neurons]


def reduce(layers: list[Layer]) -> list[Layer]:
    """Apply the structural rewrites to fixed point.

    The result is evaluate-equivalent to the input for every legal forward
    pass. For the stronger bound-driven passes, see `reduce_with_bounds`.
    """
    _require_nonempty(layers)
    return _fixed_point(layers, _STRUCTURAL)


def reduce_with_bounds(
    layers: list[Layer],
    input_lo: int | np.ndarray,
    input_hi: int | np.ndarray,
) -> list[Layer]:
    """Apply the structural *and* bound-driven rewrites to fixed point.

    Equivalent to `reduce` for inputs within `[input_lo, input_hi]`, and
    typically smaller. Outside those bounds the result may differ from the
    original — the bounds are part of the contract, not a hint.
    """
    _require_nonempty(layers)
    passes = _STRUCTURAL + [
        lambda ls: remove_unreachable_neurons(ls, input_lo, input_hi),
        lambda ls: fuse_linear_layers(ls, input_lo, input_hi),
    ]
    return _fixed_point(layers, passes)
