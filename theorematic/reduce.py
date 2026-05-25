"""Evaluate-preserving rewrites that shrink a circuit.

Three local rewrites, applied to fixed point by `reduce`:

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

The first two never change input / output width — they only touch hidden
slots. The third only drops layers strictly between layer 0 and the last
layer; the input and final-output interfaces stay intact.

Together these undo the identity padding that `parallel` introduces and
prune neurons that fixtures or hand-construction left as no-ops. The
guarantee is point-wise: `evaluate(reduce(net), x) == evaluate(net, x)`
for every legal `x` (see `tests/test_reduce.py`).
"""

from __future__ import annotations

import numpy as np

from theorematic.net import Layer


def _require_nonempty(layers: list[Layer]) -> None:
    if not layers:
        raise ValueError("layers must be non-empty")


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


def _signature(layers: list[Layer]) -> tuple:
    return tuple((l.W.shape, l.W.tobytes(), l.b.tobytes()) for l in layers)


def reduce(layers: list[Layer]) -> list[Layer]:
    """Apply all rewrites to fixed point.

    Iterates `drop_identity_layers`, `remove_always_zero_neurons`, and
    `remove_dead_neurons` until the layer list stops changing. The result
    is evaluate-equivalent to the input for every legal forward pass.
    """
    _require_nonempty(layers)
    cur = list(layers)
    while True:
        before = _signature(cur)
        cur = drop_identity_layers(cur)
        cur = remove_always_zero_neurons(cur)
        cur = remove_dead_neurons(cur)
        if _signature(cur) == before:
            return cur
