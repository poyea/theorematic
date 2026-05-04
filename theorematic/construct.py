"""Builder primitives for hand-constructing integer ReLU circuits.

The four primitives here cover the patterns that show up over and over in
`fixtures.py`:

- `linear(W, b)`           — typed `Layer` from array-likes
- `route(sources, n_in)`   — one layer that selects / permutes / copies inputs
- `stack(*circuits)`       — sequential composition (with shape checks)
- `parallel(*circuits)`    — independent circuits on disjoint input slices

Together they let you build a multi-bit equality net from a single-bit XNOR
gate plus a routing layer plus a threshold AND, instead of weaving indices by
hand. The architectural payoff: writing the next fixture is shorter than
writing the last one, and shape mismatches surface at construction time.

`parallel` auto-pads shorter branches with identity layers so all branches
end at the same depth. That assumes intermediate values are non-negative
(identity-then-ReLU == identity for x >= 0), which holds for the
boolean-valued sub-circuits we compose here.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from theorematic.net import Layer


def linear(W, b) -> Layer:
    """`Layer` from array-likes; casts to integer dtype.

    Convenience around `Layer(W=..., b=...)` that accepts nested lists and
    avoids the `dtype=int` ceremony at every call site.
    """
    W_arr = np.asarray(W, dtype=int)
    b_arr = np.asarray(b, dtype=int)
    return Layer(W=W_arr, b=b_arr)


def route(sources: Sequence[int], n_in: int) -> Layer:
    """Single layer that routes inputs to outputs by index.

    `output[i] = input[sources[i]]`. Used for permutations, copies (an index
    appearing twice), and projections (some inputs not appearing at all).
    """
    n_out = len(sources)
    W = np.zeros((n_out, n_in), dtype=int)
    for i, j in enumerate(sources):
        if not 0 <= j < n_in:
            raise ValueError(f"route source {j} out of range [0, {n_in})")
        W[i, j] = 1
    return Layer(W=W, b=np.zeros(n_out, dtype=int))


def stack(*circuits: list[Layer]) -> list[Layer]:
    """Sequential composition. Output width of each circuit must match the
    input width of the next.
    """
    out: list[Layer] = []
    prev_out: int | None = None
    for c in circuits:
        if not c:
            continue
        if prev_out is not None and prev_out != c[0].in_features:
            raise ValueError(
                f"stack: width mismatch — previous circuit emits {prev_out}, "
                f"next expects {c[0].in_features}"
            )
        out.extend(c)
        prev_out = c[-1].out_features
    return out


def _block_diag(*matrices: np.ndarray) -> np.ndarray:
    """Block-diagonal concatenation of integer matrices."""
    if not matrices:
        return np.zeros((0, 0), dtype=int)
    rows = sum(M.shape[0] for M in matrices)
    cols = sum(M.shape[1] for M in matrices)
    out = np.zeros((rows, cols), dtype=int)
    r = c = 0
    for M in matrices:
        out[r : r + M.shape[0], c : c + M.shape[1]] = M
        r += M.shape[0]
        c += M.shape[1]
    return out


def _identity_layer(n: int) -> Layer:
    return Layer(W=np.eye(n, dtype=int), b=np.zeros(n, dtype=int))


def parallel(*circuits: list[Layer]) -> list[Layer]:
    """Run circuits on disjoint input slices and concatenate the outputs.

    Input layout for the combined circuit is the concatenation of each branch's
    inputs in order. Output layout is the concatenation of each branch's
    outputs.

    Branches with fewer layers are padded with identity layers so all branches
    end together. This is safe when the values being padded are non-negative
    (true for any boolean / ReLU-emitting sub-circuit).
    """
    if not circuits:
        raise ValueError("parallel: need at least one circuit")
    if any(not c for c in circuits):
        raise ValueError("parallel: empty branches are not allowed")

    depths = [len(c) for c in circuits]
    max_depth = max(depths)

    padded: list[list[Layer]] = []
    for c in circuits:
        if len(c) < max_depth:
            tail = [_identity_layer(c[-1].out_features)] * (max_depth - len(c))
            padded.append(list(c) + tail)
        else:
            padded.append(list(c))

    result: list[Layer] = []
    for i in range(max_depth):
        Ws = [branch[i].W for branch in padded]
        bs = [branch[i].b for branch in padded]
        result.append(Layer(W=_block_diag(*Ws), b=np.concatenate(bs).astype(int)))
    return result
