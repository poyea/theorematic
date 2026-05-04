"""Small hand-built nets with known structure.

Used as shared test targets across techniques. Every fixture returns a
`list[Layer]` so it drops straight into `evaluate`.

Built using the primitives in `theorematic.construct`. Fixtures whose value
is the underlying ReLU-arithmetic *identity* (xor, equality-spike, n-bit
comparators) are still spelled out densely — the trick is the lesson. Fixtures
whose value is structural (identity, permutation, block-diagonal) are built
compositionally to make the structure self-evident.
"""

from __future__ import annotations

import numpy as np

from theorematic.construct import linear, parallel, route
from theorematic.net import Layer


def identity_net(n: int) -> list[Layer]:
    return [linear(np.eye(n), np.zeros(n))]


def permutation_net(perm: list[int]) -> list[Layer]:
    """Pure permutation: output[i] = input[perm[i]]."""
    return [route(perm, len(perm))]


def block_diagonal_net(block_sizes: list[int]) -> list[Layer]:
    """A single layer whose weight matrix has visible block-diagonal structure.

    Each block is a dense +1/-1 alternating pattern so the blocks pop visually.
    Built by composing per-block circuits in parallel — the block-diagonal
    weight matrix emerges from the composition primitive itself.
    """
    branches: list[list[Layer]] = []
    for size in block_sizes:
        block = np.fromfunction(lambda i, j: (-1) ** ((i + j).astype(int)), (size, size), dtype=int)
        branches.append([linear(block, np.zeros(size))])
    return parallel(*branches)


def xor_net() -> list[Layer]:
    """2-bit XOR via the identity XOR(a,b) = ReLU(a+b) - 2*ReLU(a+b-1).

    Input: [a, b] in {0,1}^2. Hidden: [a+b, a+b-1] through ReLU. Output: scalar.
    """
    return [linear([[1, 1], [1, 1]], [0, -1]), linear([[1, -2]], [0])]


def n_bit_equality(n: int) -> list[Layer]:
    """Output 1 iff two n-bit unsigned inputs are equal.

    Input layout (LSB-first): `[a_0, ..., a_{n-1}, b_0, ..., b_{n-1}]` in {0, 1}.
    Output: scalar in {0, 1}.

    Construction (3 layers):
      - Layer 0: per-bit pre-activations `u_i = a_i + b_i` and `v_i = a_i + b_i - 1`.
      - Layer 1: collapse to one hidden unit `1 - Σ u_i + 2 Σ v_i`, which is 1
        iff every bit pair matches and 0 otherwise.
      - Layer 2: linear identity carrying the answer to the output convention.
    """
    if n < 1:
        raise ValueError("n must be >= 1")

    W0 = np.zeros((2 * n, 2 * n), dtype=int)
    b0 = np.zeros(2 * n, dtype=int)
    for i in range(n):
        W0[2 * i, i] = 1
        W0[2 * i, n + i] = 1
        W0[2 * i + 1, i] = 1
        W0[2 * i + 1, n + i] = 1
        b0[2 * i + 1] = -1

    W1 = np.zeros((1, 2 * n), dtype=int)
    for i in range(n):
        W1[0, 2 * i] = -1
        W1[0, 2 * i + 1] = 2

    return [linear(W0, b0), linear(W1, [1]), linear([[1]], [0])]


def n_bit_less_than(n: int) -> list[Layer]:
    """Output 1 iff `a < b` as n-bit unsigned integers (LSB-first).

    Input layout: `[a_0, ..., a_{n-1}, b_0, ..., b_{n-1}]`.

    Construction (3 layers):
      - Layer 0: positional `p = ReLU(value(b) - value(a))` — zero when `a >= b`,
        positive otherwise.
      - Layer 1: copy `p` and compute `ReLU(p - 1)`.
      - Layer 2: linear `p - ReLU(p - 1)` — saturates at 1 for any p >= 1.
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    weights = [1 << i for i in range(n)]

    W0 = np.zeros((1, 2 * n), dtype=int)
    for i in range(n):
        W0[0, i] = -weights[i]
        W0[0, n + i] = weights[i]

    return [
        linear(W0, [0]),
        linear([[1], [1]], [0, -1]),
        linear([[1, -1]], [0]),
    ]


def one_hot_mux(k: int) -> list[Layer]:
    """Select one of k data bits using a one-hot select vector.

    Input layout: `[d_0, ..., d_{k-1}, sel_0, ..., sel_{k-1}]`, `sel` one-hot.
    Output: scalar = `d_i` where `sel_i == 1`.

    Behaviour outside the one-hot precondition is undefined (the circuit does
    not validate). 2 layers: per-channel `AND(d_i, sel_i)` then sum.
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    W0 = np.zeros((k, 2 * k), dtype=int)
    for i in range(k):
        W0[i, i] = 1
        W0[i, k + i] = 1
    return [linear(W0, -np.ones(k)), linear(np.ones((1, k)), [0])]


def equality_spike(target: int) -> list[Layer]:
    """Scalar equality via the triangle spike.

    EQ(x, v) = ReLU(v - x - 1) - 2*ReLU(v - x) + ReLU(v - x + 1)
    returns 1 iff x == v, else 0. Input is a single scalar.
    """
    return [
        linear([[-1], [-1], [-1]], [target - 1, target, target + 1]),
        linear([[1, -2, 1]], [0]),
    ]
