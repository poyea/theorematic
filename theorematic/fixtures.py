"""Small hand-built nets with known structure.

Used as shared test targets across techniques. Every fixture returns a
`list[Layer]` so it drops straight into `evaluate`.
"""

from __future__ import annotations

import numpy as np

from theorematic.net import Layer


def identity_net(n: int) -> list[Layer]:
    return [Layer(W=np.eye(n, dtype=int), b=np.zeros(n, dtype=int))]


def permutation_net(perm: list[int]) -> list[Layer]:
    """Pure permutation: output[i] = input[perm[i]]."""
    n = len(perm)
    W = np.zeros((n, n), dtype=int)
    for i, j in enumerate(perm):
        W[i, j] = 1
    return [Layer(W=W, b=np.zeros(n, dtype=int))]


def block_diagonal_net(block_sizes: list[int]) -> list[Layer]:
    """A single layer whose weight matrix has visible block-diagonal structure.

    Each block is a dense +1/-1 alternating pattern so the blocks pop visually.
    """
    n = sum(block_sizes)
    W = np.zeros((n, n), dtype=int)
    off = 0
    for size in block_sizes:
        block = np.fromfunction(lambda i, j: (-1) ** ((i + j).astype(int)), (size, size), dtype=int)
        W[off : off + size, off : off + size] = block
        off += size
    return [Layer(W=W, b=np.zeros(n, dtype=int))]


def xor_net() -> list[Layer]:
    """2-bit XOR via the identity XOR(a,b) = ReLU(a+b) - 2*ReLU(a+b-1).

    Input: [a, b] in {0,1}^2. Hidden: [a+b, a+b-1] through ReLU. Output: scalar.
    """
    l1 = Layer(W=np.array([[1, 1], [1, 1]], dtype=int), b=np.array([0, -1], dtype=int))
    l2 = Layer(W=np.array([[1, -2]], dtype=int), b=np.array([0], dtype=int))
    return [l1, l2]


def equality_spike(target: int, width: int = 8) -> list[Layer]:
    """Scalar equality via the triangle spike.

    EQ(x, v) = ReLU(v - x - 1) - 2*ReLU(v - x) + ReLU(v - x + 1)
    returns 1 iff x == v, else 0. Input is a single scalar. `width` sets the
    range of x values the circuit is sized for but the formula itself is exact.
    """
    del width  # kept for symmetry with later multi-byte versions
    # hidden has three units, each computing ReLU(-x + (v + k)) for k in {-1, 0, +1}
    W1 = np.array([[-1], [-1], [-1]], dtype=int)
    b1 = np.array([target - 1, target, target + 1], dtype=int)
    W2 = np.array([[1, -2, 1]], dtype=int)
    b2 = np.array([0], dtype=int)
    return [Layer(W=W1, b=b1), Layer(W=W2, b=b2)]
