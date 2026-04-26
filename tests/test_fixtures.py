import numpy as np
import pytest

from theorematic import evaluate
from theorematic.fixtures import (
    block_diagonal_net,
    equality_spike,
    identity_net,
    n_bit_equality,
    n_bit_less_than,
    one_hot_mux,
    permutation_net,
    xor_net,
)


def _bits(value: int, n: int) -> list[int]:
    return [(value >> i) & 1 for i in range(n)]


def test_identity():
    net = identity_net(4)
    x = np.array([3, -1, 7, 0])
    assert np.array_equal(evaluate(net, x), x)


def test_permutation():
    net = permutation_net([2, 0, 1])
    x = np.array([10, 20, 30])
    assert np.array_equal(evaluate(net, x), np.array([30, 10, 20]))


def test_block_diagonal_shape():
    net = block_diagonal_net([3, 2, 4])
    W = net[0].W
    assert W.shape == (9, 9)
    # corner block from size-3 is dense; the block that crosses boundaries is zero
    assert W[0, 3] == 0 and W[2, 4] == 0 and W[3, 2] == 0


@pytest.mark.parametrize("a,b", [(0, 0), (0, 1), (1, 0), (1, 1)])
def test_xor_truth_table(a, b):
    net = xor_net()
    out = evaluate(net, np.array([a, b]))
    assert int(out[0]) == (a ^ b)


@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_n_bit_equality_exhaustive(n):
    net = n_bit_equality(n)
    for a in range(1 << n):
        for b in range(1 << n):
            x = np.array(_bits(a, n) + _bits(b, n))
            out = int(evaluate(net, x)[0])
            expected = 1 if a == b else 0
            assert out == expected, f"n={n}, a={a}, b={b}, got {out}"


@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_n_bit_less_than_exhaustive(n):
    net = n_bit_less_than(n)
    for a in range(1 << n):
        for b in range(1 << n):
            x = np.array(_bits(a, n) + _bits(b, n))
            out = int(evaluate(net, x)[0])
            expected = 1 if a < b else 0
            assert out == expected, f"n={n}, a={a}, b={b}, got {out}"


@pytest.mark.parametrize("k", [1, 2, 3, 4])
def test_one_hot_mux_exhaustive(k):
    net = one_hot_mux(k)
    for data in range(1 << k):
        for sel_idx in range(k):
            d = _bits(data, k)
            sel = [1 if i == sel_idx else 0 for i in range(k)]
            x = np.array(d + sel)
            out = int(evaluate(net, x)[0])
            assert out == d[sel_idx], f"k={k}, data={d}, sel_idx={sel_idx}, got {out}"


@pytest.mark.parametrize("target", [0, 3, 7, 15])
def test_equality_spike(target):
    net = equality_spike(target)
    for x in range(-2, 18):
        out = int(evaluate(net, np.array([x]))[0])
        assert out == (1 if x == target else 0), f"x={x}, target={target}, got {out}"
