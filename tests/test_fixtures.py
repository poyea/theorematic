import numpy as np
import pytest

from theorematic import evaluate
from theorematic.fixtures import (
    block_diagonal_net,
    equality_spike,
    identity_net,
    permutation_net,
    xor_net,
)


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


@pytest.mark.parametrize("target", [0, 3, 7, 15])
def test_equality_spike(target):
    net = equality_spike(target)
    for x in range(-2, 18):
        out = int(evaluate(net, np.array([x]))[0])
        assert out == (1 if x == target else 0), f"x={x}, target={target}, got {out}"
