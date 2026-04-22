import numpy as np
import pytest

from theorematic import evaluate
from theorematic.fixtures import (
    block_diagonal_net,
    equality_spike,
    identity_net,
    xor_net,
)
from theorematic.ilp import invert, preact_bounds


def test_preact_bounds_xor():
    net = xor_net()
    lo = np.zeros(2)
    hi = np.ones(2)
    bounds = preact_bounds(net, lo, hi)
    assert len(bounds) == 2
    # layer 0: z = [a+b, a+b-1] over {0,1}^2 -> [0,2] and [-1,1]
    z0_lo, z0_hi = bounds[0]
    assert np.array_equal(z0_lo, [0, -1])
    assert np.array_equal(z0_hi, [2, 1])


def test_invert_xor_finds_one():
    net = xor_net()
    r = invert(net, target=[1], input_lo=0, input_hi=1)
    assert r.feasible
    assert tuple(int(v) for v in r.x) in {(0, 1), (1, 0)}
    # the found input really does evaluate to 1
    assert int(evaluate(net, r.x)[0]) == 1


def test_invert_xor_finds_zero():
    net = xor_net()
    r = invert(net, target=[0], input_lo=0, input_hi=1)
    assert r.feasible
    assert tuple(int(v) for v in r.x) in {(0, 0), (1, 1)}


def test_invert_xor_infeasible_target():
    net = xor_net()
    # XOR cannot produce 2 on boolean inputs
    r = invert(net, target=[2], input_lo=0, input_hi=1)
    assert not r.feasible


@pytest.mark.parametrize("target", [0, 3, 7, 15])
def test_invert_equality_spike(target):
    net = equality_spike(target)
    r = invert(net, target=[1], input_lo=0, input_hi=20)
    assert r.feasible
    assert int(r.x[0]) == target


def test_invert_equality_spike_miss_is_infeasible_with_constrained_input():
    # Target value is 7 but we restrict the input domain to 0..5 -> no preimage.
    net = equality_spike(7)
    r = invert(net, target=[1], input_lo=0, input_hi=5)
    assert not r.feasible


def test_invert_identity_returns_target():
    net = identity_net(3)
    r = invert(net, target=[2, 0, 1], input_lo=0, input_hi=5)
    assert r.feasible
    assert np.array_equal(r.x, [2, 0, 1])


def test_invert_respects_dont_care():
    net = identity_net(3)
    # only pin the middle coordinate
    r = invert(net, target=[None, 4, None], input_lo=0, input_hi=5)
    assert r.feasible
    assert int(r.x[1]) == 4


def test_invert_block_diagonal_is_feasible_for_zero_output():
    # block_diagonal output is a signed combination; zero input -> zero output
    net = block_diagonal_net([2, 3])
    r = invert(net, target=[0] * 5, input_lo=0, input_hi=1)
    assert r.feasible
    # verify round trip
    assert np.array_equal(evaluate(net, r.x), np.zeros(5))
