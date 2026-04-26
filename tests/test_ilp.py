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
    xor_net,
)
from theorematic.ilp import VerificationError, invert, preact_bounds


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


def test_invert_round_trip_for_all_basic_fixtures():
    """Every feasible solve must produce an x that evaluates to the target.

    This is the verification safety net at the test level: even if `invert`
    silently swallowed its internal check one day, this catches it.
    """
    cases = [
        (xor_net(), [1], 0, 1),
        (xor_net(), [0], 0, 1),
        (equality_spike(7), [1], 0, 20),
        (identity_net(3), [4, 0, 2], 0, 5),
    ]
    for net, target, lo, hi in cases:
        r = invert(net, target=target, input_lo=lo, input_hi=hi)
        assert r.feasible
        actual = evaluate(net, r.x)
        for j, t in enumerate(target):
            assert int(actual[j]) == t, f"target={target}, got {actual.tolist()}"


def test_verification_fires_when_target_is_secretly_unreachable(monkeypatch):
    """If the encoding ever produces an inconsistent solution, raise loudly.

    We simulate this by monkeypatching `evaluate` so the recomputed forward
    pass disagrees with the solver. The exception path is what's being tested,
    not the mock.
    """
    import theorematic.ilp as ilp_mod

    real_evaluate = ilp_mod.evaluate

    def lying_evaluate(layers, x, **kw):
        out = real_evaluate(layers, x, **kw)
        return out + 999  # guaranteed not to match any target

    monkeypatch.setattr(ilp_mod, "evaluate", lying_evaluate)
    with pytest.raises(VerificationError, match="big-M"):
        invert(xor_net(), target=[1], input_lo=0, input_hi=1)


@pytest.mark.parametrize("n", [2, 3])
def test_invert_n_bit_equality_finds_match(n):
    """Asking for output 1 should yield two halves that are bitwise equal."""
    r = invert(n_bit_equality(n), target=[1], input_lo=0, input_hi=1)
    assert r.feasible
    a_bits = r.x[:n]
    b_bits = r.x[n:]
    assert np.array_equal(a_bits, b_bits), f"got a={a_bits}, b={b_bits}"


@pytest.mark.parametrize("n", [2, 3])
def test_invert_n_bit_less_than_finds_strict_pair(n):
    r = invert(n_bit_less_than(n), target=[1], input_lo=0, input_hi=1)
    assert r.feasible
    a = sum(int(r.x[i]) << i for i in range(n))
    b = sum(int(r.x[n + i]) << i for i in range(n))
    assert a < b, f"got a={a}, b={b}"


def test_invert_n_bit_less_than_zero_targets_a_geq_b():
    r = invert(n_bit_less_than(3), target=[0], input_lo=0, input_hi=1)
    assert r.feasible
    a = sum(int(r.x[i]) << i for i in range(3))
    b = sum(int(r.x[3 + i]) << i for i in range(3))
    assert a >= b


def test_invert_one_hot_mux_recovers_a_one_selecting_pair():
    """Forcing output=1 must produce a config with d[sel_idx] == 1."""
    k = 4
    r = invert(one_hot_mux(k), target=[1], input_lo=0, input_hi=1)
    assert r.feasible
    data = r.x[:k]
    sel = r.x[k:]
    sel_indices = [i for i, s in enumerate(sel) if s == 1]
    assert len(sel_indices) == 1, f"expected one-hot sel, got {sel.tolist()}"
    assert data[sel_indices[0]] == 1


def test_invert_block_diagonal_is_feasible_for_zero_output():
    # block_diagonal output is a signed combination; zero input -> zero output
    net = block_diagonal_net([2, 3])
    r = invert(net, target=[0] * 5, input_lo=0, input_hi=1)
    assert r.feasible
    # verify round trip
    assert np.array_equal(evaluate(net, r.x), np.zeros(5))
