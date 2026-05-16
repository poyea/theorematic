import numpy as np
import pytest

from theorematic import evaluate
from theorematic.errors import VerificationError
from theorematic.fixtures import (
    block_diagonal_net,
    equality_spike,
    identity_net,
    n_bit_equality,
    n_bit_less_than,
    one_hot_mux,
    xor_net,
)
from theorematic.sat import invert


def test_invert_xor_finds_one():
    net = xor_net()
    r = invert(net, target=[1], input_lo=0, input_hi=1)
    assert r.feasible
    assert tuple(int(v) for v in r.x) in {(0, 1), (1, 0)}
    assert int(evaluate(net, r.x)[0]) == 1


def test_invert_xor_finds_zero():
    net = xor_net()
    r = invert(net, target=[0], input_lo=0, input_hi=1)
    assert r.feasible
    assert tuple(int(v) for v in r.x) in {(0, 0), (1, 1)}


def test_invert_xor_infeasible_target():
    net = xor_net()
    r = invert(net, target=[2], input_lo=0, input_hi=1)
    assert not r.feasible
    assert r.status == "unsat"


@pytest.mark.parametrize("target", [0, 3, 7, 15])
def test_invert_equality_spike(target):
    net = equality_spike(target)
    r = invert(net, target=[1], input_lo=0, input_hi=20)
    assert r.feasible
    assert int(r.x[0]) == target


def test_invert_equality_spike_miss_is_infeasible_with_constrained_input():
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
    r = invert(net, target=[None, 4, None], input_lo=0, input_hi=5)
    assert r.feasible
    assert int(r.x[1]) == 4


@pytest.mark.parametrize("n", [2, 3])
def test_invert_n_bit_equality_finds_match(n):
    r = invert(n_bit_equality(n), target=[1], input_lo=0, input_hi=1)
    assert r.feasible
    a_bits = r.x[:n]
    b_bits = r.x[n:]
    assert np.array_equal(a_bits, b_bits)


@pytest.mark.parametrize("n", [2, 3])
def test_invert_n_bit_less_than_finds_strict_pair(n):
    r = invert(n_bit_less_than(n), target=[1], input_lo=0, input_hi=1)
    assert r.feasible
    a = sum(int(r.x[i]) << i for i in range(n))
    b = sum(int(r.x[n + i]) << i for i in range(n))
    assert a < b


def test_invert_n_bit_less_than_zero_targets_a_geq_b():
    r = invert(n_bit_less_than(3), target=[0], input_lo=0, input_hi=1)
    assert r.feasible
    a = sum(int(r.x[i]) << i for i in range(3))
    b = sum(int(r.x[3 + i]) << i for i in range(3))
    assert a >= b


def test_invert_one_hot_mux_recovers_a_one_selecting_pair():
    k = 4
    r = invert(one_hot_mux(k), target=[1], input_lo=0, input_hi=1)
    assert r.feasible
    data = r.x[:k]
    sel = r.x[k:]
    sel_indices = [i for i, s in enumerate(sel) if s == 1]
    assert len(sel_indices) == 1
    assert data[sel_indices[0]] == 1


def test_invert_block_diagonal_is_feasible_for_zero_output():
    net = block_diagonal_net([2, 3])
    r = invert(net, target=[0] * 5, input_lo=0, input_hi=1)
    assert r.feasible
    assert np.array_equal(evaluate(net, r.x), np.zeros(5))


def test_invert_round_trip_for_all_basic_fixtures():
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
            assert int(actual[j]) == t


def test_verification_fires_when_forward_pass_disagrees(monkeypatch):
    """If z3 returns sat but evaluate disagrees, raise VerificationError."""
    import theorematic.sat as sat_mod

    real_evaluate = sat_mod.evaluate

    def lying_evaluate(layers, x, **kw):
        return real_evaluate(layers, x, **kw) + 999

    monkeypatch.setattr(sat_mod, "evaluate", lying_evaluate)
    with pytest.raises(VerificationError, match="SMT encoding"):
        invert(xor_net(), target=[1], input_lo=0, input_hi=1)


def test_rejects_target_length_mismatch():
    with pytest.raises(ValueError, match="target has"):
        invert(xor_net(), target=[1, 0], input_lo=0, input_hi=1)


def test_rejects_empty_layers():
    with pytest.raises(ValueError, match="non-empty"):
        invert([], target=[1], input_lo=0, input_hi=1)


def test_rejects_inverted_bounds():
    with pytest.raises(ValueError, match="input_lo"):
        invert(xor_net(), target=[1], input_lo=5, input_hi=0)
