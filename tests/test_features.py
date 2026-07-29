import itertools

import numpy as np
import pytest

from theorematic import evaluate, linear
from theorematic.features import (
    ALWAYS,
    NEVER,
    SOMETIMES,
    UNKNOWN,
    describe_all,
    describe_neuron,
    input_support,
    preact_net,
)
from theorematic.fixtures import (
    equality_spike,
    n_bit_equality,
    n_bit_less_than,
    one_hot_mux,
    xor_net,
)
from theorematic.interp import activation_profile
from theorematic.reduce import remove_unreachable_neurons


def _box(n: int, lo: int, hi: int) -> np.ndarray:
    return np.array(list(itertools.product(range(lo, hi + 1), repeat=n)))


# --- preact_net ---------------------------------------------------------------


def test_preact_net_exposes_pre_activation_not_post_relu():
    # Layer 0 goes negative at x = [0, 1]; the prefix must report -1, not 0.
    net = [linear([[1, -1]], [0]), linear([[1]], [0])]
    prefix = preact_net(net, 0)
    assert len(prefix) == 1
    assert evaluate(prefix, np.array([0, 1]))[0] == -1


@pytest.mark.parametrize("layer_index", [0, 1, 2])
def test_preact_net_matches_manual_forward_pass(layer_index):
    net = n_bit_less_than(2)
    x = np.array([1, 0, 0, 1])
    h = x
    for i, layer in enumerate(net[: layer_index + 1]):
        pre = layer.W @ h + layer.b
        h = pre if i == layer_index else np.maximum(pre, 0)
    assert np.array_equal(evaluate(preact_net(net, layer_index), x), h)


def test_preact_net_rejects_bad_input():
    with pytest.raises(ValueError, match="non-empty"):
        preact_net([], 0)
    with pytest.raises(ValueError, match="out of range"):
        preact_net(xor_net(), 5)


# --- input_support -----------------------------------------------------------


def test_input_support_recovers_per_bit_structure_of_equality():
    # n_bit_equality compares bit i of a against bit i of b, so each layer-0
    # neuron must depend on exactly one such pair (LSB-first layout).
    net = n_bit_equality(2)
    assert input_support(net, 0, 0) == (0, 2)
    assert input_support(net, 0, 1) == (0, 2)
    assert input_support(net, 0, 2) == (1, 3)
    assert input_support(net, 0, 3) == (1, 3)


def test_input_support_recovers_mux_pairing():
    k = 3
    net = one_hot_mux(k)
    for i in range(k):
        assert input_support(net, 0, i) == (i, k + i)


def test_input_support_widens_with_depth():
    net = n_bit_equality(2)
    # Layer 1 aggregates every bit pair, so its support is the whole input.
    assert input_support(net, 1, 0) == (0, 1, 2, 3)


def test_coordinates_outside_support_cannot_change_preactivation():
    net = n_bit_equality(2)
    layer_index, neuron_index = 0, 0
    support = set(input_support(net, layer_index, neuron_index))
    outside = [c for c in range(net[0].in_features) if c not in support]
    prefix = preact_net(net, layer_index)
    base = np.zeros(net[0].in_features, dtype=int)
    reference = evaluate(prefix, base)[neuron_index]
    for coord in outside:
        probe = base.copy()
        probe[coord] = 1
        assert evaluate(prefix, probe)[neuron_index] == reference


def test_input_support_rejects_bad_position():
    with pytest.raises(ValueError, match="non-empty"):
        input_support([], 0, 0)
    with pytest.raises(ValueError, match="layer_index"):
        input_support(xor_net(), 9, 0)
    with pytest.raises(ValueError, match="neuron_index"):
        input_support(xor_net(), 0, 9)


# --- describe_neuron ---------------------------------------------------------


def test_describe_neuron_firing_set_matches_brute_force():
    net = xor_net()
    for k in range(net[0].out_features):
        feature = describe_neuron(net, 0, k, input_lo=0, input_hi=1)
        expected = {
            (a, b)
            for a in (0, 1)
            for b in (0, 1)
            if evaluate(preact_net(net, 0), np.array([a, b]))[k] > 0
        }
        assert set(feature.firing_inputs) == expected


def test_describe_neuron_reports_mux_and_gate():
    # A mux data/select pair fires only when both bits are 1 — an AND gate,
    # recovered from the weights alone.
    net = one_hot_mux(3)
    feature = describe_neuron(net, 0, 0, input_lo=0, input_hi=1)
    assert feature.verdict == SOMETIMES
    assert feature.support == (0, 3)
    assert feature.firing_inputs == ((1, 1),)


def test_describe_neuron_never_fires_when_bounds_exclude_it():
    # Neuron 1 needs a + b >= 3, unreachable on binary inputs.
    net = [linear([[1, 1], [1, 1]], [0, -3]), linear([[1, 1]], [0])]
    feature = describe_neuron(net, 0, 1, input_lo=0, input_hi=1)
    assert feature.verdict == NEVER
    assert feature.is_constant
    assert feature.essential == ()


def test_describe_neuron_always_fires_when_bounds_force_it():
    net = [linear([[1, 1], [1, 1]], [0, 5]), linear([[1, 1]], [0])]
    feature = describe_neuron(net, 0, 1, input_lo=0, input_hi=1)
    assert feature.verdict == ALWAYS
    assert feature.is_constant


def test_describe_neuron_separates_essential_from_support():
    # z = 10*x0 + x1 - 5 fires iff x0 == 1, whatever x1 does. x1 has a
    # non-zero weight path, so it is in the support but cannot flip the
    # outcome — exactly the distinction `essential` is for.
    net = [linear([[10, 1], [1, 1]], [-5, 0]), linear([[1, 1]], [0])]
    feature = describe_neuron(net, 0, 0, input_lo=0, input_hi=1)
    assert feature.verdict == SOMETIMES
    assert feature.support == (0, 1)
    assert feature.essential == (0,)


def test_describe_neuron_degrades_to_unknown_instead_of_guessing():
    net = n_bit_less_than(3)
    feature = describe_neuron(net, 0, 0, input_lo=0, input_hi=1, max_evaluations=4)
    assert feature.verdict == UNKNOWN
    assert feature.firing_inputs is None
    assert feature.essential is None
    # The support is structural, so it survives even when enumeration is off.
    assert len(feature.support) == 6


def test_describe_neuron_rejects_bad_input():
    net = xor_net()
    with pytest.raises(ValueError, match="layer_index"):
        describe_neuron(net, 7, 0)
    with pytest.raises(ValueError, match="neuron_index"):
        describe_neuron(net, 0, 7)
    with pytest.raises(ValueError, match="input_lo"):
        describe_neuron(net, 0, 0, input_lo=1, input_hi=0)


# --- describe_all ------------------------------------------------------------


def test_describe_all_covers_hidden_neurons_in_order():
    net = n_bit_equality(2)
    features = describe_all(net)
    expected = [(0, k) for k in range(net[0].out_features)] + [
        (1, k) for k in range(net[1].out_features)
    ]
    assert [(f.layer, f.neuron) for f in features] == expected


def test_describe_all_excludes_final_layer_by_default():
    net = n_bit_equality(2)
    assert all(f.layer < len(net) - 1 for f in describe_all(net))
    assert any(f.layer == len(net) - 1 for f in describe_all(net, hidden_only=False))


def test_describe_all_rejects_empty():
    with pytest.raises(ValueError, match="non-empty"):
        describe_all([])


def test_describe_all_on_single_layer_net_has_no_hidden_neurons():
    # Matches interp.cluster_neurons_by_activation returning {} for one layer:
    # the only layer is the linear output, so there is nothing hidden to probe.
    net = [linear([[1, -1]], [0])]
    assert describe_all(net) == []
    assert len(describe_all(net, hidden_only=False)) == 1


def test_absent_firing_set_distinguishes_unenumerated_from_empty():
    # The two absence conventions are not interchangeable: None means no
    # enumeration ran, () means it ran and found nothing.
    unreachable = [linear([[1, 1], [1, 1]], [0, -3]), linear([[1, 1]], [0])]
    settled_by_bounds = describe_neuron(unreachable, 0, 1, input_lo=0, input_hi=1)
    assert settled_by_bounds.verdict == NEVER
    assert settled_by_bounds.firing_inputs is None
    assert settled_by_bounds.essential == ()

    too_wide = describe_neuron(n_bit_less_than(3), 0, 0, max_evaluations=4)
    assert too_wide.verdict == UNKNOWN
    assert too_wide.firing_inputs is None
    assert too_wide.essential is None


# --- agreement with the neighbouring modules --------------------------------


def test_verdicts_agree_with_interp_activation_rates():
    net = n_bit_equality(2)
    sweep = _box(net[0].in_features, 0, 1)
    profile = activation_profile(net, sweep)
    for feature in describe_all(net):
        rate = profile[feature.layer][feature.neuron]
        if feature.verdict == NEVER:
            assert rate == 0.0
        elif feature.verdict == ALWAYS:
            assert rate == 1.0
        elif feature.verdict == SOMETIMES:
            assert 0.0 < rate < 1.0


def test_never_verdict_agrees_with_bound_driven_reduction():
    # A neuron this module calls "never" is one reduce should be able to drop.
    net = [linear([[1, 1], [1, 1], [1, -1]], [0, -3, 0]), linear([[1, 1, 1]], [0])]
    features = describe_all(net, input_lo=0, input_hi=1)
    dead = [f.neuron for f in features if f.verdict == NEVER]
    assert dead == [1]
    reduced = remove_unreachable_neurons(net, 0, 1)
    assert reduced[0].out_features == net[0].out_features - len(dead)


def test_essential_is_always_a_subset_of_support():
    for net in (xor_net(), n_bit_equality(2), one_hot_mux(3), equality_spike(7)):
        for feature in describe_all(net, input_lo=0, input_hi=1):
            if feature.essential is not None:
                assert set(feature.essential) <= set(feature.support)
