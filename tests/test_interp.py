import numpy as np
import pytest

from theorematic import Layer
from theorematic.fixtures import (
    equality_spike,
    n_bit_equality,
    xor_net,
)
from theorematic.interp import (
    activation_profile,
    active_neurons,
    cluster_neurons_by_activation,
)


def _bits(value: int, n: int) -> list[int]:
    return [(value >> i) & 1 for i in range(n)]


def _all_two_input_bits() -> np.ndarray:
    return np.array([[a, b] for a in (0, 1) for b in (0, 1)])


# -- active_neurons: xor truth table ------------------------------------------


def test_active_neurons_xor_01():
    # xor: layer 0 has neurons (a+b, a+b-1). For (0,1): a+b=1 fires, a+b-1=0 dead.
    net = xor_net()
    active = active_neurons(net, np.array([0, 1]))
    # exactly one of the two hidden neurons fires
    hidden = {(i, k) for (i, k) in active if i == 0}
    assert hidden == {(0, 0)}


def test_active_neurons_xor_10():
    net = xor_net()
    active = active_neurons(net, np.array([1, 0]))
    hidden = {(i, k) for (i, k) in active if i == 0}
    assert hidden == {(0, 0)}


def test_active_neurons_xor_11_both_fire():
    # (1,1): a+b=2 fires, a+b-1=1 fires
    net = xor_net()
    active = active_neurons(net, np.array([1, 1]))
    hidden = {(i, k) for (i, k) in active if i == 0}
    assert hidden == {(0, 0), (0, 1)}


def test_active_neurons_xor_00_none_fire():
    net = xor_net()
    active = active_neurons(net, np.array([0, 0]))
    hidden = {(i, k) for (i, k) in active if i == 0}
    assert hidden == set()


# -- activation_profile -------------------------------------------------------


def test_activation_profile_shapes_match_layer_widths():
    net = xor_net()
    profile = activation_profile(net, _all_two_input_bits())
    assert len(profile) == len(net)
    for row, layer in zip(profile, net):
        assert row.shape == (layer.out_features,)


def test_activation_profile_equality_spike_only_fires_near_target():
    # equality_spike has 3 hidden neurons. Far from target they should never
    # collectively produce the spike. We sweep wide and check that the spike
    # output (final layer) fires exactly once: at target.
    target = 4
    net = equality_spike(target)
    xs = np.arange(-3, 12).reshape(-1, 1)
    profile = activation_profile(net, xs)
    # final layer should fire exactly once across the sweep
    assert profile[-1][0] == pytest.approx(1.0 / xs.shape[0])
    # the three hidden neurons each fire on some subset, never all-or-nothing.
    hidden = profile[0]
    assert hidden.shape == (3,)
    assert (hidden > 0).all()
    assert (hidden < 1).all()


def test_activation_profile_xor_never_and_always():
    # On the full 2-input sweep:
    # neuron 0 (a+b) fires for {(0,1),(1,0),(1,1)} -> 3/4
    # neuron 1 (a+b-1) fires only for (1,1) -> 1/4
    net = xor_net()
    profile = activation_profile(net, _all_two_input_bits())
    assert profile[0][0] == pytest.approx(3 / 4)
    assert profile[0][1] == pytest.approx(1 / 4)


# -- cluster_neurons_by_activation --------------------------------------------


def test_cluster_groups_duplicate_neurons():
    # Two neurons in the same hidden layer with *identical* incoming weights
    # have identical activation patterns and must cluster together.
    h = Layer(
        W=np.array([[1, 1], [1, 1], [1, -1]], dtype=int),  # neurons 0 and 1 identical
        b=np.array([0, 0, 0], dtype=int),
    )
    out = Layer(W=np.array([[1, 1, 1]], dtype=int), b=np.array([0], dtype=int))
    net = [h, out]
    clusters = cluster_neurons_by_activation(net, _all_two_input_bits())
    # neurons (0,0) and (0,1) share a pattern; (0,2) has its own.
    sizes = sorted(len(v) for v in clusters.values())
    assert sizes == [1, 2]


def test_cluster_n_bit_equality_sweep_runs():
    # Smoke test: clustering n_bit_equality(2) over the full 2^(2n) sweep
    # yields a non-empty partition that covers every hidden neuron exactly
    # once.
    n = 2
    net = n_bit_equality(n)
    inputs = np.array([_bits(a, n) + _bits(b, n) for a in range(1 << n) for b in range(1 << n)])
    clusters = cluster_neurons_by_activation(net, inputs)
    total_hidden = sum(layer.out_features for layer in net[:-1])
    members = [m for v in clusters.values() for m in v]
    assert len(members) == total_hidden
    assert len(set(members)) == total_hidden


def test_cluster_skips_final_layer():
    # Single-layer net: no hidden layers, clustering is empty.
    one = [Layer(W=np.eye(2, dtype=int), b=np.zeros(2, dtype=int))]
    clusters = cluster_neurons_by_activation(one, np.array([[0, 0], [1, 1]]))
    assert clusters == {}


def test_cluster_groups_by_pattern():
    net = xor_net()
    clusters = cluster_neurons_by_activation(net, _all_two_input_bits())
    # The two hidden neurons have *different* patterns (see profile above),
    # so they end up in two distinct clusters.
    layer0_members = [m for members in clusters.values() for m in members if m[0] == 0]
    assert sorted(layer0_members) == [(0, 0), (0, 1)]
    assert len(clusters) == 2


# -- validation ---------------------------------------------------------------


def test_activation_profile_rejects_empty_layers():
    with pytest.raises(ValueError, match="non-empty"):
        activation_profile([], np.array([[0, 0]]))


def test_active_neurons_rejects_empty_layers():
    with pytest.raises(ValueError, match="non-empty"):
        active_neurons([], np.array([0, 0]))


def test_cluster_rejects_empty_layers():
    with pytest.raises(ValueError, match="non-empty"):
        cluster_neurons_by_activation([], np.array([[0, 0]]))


def test_active_neurons_rejects_wrong_width():
    net = xor_net()
    with pytest.raises(ValueError, match="input width"):
        active_neurons(net, np.array([0, 0, 0]))


def test_activation_profile_rejects_wrong_width():
    net = xor_net()
    with pytest.raises(ValueError, match="input width"):
        activation_profile(net, np.array([[0, 0, 0], [1, 1, 1]]))


def test_activation_profile_rejects_empty_sweep():
    net = xor_net()
    with pytest.raises(ValueError, match="at least one row"):
        activation_profile(net, np.zeros((0, 2), dtype=int))


def test_active_neurons_rejects_2d_input():
    net = xor_net()
    with pytest.raises(ValueError, match="1-D"):
        active_neurons(net, np.array([[0, 1]]))
