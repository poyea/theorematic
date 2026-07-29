import numpy as np
import pytest

from theorematic import Layer, evaluate, linear, parallel, stack
from theorematic.fixtures import (
    equality_spike,
    identity_net,
    n_bit_equality,
    n_bit_less_than,
    one_hot_mux,
    xor_net,
)
from theorematic.reduce import (
    drop_identity_layers,
    fuse_linear_layers,
    reduce,
    reduce_with_bounds,
    remove_always_zero_neurons,
    remove_dead_neurons,
    remove_unreachable_neurons,
)


def _bits(value: int, n: int) -> list[int]:
    return [(value >> i) & 1 for i in range(n)]


def test_reduce_rejects_empty():
    with pytest.raises(ValueError, match="non-empty"):
        reduce([])


def test_reduce_preserves_xor():
    net = xor_net()
    small = reduce(net)
    for a in (0, 1):
        for b in (0, 1):
            x = np.array([a, b])
            assert np.array_equal(evaluate(small, x), evaluate(net, x))


@pytest.mark.parametrize("n", [2, 3])
def test_reduce_preserves_n_bit_equality(n):
    net = n_bit_equality(n)
    small = reduce(net)
    for a in range(1 << n):
        for b in range(1 << n):
            x = np.array(_bits(a, n) + _bits(b, n))
            assert np.array_equal(evaluate(small, x), evaluate(net, x))


@pytest.mark.parametrize("n", [2, 3])
def test_reduce_preserves_n_bit_less_than(n):
    net = n_bit_less_than(n)
    small = reduce(net)
    for a in range(1 << n):
        for b in range(1 << n):
            x = np.array(_bits(a, n) + _bits(b, n))
            assert np.array_equal(evaluate(small, x), evaluate(net, x))


def test_reduce_strips_parallel_identity_padding():
    # parallel pads the shorter branch with identity layers — reduce should
    # remove them once the branches are merged with a downstream consumer.
    shallow = identity_net(2)  # depth 1
    deep = stack(identity_net(2), identity_net(2))  # depth 2
    combined = parallel(shallow, deep)  # parallel pads shallow to depth 2
    # Add a final linear "consumer" so the padding is genuinely interior.
    consumer = [linear(np.eye(4, dtype=int), [0, 0, 0, 0])]
    net = stack(combined, consumer)
    small = reduce(net)
    assert len(small) < len(net)
    for v in range(16):
        x = np.array([v & 1, (v >> 1) & 1, (v >> 2) & 1, (v >> 3) & 1])
        assert np.array_equal(evaluate(small, x), evaluate(net, x))


def test_remove_dead_neurons_drops_unused_hidden_unit():
    # Hidden layer has 3 neurons; the next layer ignores the middle one.
    h = linear([[1, 0], [0, 0], [0, 1]], [0, 0, 0])  # 3 hidden from 2 inputs
    out = linear([[1, 0, 1]], [0])  # column 1 is all zero
    net = [h, out]
    small = remove_dead_neurons(net)
    assert small[0].W.shape == (2, 2)
    assert small[1].W.shape == (1, 2)
    for x_val in range(4):
        x = np.array([x_val & 1, (x_val >> 1) & 1])
        assert np.array_equal(evaluate(small, x), evaluate(net, x))


def test_remove_always_zero_neurons_drops_zero_row():
    # Hidden layer: one neuron has W=0 and b=-1, so it always outputs 0.
    h = linear([[1, 1], [0, 0], [1, -1]], [0, -1, 0])
    out = linear([[1, 1, 1]], [0])
    net = [h, out]
    small = remove_always_zero_neurons(net)
    assert small[0].W.shape == (2, 2)
    assert small[1].W.shape == (1, 2)
    for a in (0, 1):
        for b in (0, 1):
            x = np.array([a, b])
            assert np.array_equal(evaluate(small, x), evaluate(net, x))


def test_drop_identity_layers_preserves_endpoints():
    n = 3
    interior_id = Layer(W=np.eye(n, dtype=int), b=np.zeros(n, dtype=int))
    net = [interior_id, interior_id, interior_id]  # all 3 are identity
    small = drop_identity_layers(net)
    # First and last must remain; only the middle one drops.
    assert len(small) == 2
    x = np.array([1, -2, 3])
    # The first layer's ReLU still clamps the negative, so behavior matches.
    assert np.array_equal(evaluate(small, x), evaluate(net, x))


def test_reduce_is_idempotent():
    net = n_bit_equality(3)
    once = reduce(net)
    twice = reduce(once)
    assert len(once) == len(twice)
    for a in range(8):
        for b in range(8):
            x = np.array(_bits(a, 3) + _bits(b, 3))
            assert np.array_equal(evaluate(once, x), evaluate(twice, x))


def test_reduce_preserves_one_hot_mux():
    k = 3
    net = one_hot_mux(k)
    small = reduce(net)
    for data in range(1 << k):
        for sel_idx in range(k):
            d = _bits(data, k)
            sel = [1 if i == sel_idx else 0 for i in range(k)]
            x = np.array(d + sel)
            assert np.array_equal(evaluate(small, x), evaluate(net, x))


def test_reduce_preserves_equality_spike():
    net = equality_spike(7)
    small = reduce(net)
    for x_val in range(-2, 20):
        x = np.array([x_val])
        assert np.array_equal(evaluate(small, x), evaluate(net, x))


def test_remove_unreachable_neurons_drops_neuron_that_cannot_fire():
    # Neuron 1 needs a+b >= 3 to fire, unreachable for binary inputs.
    h = linear([[1, 1], [1, 1], [1, -1]], [0, -3, 0])
    out = linear([[1, 1, 1]], [0])
    net = [h, out]
    small = remove_unreachable_neurons(net, 0, 1)
    assert small[0].W.shape == (2, 2)
    assert small[1].W.shape == (1, 2)
    for a in (0, 1):
        for b in (0, 1):
            x = np.array([a, b])
            assert np.array_equal(evaluate(small, x), evaluate(net, x))


def test_remove_unreachable_neurons_keeps_neuron_reachable_at_wider_bounds():
    h = linear([[1, 1], [1, 1], [1, -1]], [0, -3, 0])
    out = linear([[1, 1, 1]], [0])
    net = [h, out]
    assert remove_unreachable_neurons(net, 0, 5)[0].W.shape == (3, 2)


def test_remove_unreachable_neurons_subsumes_always_zero():
    h = linear([[1, 1], [0, 0], [1, -1]], [0, -1, 0])
    out = linear([[1, 1, 1]], [0])
    net = [h, out]
    structural = remove_always_zero_neurons(net)
    bound_driven = remove_unreachable_neurons(net, 0, 1)
    assert structural[0].W.shape == bound_driven[0].W.shape


def test_fuse_linear_layers_merges_when_relu_is_identity():
    # Both hidden preacts stay >= 0 on [0, 1], so the ReLU never clips.
    h = linear([[1, 1], [2, 0]], [0, 1])
    out = linear([[1, -1], [0, 2]], [3, 0])
    net = [h, out]
    fused = fuse_linear_layers(net, 0, 1)
    assert len(fused) == 1
    for a in (0, 1):
        for b in (0, 1):
            x = np.array([a, b])
            assert np.array_equal(evaluate(fused, x), evaluate(net, x))


def test_fuse_linear_layers_refuses_when_relu_can_clip():
    h = linear([[1, -1], [1, 1]], [0, 0])
    out = linear([[1, 1]], [0])
    net = [h, out]
    unchanged = fuse_linear_layers(net, 0, 1)
    assert len(unchanged) == 2
    assert np.array_equal(unchanged[0].W, h.W)


def test_fuse_linear_layers_preserves_integer_dtype():
    h = linear([[2, 3], [1, 1]], [1, 2])
    out = linear([[5, -7]], [0])
    fused = fuse_linear_layers([h, out], 0, 1)
    assert np.issubdtype(fused[0].W.dtype, np.integer)
    assert np.issubdtype(fused[0].b.dtype, np.integer)


@pytest.mark.parametrize("n", [2, 3])
def test_reduce_with_bounds_preserves_n_bit_less_than(n):
    net = n_bit_less_than(n)
    small = reduce_with_bounds(net, 0, 1)
    assert len(small) <= len(net)
    for a in range(1 << n):
        for b in range(1 << n):
            x = np.array(_bits(a, n) + _bits(b, n))
            assert np.array_equal(evaluate(small, x), evaluate(net, x))


@pytest.mark.parametrize("n", [2, 3])
def test_reduce_with_bounds_preserves_n_bit_equality(n):
    net = n_bit_equality(n)
    small = reduce_with_bounds(net, 0, 1)
    for a in range(1 << n):
        for b in range(1 << n):
            x = np.array(_bits(a, n) + _bits(b, n))
            assert np.array_equal(evaluate(small, x), evaluate(net, x))


def test_reduce_with_bounds_preserves_one_hot_mux():
    k = 3
    net = one_hot_mux(k)
    small = reduce_with_bounds(net, 0, 1)
    for data in range(1 << k):
        for sel_idx in range(k):
            sel = [1 if i == sel_idx else 0 for i in range(k)]
            x = np.array(_bits(data, k) + sel)
            assert np.array_equal(evaluate(small, x), evaluate(net, x))


def test_reduce_with_bounds_is_at_least_as_strong_as_reduce():
    net = n_bit_less_than(3)
    assert len(reduce_with_bounds(net, 0, 1)) <= len(reduce(net))


def test_reduce_with_bounds_is_idempotent():
    net = n_bit_less_than(3)
    once = reduce_with_bounds(net, 0, 1)
    twice = reduce_with_bounds(once, 0, 1)
    assert len(once) == len(twice)
    for a in range(8):
        for b in range(8):
            x = np.array(_bits(a, 3) + _bits(b, 3))
            assert np.array_equal(evaluate(once, x), evaluate(twice, x))


def test_reduce_with_bounds_accepts_per_coordinate_bounds():
    net = n_bit_less_than(2)
    lo = np.zeros(4, dtype=int)
    hi = np.ones(4, dtype=int)
    small = reduce_with_bounds(net, lo, hi)
    for a in range(4):
        for b in range(4):
            x = np.array(_bits(a, 2) + _bits(b, 2))
            assert np.array_equal(evaluate(small, x), evaluate(net, x))


def test_reduce_with_bounds_collapses_unreachable_spike():
    # The spike fires only at x == 7. Restrict the domain below that and the
    # whole detector provably collapses; include 7 and it must survive intact.
    net = equality_spike(7)
    collapsed = reduce_with_bounds(net, 0, 6)
    assert len(collapsed) == 1
    for x_val in range(7):
        x = np.array([x_val])
        assert np.array_equal(evaluate(collapsed, x), evaluate(net, x))

    intact = reduce_with_bounds(net, 0, 7)
    assert len(intact) == len(net)


def test_reduce_with_bounds_may_differ_outside_declared_bounds():
    # The contract is explicit that equivalence is domain-limited. Reducing
    # under [0, 6] discards the spike, so at x == 7 the nets disagree.
    net = equality_spike(7)
    collapsed = reduce_with_bounds(net, 0, 6)
    x = np.array([7])
    assert not np.array_equal(evaluate(collapsed, x), evaluate(net, x))


def test_bound_driven_passes_survive_a_fully_unreachable_hidden_layer():
    # Every hidden neuron is unreachable, so the layer drops to zero width and
    # its successor to zero inputs. That degenerate pair is still well-defined
    # (a 0-column matmul yields zeros, leaving the bias), and fusion then
    # folds it back into a single layer. Constant output preserved throughout.
    hidden = linear([[1, 1], [1, 1]], [-5, -9])
    out = linear([[1, 1]], [3])
    net = [hidden, out]

    pruned = remove_unreachable_neurons(net, 0, 1)
    assert pruned[0].W.shape == (0, 2)
    assert pruned[1].W.shape == (1, 0)

    collapsed = reduce_with_bounds(net, 0, 1)
    assert len(collapsed) == 1
    for a in (0, 1):
        for b in (0, 1):
            x = np.array([a, b])
            assert np.array_equal(evaluate(collapsed, x), evaluate(net, x))


def test_fusion_declines_when_composed_weights_would_overflow():
    # Fusion multiplies weights, so it grows them far faster than a forward
    # pass does. These two compose to ~1e20, past int64 — numpy would wrap it
    # silently and the rewrite would stop preserving evaluate.
    big = 10**10
    net = [linear([[big]], [0]), linear([[big]], [0])]
    assert len(fuse_linear_layers(net, 0, 1)) == 2


def test_fusion_proceeds_when_composed_weights_fit():
    # Same shape, magnitudes that stay in range: fusion must still happen, so
    # the guard above is a magnitude check and not a blanket refusal.
    net = [linear([[10**4]], [0]), linear([[10**4]], [0])]
    fused = fuse_linear_layers(net, 0, 1)
    assert len(fused) == 1
    assert fused[0].W[0, 0] == 10**8
    for x_val in (0, 1):
        x = np.array([x_val])
        assert np.array_equal(evaluate(fused, x), evaluate(net, x))


def test_bound_driven_passes_decline_when_the_net_can_overflow_int64():
    # A net whose own forward pass overflows makes its ReLU decisions on
    # wrapped values, while preact_bounds describes the unwrapped ones. No
    # bound-driven rewrite is sound there, so both passes must stand down --
    # including at layer 0, whose own bounds still look small.
    big = 10**9
    net = [
        linear([[big, big]], [0]),
        linear([[big]], [0]),
        linear([[big]], [0]),
        linear([[big]], [0]),
    ]
    assert len(reduce_with_bounds(net, 0, 1)) == len(net)
    assert remove_unreachable_neurons(net, 0, 1)[0].W.shape == net[0].W.shape
    assert len(fuse_linear_layers(net, 0, 1)) == len(net)


def test_bound_driven_passes_reject_empty():
    with pytest.raises(ValueError, match="non-empty"):
        reduce_with_bounds([], 0, 1)
    with pytest.raises(ValueError, match="non-empty"):
        remove_unreachable_neurons([], 0, 1)
    with pytest.raises(ValueError, match="non-empty"):
        fuse_linear_layers([], 0, 1)
