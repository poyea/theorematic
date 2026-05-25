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
    reduce,
    remove_always_zero_neurons,
    remove_dead_neurons,
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
