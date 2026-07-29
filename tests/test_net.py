import numpy as np
import pytest

from theorematic import Layer, evaluate, relu
from theorematic.net import preact_bounds


def test_relu_clips_negatives():
    x = np.array([-3, -1, 0, 1, 4])
    assert np.array_equal(relu(x), np.array([0, 0, 0, 1, 4]))


def test_single_layer_identity():
    layer = Layer(W=np.eye(3, dtype=int), b=np.zeros(3, dtype=int))
    x = np.array([2, -1, 5])
    assert np.array_equal(evaluate([layer], x), x)


def test_single_layer_with_final_relu():
    layer = Layer(W=np.eye(3, dtype=int), b=np.zeros(3, dtype=int))
    x = np.array([2, -1, 5])
    assert np.array_equal(evaluate([layer], x, final_relu=True), np.array([2, 0, 5]))


def test_two_layers_compose():
    l1 = Layer(W=2 * np.eye(2, dtype=int), b=np.zeros(2, dtype=int))
    l2 = Layer(W=np.eye(2, dtype=int), b=np.ones(2, dtype=int))
    x = np.array([3, 4])
    assert np.array_equal(evaluate([l1, l2], x), np.array([7, 9]))


def test_shape_validation():
    with pytest.raises(ValueError):
        Layer(W=np.zeros((2, 3), dtype=int), b=np.zeros(3, dtype=int))


def test_layer_rejects_float_weights():
    with pytest.raises(TypeError, match="integer dtype"):
        Layer(W=np.eye(2), b=np.zeros(2, dtype=int))  # W defaults to float64


def test_layer_rejects_float_bias():
    with pytest.raises(TypeError, match="integer dtype"):
        Layer(W=np.eye(2, dtype=int), b=np.zeros(2))  # b defaults to float64


def test_evaluate_rejects_wrong_input_width():
    layer = Layer(W=np.eye(3, dtype=int), b=np.zeros(3, dtype=int))
    with pytest.raises(ValueError, match="input width"):
        evaluate([layer], np.array([1, 2]))


def test_evaluate_rejects_non_1d_input():
    layer = Layer(W=np.eye(3, dtype=int), b=np.zeros(3, dtype=int))
    with pytest.raises(ValueError, match="1-D"):
        evaluate([layer], np.array([[1, 2, 3]]))


def test_evaluate_rejects_empty_layer_list():
    with pytest.raises(ValueError, match="non-empty"):
        evaluate([], np.array([1, 2, 3]))


def test_preact_bounds_scalar_and_array_bounds_agree():
    net = [Layer(W=np.array([[1, -1]]), b=np.array([0]))]
    scalar = preact_bounds(net, 0, 1)
    array = preact_bounds(net, np.zeros(2, dtype=int), np.ones(2, dtype=int))
    assert np.array_equal(scalar[0][0], array[0][0])
    assert np.array_equal(scalar[0][1], array[0][1])


def test_preact_bounds_brackets_every_reachable_preact():
    net = [Layer(W=np.array([[1, -1], [2, 1]]), b=np.array([0, -1]))]
    z_lo, z_hi = preact_bounds(net, 0, 1)[0]
    for a in (0, 1):
        for b in (0, 1):
            z = net[0].W @ np.array([a, b]) + net[0].b
            assert np.all(z_lo <= z) and np.all(z <= z_hi)


def test_preact_bounds_clips_at_zero_between_layers():
    # Layer 0 can go negative; layer 1 must see the post-ReLU range [0, 1].
    net = [
        Layer(W=np.array([[1, -1]]), b=np.array([0])),
        Layer(W=np.array([[1]]), b=np.array([0])),
    ]
    bounds = preact_bounds(net, 0, 1)
    assert bounds[0][0] == -1
    assert bounds[1][0] == 0 and bounds[1][1] == 1


def test_preact_bounds_rejects_empty_layer_list():
    with pytest.raises(ValueError, match="non-empty"):
        preact_bounds([], 0, 1)


def test_preact_bounds_rejects_inverted_bounds():
    net = [Layer(W=np.eye(2, dtype=int), b=np.zeros(2, dtype=int))]
    with pytest.raises(ValueError, match="input_lo exceeds input_hi"):
        preact_bounds(net, 1, 0)


def test_layer_repr_shows_shapes_not_arrays():
    layer = Layer(W=np.zeros((3, 5), dtype=int), b=np.zeros(3, dtype=int))
    r = repr(layer)
    assert "(3, 5)" in r and "(3,)" in r
    # the full array dump should not appear
    assert "[[0" not in r
