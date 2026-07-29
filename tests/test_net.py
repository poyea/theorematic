import numpy as np
import pytest

from theorematic import Layer, evaluate, relu
from theorematic.errors import IntegerOverflowError
from theorematic.net import bounds_are_exact, preact_bounds


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


def test_evaluate_raises_rather_than_wrapping_on_overflow():
    # 1e10 * 1e10 * 2 is ~2e20, well past int64. numpy would wrap this to a
    # meaningless (and here, negative) value without complaint.
    big = 10**10
    net = [
        Layer(W=np.array([[big, big]]), b=np.array([0])),
        Layer(W=np.array([[big]]), b=np.array([0])),
    ]
    with pytest.raises(IntegerOverflowError, match="outside int64"):
        evaluate(net, np.array([1, 1]))


def test_overflow_error_names_the_offending_neuron():
    big = 10**10
    net = [
        Layer(W=np.array([[1, 0], [big, big]]), b=np.array([0, 0])),
        Layer(W=np.array([[0, big]]), b=np.array([0])),
    ]
    with pytest.raises(IntegerOverflowError, match=r"layer 1 neuron 0"):
        evaluate(net, np.array([1, 1]))


def test_evaluate_does_not_false_alarm_near_the_boundary():
    # Comfortably representable, so the guard must stay out of the way. 3e9
    # squared is ~9e18, just under int64's ceiling.
    net = [Layer(W=np.array([[3 * 10**9]]), b=np.array([0]))]
    assert evaluate(net, np.array([3 * 10**9]))[0] == 9 * 10**18


def test_evaluate_overflow_check_ignores_float_input():
    # Floats do not wrap, they lose precision, which is a different problem and
    # not this check's business.
    net = [Layer(W=np.array([[10**10]]), b=np.array([0]))]
    assert evaluate(net, np.array([1e10]))[0] == pytest.approx(1e20)


def test_bounds_are_exact_flags_when_float_bounds_stop_being_integers():
    small = [Layer(W=np.array([[2, 3]]), b=np.array([1]))]
    assert bounds_are_exact(preact_bounds(small, 0, 1))

    huge = [
        Layer(W=np.array([[10**9, 10**9]]), b=np.array([0])),
        Layer(W=np.array([[10**9]]), b=np.array([0])),
    ]
    assert not bounds_are_exact(preact_bounds(huge, 0, 1))


def test_layer_repr_shows_shapes_not_arrays():
    layer = Layer(W=np.zeros((3, 5), dtype=int), b=np.zeros(3, dtype=int))
    r = repr(layer)
    assert "(3, 5)" in r and "(3,)" in r
    # the full array dump should not appear
    assert "[[0" not in r
