import numpy as np
import pytest

from theorematic import Layer, evaluate, relu


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
