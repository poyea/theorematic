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
