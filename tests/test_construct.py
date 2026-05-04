import numpy as np
import pytest

from theorematic import evaluate
from theorematic.construct import linear, parallel, route, stack
from theorematic.net import Layer


def test_linear_accepts_lists():
    layer = linear([[1, -1], [0, 2]], [0, 1])
    assert layer.W.shape == (2, 2)
    assert layer.b.tolist() == [0, 1]


def test_linear_rejects_floats_via_layer_invariant():
    # asarray(dtype=int) coerces; pass an actual non-int and confirm coercion.
    layer = linear([[1.0, 2.0]], [0])  # 1.0 -> 1, 2.0 -> 2
    assert layer.W.tolist() == [[1, 2]]


def test_route_permutation():
    layer = route([2, 0, 1], n_in=3)
    x = np.array([10, 20, 30])
    assert evaluate([layer], x).tolist() == [30, 10, 20]


def test_route_copy_and_project():
    # output 0 = input 1, output 1 = input 1, output 2 = input 0
    layer = route([1, 1, 0], n_in=3)
    x = np.array([5, 7, 9])
    assert evaluate([layer], x).tolist() == [7, 7, 5]


def test_route_rejects_out_of_range():
    with pytest.raises(ValueError, match="out of range"):
        route([0, 5], n_in=3)


def test_stack_concatenates_layers():
    a = [linear([[1, 0]], [0])]  # 2 -> 1
    b = [linear([[2]], [3])]  # 1 -> 1
    composed = stack(a, b)
    assert len(composed) == 2
    x = np.array([4, 9])
    # a(x) = 4, b(4) = 2*4 + 3 = 11
    assert evaluate(composed, x).tolist() == [11]


def test_stack_rejects_width_mismatch():
    a = [linear([[1, 0]], [0])]  # emits width 1
    b = [linear([[1, 1]], [0])]  # expects width 2
    with pytest.raises(ValueError, match="width mismatch"):
        stack(a, b)


def test_stack_skips_empty_segments():
    a = [linear([[1]], [0])]
    composed = stack(a, [], a)
    assert len(composed) == 2


def test_parallel_two_disjoint_identities():
    # input: [a, b, c, d]; left branch identity on [a,b], right on [c,d]
    left = [linear(np.eye(2, dtype=int), [0, 0])]
    right = [linear(np.eye(2, dtype=int), [0, 0])]
    p = parallel(left, right)
    assert len(p) == 1
    assert p[0].W.shape == (4, 4)
    assert evaluate(p, np.array([1, 2, 3, 4])).tolist() == [1, 2, 3, 4]


def test_parallel_pads_shorter_branch_with_identity():
    # Both branches pass non-negative values, identity padding is safe.
    deep = [linear([[1]], [0]), linear([[1]], [0])]  # 1 -> 1 -> 1, total 2 layers
    shallow = [linear([[1]], [0])]  # 1 -> 1, single layer
    p = parallel(deep, shallow)
    assert len(p) == 2  # padded to deeper branch
    out = evaluate(p, np.array([3, 5]))
    assert out.tolist() == [3, 5]


def test_parallel_block_diagonal_structure():
    a = [linear([[2, 0], [0, 3]], [0, 0])]
    b = [linear([[5]], [1])]
    p = parallel(a, b)
    W = p[0].W
    # off-diagonal blocks are zero
    assert W[:2, 2:].sum() == 0
    assert W[2:, :2].sum() == 0
    # input [1, 1, 1] -> [2, 3, 6]
    assert evaluate(p, np.array([1, 1, 1])).tolist() == [2, 3, 6]


def test_parallel_rejects_empty():
    with pytest.raises(ValueError, match="at least one"):
        parallel()


def test_parallel_rejects_empty_branch():
    with pytest.raises(ValueError, match="empty branches"):
        parallel([linear([[1]], [0])], [])


def test_stack_then_parallel_composes_cleanly():
    """The architectural payoff: build a small circuit by composition."""
    # double each input independently, then sum the two doubled values.
    branch = [linear([[2]], [0])]
    doubler = parallel(branch, branch)
    summer = [linear([[1, 1]], [0])]
    net = stack(doubler, summer)
    # input [3, 4] -> doublers -> [6, 8] -> summer -> 14
    assert evaluate(net, np.array([3, 4])).tolist() == [14]
