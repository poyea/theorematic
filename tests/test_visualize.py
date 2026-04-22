import matplotlib.pyplot as plt
import numpy as np

from theorematic.fixtures import block_diagonal_net, equality_spike, xor_net
from theorematic.visualize import network_heatmaps, weight_heatmap, weight_stats


def test_weight_heatmap_returns_figure():
    W = np.arange(-6, 6).reshape(3, 4)
    fig = weight_heatmap(W, title="test")
    assert fig.axes, "figure should have axes"
    plt.close(fig)


def test_weight_heatmap_writes_file(tmp_path):
    W = np.eye(5, dtype=int)
    out = tmp_path / "eye.png"
    fig = weight_heatmap(W, path=out)
    plt.close(fig)
    assert out.exists() and out.stat().st_size > 0


def test_network_heatmaps_writes_one_per_layer(tmp_path):
    net = xor_net()
    paths = network_heatmaps(net, tmp_path)
    assert len(paths) == len(net)
    assert all(p.exists() and p.stat().st_size > 0 for p in paths)


def test_weight_stats_on_block_diagonal():
    net = block_diagonal_net([3, 2, 4])
    s = weight_stats(net[0])
    assert s["shape_out"] == 9
    assert s["shape_in"] == 9
    assert 0 < s["density"] < 1
    assert s["abs_max"] == 1


def test_weight_stats_on_equality_spike():
    net = equality_spike(target=5)
    s = weight_stats(net[0])
    # first layer W = [[-1],[-1],[-1]] — one unique value
    assert s["unique_values"] == 1
