"""Weight visualization.

The first question to ask of a mystery network is: what do the weight matrices
look like? Block structure, sparsity, repeats, and symmetries are often
obvious once plotted and invisible in a raw dump.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from theorematic.net import Layer, evaluate, relu


def weight_heatmap(
    W: np.ndarray,
    b: np.ndarray | None = None,
    *,
    title: str | None = None,
    path: str | Path | None = None,
    symmetric: bool = True,
) -> plt.Figure:
    """Heatmap of a weight matrix with an optional bias bar chart.

    Pass `b` to add a right-hand panel showing per-neuron biases.
    `symmetric=True` centers the colormap at zero so sign is readable.
    """
    ncols = 2 if b is not None else 1
    width_ratios = [max(1, W.shape[1]), 1] if b is not None else [1]
    fig, axes = plt.subplots(
        1,
        ncols,
        figsize=(max(3, W.shape[1] * 0.35 + (1.5 if b is not None else 0)), max(3, W.shape[0] * 0.35)),
        gridspec_kw={"width_ratios": width_ratios} if ncols > 1 else None,
    )
    ax_w = axes[0] if ncols > 1 else axes

    vmax = float(np.max(np.abs(W))) if symmetric and W.size else 1.0
    vmin = -vmax if symmetric else None
    im = ax_w.imshow(W, cmap="RdBu_r" if symmetric else "viridis", vmin=vmin, vmax=vmax, aspect="auto")
    ax_w.set_xlabel("input")
    ax_w.set_ylabel("output")
    if title:
        ax_w.set_title(title)
    fig.colorbar(im, ax=ax_w, fraction=0.046, pad=0.04)

    if b is not None:
        ax_b = axes[1]
        n_out = len(b)
        ys = np.arange(n_out)
        colors = ["#d62728" if v > 0 else "#1f77b4" for v in b]
        ax_b.barh(ys, b, color=colors, height=0.7)
        ax_b.axvline(0, color="black", linewidth=0.8)
        ax_b.set_ylim(-0.5, n_out - 0.5)
        ax_b.invert_yaxis()
        ax_b.set_xlabel("bias")
        ax_b.set_yticks([])
        ax_b.set_title("b")
        bmax = float(np.max(np.abs(b))) if b.size else 1.0
        ax_b.set_xlim(-bmax * 1.3 - 0.5, bmax * 1.3 + 0.5)

    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=120)
    return fig


def network_heatmaps(layers: list[Layer], out_dir: str | Path) -> list[Path]:
    """Save a heatmap+bias panel per layer. Returns the list of written paths."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for i, layer in enumerate(layers):
        p = out / f"layer_{i:02d}.png"
        fig = weight_heatmap(layer.W, layer.b, title=f"layer {i}: W{layer.W.shape}", path=p)
        plt.close(fig)
        written.append(p)
    return written


def weight_stats(layer: Layer) -> dict[str, float]:
    """Scalar summaries that are worth eyeballing before plotting."""
    W = layer.W
    total = W.size
    nz = int(np.count_nonzero(W))
    return {
        "shape_out": float(W.shape[0]),
        "shape_in": float(W.shape[1]),
        "density": nz / total if total else 0.0,
        "min": float(W.min()) if total else 0.0,
        "max": float(W.max()) if total else 0.0,
        "abs_max": float(np.max(np.abs(W))) if total else 0.0,
        "unique_values": float(len(np.unique(W))),
    }


def activation_flow(
    layers: list[Layer],
    x: np.ndarray,
    *,
    path: str | Path | None = None,
) -> plt.Figure:
    """Bar-chart of pre-activation and post-ReLU values at each layer for input x.

    Each row is one layer. Left panel: pre-activation (Wx+b). Right panel:
    post-ReLU (clamped at 0). The final layer has no ReLU by convention, so its
    right panel mirrors the left.

    Dead neurons (post-ReLU == 0 but pre-activation < 0) are shown in grey;
    active neurons in steelblue; the final layer in a neutral green.
    """
    n_layers = len(layers)
    fig, axes = plt.subplots(
        n_layers,
        2,
        figsize=(10, max(2, n_layers * 1.8)),
        squeeze=False,
    )

    current = x.astype(float)
    for i, layer in enumerate(layers):
        pre = layer.W @ current + layer.b
        is_final = i == n_layers - 1
        post = pre if is_final else relu(pre)

        ax_pre, ax_post = axes[i, 0], axes[i, 1]
        xs = np.arange(len(pre))

        # pre-activation
        pre_colors = ["#d62728" if v > 0 else "#aec7e8" for v in pre]
        ax_pre.bar(xs, pre, color=pre_colors, width=0.7)
        ax_pre.axhline(0, color="black", linewidth=0.6)
        ax_pre.set_ylabel(f"L{i} pre")
        ax_pre.set_xticks(xs)

        # post-ReLU (or raw for final layer)
        if is_final:
            post_colors = ["#2ca02c"] * len(post)
            label = f"L{i} out"
        else:
            post_colors = ["#1f77b4" if v > 0 else "#cccccc" for v in post]
            label = f"L{i} post"
        ax_post.bar(xs, post, color=post_colors, width=0.7)
        ax_post.axhline(0, color="black", linewidth=0.6)
        ax_post.set_ylabel(label)
        ax_post.set_xticks(xs)

        if i == 0:
            ax_pre.set_title("pre-activation (Wx+b)")
            ax_post.set_title("post-ReLU  [grey = dead]")

        current = post

    fig.tight_layout()
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=120)
    return fig
