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

from theorematic.net import Layer


def weight_heatmap(
    W: np.ndarray,
    *,
    title: str | None = None,
    path: str | Path | None = None,
    symmetric: bool = True,
) -> plt.Figure:
    """Heatmap of a single weight matrix.

    `symmetric=True` centers the colormap at zero so sign is readable.
    """
    fig, ax = plt.subplots(figsize=(max(3, W.shape[1] * 0.25), max(3, W.shape[0] * 0.25)))
    vmax = float(np.max(np.abs(W))) if symmetric and W.size else 1.0
    vmin = -vmax if symmetric else None
    im = ax.imshow(W, cmap="RdBu_r" if symmetric else "viridis", vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xlabel("input")
    ax.set_ylabel("output")
    if title:
        ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=120)
    return fig


def network_heatmaps(layers: list[Layer], out_dir: str | Path) -> list[Path]:
    """Save a heatmap per layer. Returns the list of written paths."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for i, layer in enumerate(layers):
        p = out / f"layer_{i:02d}.png"
        fig = weight_heatmap(layer.W, title=f"layer {i}: {layer.W.shape}", path=p)
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
