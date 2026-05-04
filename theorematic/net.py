"""Integer-weighted ReLU MLP: the one evaluator used everywhere."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Layer:
    W: np.ndarray
    b: np.ndarray

    def __post_init__(self) -> None:
        if self.W.ndim != 2 or self.b.ndim != 1 or self.W.shape[0] != self.b.shape[0]:
            raise ValueError(f"shape mismatch: W={self.W.shape}, b={self.b.shape}")
        if not np.issubdtype(self.W.dtype, np.integer) or not np.issubdtype(self.b.dtype, np.integer):
            raise TypeError(
                f"Layer requires integer dtype: W={self.W.dtype}, b={self.b.dtype}. "
                f"This project's reverse-engineering techniques assume a discrete weight alphabet."
            )

    @property
    def in_features(self) -> int:
        return int(self.W.shape[1])

    @property
    def out_features(self) -> int:
        return int(self.W.shape[0])


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0)


def evaluate(layers: list[Layer], x: np.ndarray, *, final_relu: bool = False) -> np.ndarray:
    """Forward pass. Last layer is linear unless `final_relu=True`."""
    if not layers:
        raise ValueError("layers must be non-empty")
    if x.ndim != 1:
        raise ValueError(f"x must be 1-D, got shape {x.shape}")
    expected = layers[0].in_features
    if x.shape[0] != expected:
        raise ValueError(f"input width {x.shape[0]} does not match layer 0 in_features={expected}")
    h = x
    last = len(layers) - 1
    for i, layer in enumerate(layers):
        h = layer.W @ h + layer.b
        if final_relu or i != last:
            h = relu(h)
    return h
