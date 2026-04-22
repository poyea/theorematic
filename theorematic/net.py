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
    h = x
    last = len(layers) - 1
    for i, layer in enumerate(layers):
        h = layer.W @ h + layer.b
        if final_relu or i != last:
            h = relu(h)
    return h
