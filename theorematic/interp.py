"""Interpretability probes for integer-weighted ReLU nets.

Three small probes that sit one level above `visualize.activation_flow`.
Where `activation_flow` answers *which neurons fire for this single input*,
this module sweeps a collection of inputs and asks structural questions:

- `activation_profile(layers, inputs)` — over a sweep, how often does each
  neuron fire? Returns one row per layer; entry `[i][k]` is the fraction
  of `inputs` that drove neuron `k` of layer `i` above zero (post-ReLU for
  hidden layers, raw output for the final layer).
- `active_neurons(layers, x)` — for one input, the set of `(layer, neuron)`
  positions that are alive. The minimal "what's lit up" probe.
- `cluster_neurons_by_activation(layers, inputs)` — group **hidden**
  neurons by their boolean activation vector across the sweep. Neurons in
  the same cluster fire on exactly the same subset of inputs.

Conventions:

- The final layer is **linear** (no ReLU). `activation_profile` includes
  it but uses the threshold `> 0` on the raw output — the result is
  meaningful as "fraction of inputs that gave a positive answer" but is
  not an activation in the ReLU sense. The function's return type makes
  the final-layer row callable-out by index (last row); callers can
  slice it off.
- `cluster_neurons_by_activation` skips the final layer entirely:
  "fires/doesn't fire" is ill-defined for a linear signed output.
"""

from __future__ import annotations

import numpy as np

from theorematic.net import Layer, relu


def _require_nonempty(layers: list[Layer]) -> None:
    if not layers:
        raise ValueError("layers must be non-empty")


def _validate_input(layers: list[Layer], x: np.ndarray) -> None:
    if x.ndim != 1:
        raise ValueError(f"x must be 1-D, got shape {x.shape}")
    expected = layers[0].in_features
    if x.shape[0] != expected:
        raise ValueError(f"input width {x.shape[0]} does not match layer 0 in_features={expected}")


def _validate_inputs_sweep(layers: list[Layer], inputs: np.ndarray) -> np.ndarray:
    arr = np.asarray(inputs)
    if arr.ndim != 2:
        raise ValueError(f"inputs must be 2-D (n_inputs, width), got shape {arr.shape}")
    expected = layers[0].in_features
    if arr.shape[1] != expected:
        raise ValueError(
            f"input width {arr.shape[1]} does not match layer 0 in_features={expected}"
        )
    if arr.shape[0] == 0:
        raise ValueError("inputs must contain at least one row")
    return arr


def _layer_activations(layers: list[Layer], x: np.ndarray) -> list[np.ndarray]:
    """Per-layer output values for input `x`. Hidden = post-ReLU, final = linear."""
    outs: list[np.ndarray] = []
    h = x.astype(np.int64)
    last = len(layers) - 1
    for i, layer in enumerate(layers):
        pre = layer.W @ h + layer.b
        h = pre if i == last else relu(pre)
        outs.append(h)
    return outs


def activation_profile(layers: list[Layer], inputs: np.ndarray) -> list[np.ndarray]:
    """Fraction of inputs that activate each neuron, per layer.

    Returns a list of 1-D float arrays — one per layer. Entry `[i][k]` is
    the fraction of rows in `inputs` for which neuron `k` of layer `i` is
    `> 0` (post-ReLU for hidden layers, raw linear output for the final
    layer). "Always firing" is `profile[i][k] == 1.0`, "never firing" is
    `profile[i][k] == 0.0`.

    The final-layer row's interpretation is "fraction of inputs producing
    a positive output" — useful, but distinct from a ReLU activation rate.
    Callers who only care about hidden neurons should slice off `[-1]`.
    """
    _require_nonempty(layers)
    arr = _validate_inputs_sweep(layers, inputs)

    counts: list[np.ndarray] = [np.zeros(layer.out_features, dtype=np.int64) for layer in layers]
    for x in arr:
        for i, h in enumerate(_layer_activations(layers, x)):
            counts[i] += (h > 0).astype(np.int64)
    n = arr.shape[0]
    return [c.astype(float) / n for c in counts]


def active_neurons(layers: list[Layer], x: np.ndarray) -> set[tuple[int, int]]:
    """The set of `(layer_index, neuron_index)` positions firing on `x`.

    A neuron "fires" when its post-ReLU value is strictly positive. The
    final layer is included on the same `> 0` rule; callers concerned with
    only hidden neurons can filter by `layer_index < len(layers) - 1`.
    """
    _require_nonempty(layers)
    x = np.asarray(x)
    _validate_input(layers, x)
    active: set[tuple[int, int]] = set()
    for i, h in enumerate(_layer_activations(layers, x)):
        for k in np.flatnonzero(h > 0):
            active.add((i, int(k)))
    return active


def cluster_neurons_by_activation(
    layers: list[Layer], inputs: np.ndarray
) -> dict[tuple[int, ...], list[tuple[int, int]]]:
    """Group hidden neurons by their boolean activation vector across `inputs`.

    For each hidden neuron, build the tuple `(fires_on_input_0,
    fires_on_input_1, ...)`. Neurons with the same tuple end up in the
    same cluster. Useful for spotting redundant or symmetric neurons:
    two neurons sharing a cluster have identical computational role
    under the sweep, and one can be removed without behavioural change.

    The final layer is excluded: "fires/doesn't fire" is ill-defined for
    a linear signed output. If the network has a single layer, no
    clustering is possible and the result is `{}`.
    """
    _require_nonempty(layers)
    arr = _validate_inputs_sweep(layers, inputs)

    n_inputs = arr.shape[0]
    last = len(layers) - 1
    # patterns[(layer, neuron)] = bool vector of length n_inputs
    patterns: dict[tuple[int, int], np.ndarray] = {}
    for i in range(last):  # hidden layers only
        patterns_layer = np.zeros((n_inputs, layers[i].out_features), dtype=bool)
        for r, x in enumerate(arr):
            h = _layer_activations(layers, x)[i]
            patterns_layer[r] = h > 0
        for k in range(layers[i].out_features):
            patterns[(i, k)] = patterns_layer[:, k]

    clusters: dict[tuple[int, ...], list[tuple[int, int]]] = {}
    for key, vec in patterns.items():
        sig = tuple(bool(v) for v in vec)
        clusters.setdefault(sig, []).append(key)
    return clusters
