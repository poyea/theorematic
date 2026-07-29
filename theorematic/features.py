"""Per-neuron feature extraction: what does each neuron actually compute?

`interp` answers this empirically — sweep some inputs, see who fires. This
module answers it structurally, per neuron, and projects the answer back onto
input coordinates.

Three ideas, each building on the last:

- `preact_net(layers, i)` — the prefix `layers[:i+1]`. Because the final layer
  of any net is linear by convention, evaluating the prefix yields layer `i`'s
  *pre-activations* directly. No re-encoding, no solver: the convention does
  the work.
- `input_support(layers, i, k)` — the input coordinates that can influence
  neuron `(i,k)`, found by walking non-zero weights backwards. Coordinates
  outside the support provably cannot change the neuron's pre-activation.
- `describe_neuron(layers, i, k, ...)` — the feature itself: is the neuron
  constant or conditional, and if conditional, exactly which input patterns
  light it up.

Why enumeration and not a SAT query per neuron: restricted to its support, a
neuron's domain is usually tiny, and enumerating it gives an *exact* answer
plus the full firing set, where a solver would give one witness. It also
keeps this module free of a solver dependency, so it does not need to reach
into `sat.py` (siblings must not import each other). The cost is that a
wide-support neuron over a wide domain is not enumerable; `max_evaluations`
bounds the work and the verdict degrades to `"unknown"` rather than lying.
That is the case where a SAT query would earn its keep.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np

from theorematic.net import Layer, bounds_are_exact, evaluate, preact_bounds

NEVER = "never"
ALWAYS = "always"
SOMETIMES = "sometimes"
UNKNOWN = "unknown"


@dataclass(frozen=True)
class NeuronFeature:
    """What neuron `(layer, neuron)` computes over a declared input domain.

    `verdict` is one of:

    - `"never"`   — the neuron cannot fire anywhere in the domain.
    - `"always"`  — it fires everywhere, so its ReLU acts linearly.
    - `"sometimes"` — it fires on some inputs and not others; `firing_inputs`
      holds exactly which, as patterns over `support`.
    - `"unknown"` — the domain was too large to enumerate and interval bounds
      were inconclusive. Nothing is claimed.

    `support` is the set of input coordinates that can influence the neuron.
    `essential` narrows that to coordinates that *do* change the outcome
    somewhere — a coordinate can sit in the support with a non-zero weight
    path and still never flip the firing decision.

    Two absence conventions, which are not the same thing:

    - `firing_inputs is None` means no enumeration ran. An empty tuple means
      it ran and found nothing, which only happens alongside `"never"`.
    - `essential is None` means the outcome is undetermined, i.e. `"unknown"`.
      A constant neuron gets `()` whether bounds or enumeration settled it:
      something that never varies depends on nothing.
    """

    layer: int
    neuron: int
    verdict: str
    support: tuple[int, ...]
    essential: tuple[int, ...] | None
    firing_inputs: tuple[tuple[int, ...], ...] | None

    @property
    def is_constant(self) -> bool:
        return self.verdict in (NEVER, ALWAYS)

    def __repr__(self) -> str:
        parts = [f"({self.layer},{self.neuron})", self.verdict, f"support={list(self.support)}"]
        if self.essential is not None:
            parts.append(f"essential={list(self.essential)}")
        if self.firing_inputs is not None:
            parts.append(f"fires_on={len(self.firing_inputs)}")
        return "NeuronFeature(" + ", ".join(parts) + ")"


def _validate_layer(layers: list[Layer], layer_index: int) -> None:
    if not layers:
        raise ValueError("layers must be non-empty")
    if not 0 <= layer_index < len(layers):
        raise ValueError(f"layer_index {layer_index} out of range for {len(layers)} layers")


def _validate_position(layers: list[Layer], layer_index: int, neuron_index: int) -> None:
    _validate_layer(layers, layer_index)
    width = layers[layer_index].out_features
    if not 0 <= neuron_index < width:
        raise ValueError(f"neuron_index {neuron_index} out of range for layer width {width}")


def preact_net(layers: list[Layer], layer_index: int) -> list[Layer]:
    """The prefix whose linear output is layer `layer_index`'s pre-activation.

    `evaluate(preact_net(layers, i), x)[k]` equals `Wx + b` for neuron
    `(i, k)` — the value *before* its ReLU. This works because `evaluate`
    leaves the last layer linear, and the prefix's last layer is layer `i`.
    """
    _validate_layer(layers, layer_index)
    return list(layers[: layer_index + 1])


def input_support(layers: list[Layer], layer_index: int, neuron_index: int) -> tuple[int, ...]:
    """Input coordinates that can influence neuron `(layer_index, neuron_index)`.

    Walks backwards from the neuron, keeping any coordinate joined to a
    reached neuron by a non-zero weight. A coordinate outside the result has
    an all-zero weight path, so changing it cannot move the pre-activation —
    that direction is exact. The converse is an over-approximation: presence
    in the support does not prove influence, only the possibility of it.
    """
    _validate_position(layers, layer_index, neuron_index)
    reached = np.zeros(layers[layer_index].out_features, dtype=bool)
    reached[neuron_index] = True
    for layer in reversed(layers[: layer_index + 1]):
        reached = np.any(layer.W[reached] != 0, axis=0)
    return tuple(int(c) for c in np.flatnonzero(reached))


def describe_neuron(
    layers: list[Layer],
    layer_index: int,
    neuron_index: int,
    *,
    input_lo: int = 0,
    input_hi: int = 1,
    max_evaluations: int = 4096,
) -> NeuronFeature:
    """Characterise one neuron over the input box `[input_lo, input_hi]`.

    Tries interval bounds first — they settle the constant cases outright and
    cost one propagation. If they are inconclusive, enumerates the neuron's
    support (holding non-support coordinates at `input_lo`, which is sound
    because they cannot influence the result) and returns the exact firing
    set. Enumeration is skipped when the domain exceeds `max_evaluations`,
    leaving the verdict `"unknown"`.

    `max_evaluations` caps the forward passes, which dominate. Deriving
    `essential` afterwards costs a further pass over the same domain per
    support coordinate, but those are set lookups, not evaluations.
    """
    _validate_position(layers, layer_index, neuron_index)
    if input_lo > input_hi:
        raise ValueError(f"input_lo ({input_lo}) > input_hi ({input_hi})")

    support = input_support(layers, layer_index, neuron_index)
    bounds = preact_bounds(layers, input_lo, input_hi)
    z_lo, z_hi = bounds[layer_index]

    def feature(verdict: str, essential=None, firing=None) -> NeuronFeature:
        return NeuronFeature(
            layer=layer_index,
            neuron=neuron_index,
            verdict=verdict,
            support=support,
            essential=essential,
            firing_inputs=firing,
        )

    # The two decision paths below use different arithmetic: bounds in float64,
    # enumeration via `evaluate` in int64. They agree only while the net stays
    # inside int64, so past that point neither answer is about the same
    # function and the only honest verdict is that there isn't one.
    if not bounds_are_exact(bounds):
        return feature(UNKNOWN)

    if z_hi[neuron_index] <= 0:
        return feature(NEVER, essential=())
    if z_lo[neuron_index] > 0:
        return feature(ALWAYS, essential=())

    domain = range(int(input_lo), int(input_hi) + 1)
    if len(domain) ** len(support) > max_evaluations:
        return feature(UNKNOWN)

    prefix = preact_net(layers, layer_index)
    n_in = layers[0].in_features
    firing: list[tuple[int, ...]] = []
    for pattern in itertools.product(domain, repeat=len(support)):
        x = np.full(n_in, int(input_lo), dtype=int)
        for coord, value in zip(support, pattern):
            x[coord] = value
        if evaluate(prefix, x)[neuron_index] > 0:
            firing.append(pattern)

    if not firing:
        return feature(NEVER, essential=(), firing=())
    if len(firing) == len(domain) ** len(support):
        return feature(ALWAYS, essential=(), firing=tuple(firing))

    fires = set(firing)
    essential = tuple(
        coord
        for position, coord in enumerate(support)
        if _flips_outcome(fires, position, domain, len(support))
    )
    return feature(SOMETIMES, essential=essential, firing=tuple(firing))


def _flips_outcome(fires: set[tuple[int, ...]], position: int, domain: range, width: int) -> bool:
    """Does varying `position` alone change the firing outcome anywhere?

    Iterates assignments to the *other* coordinates and sweeps `position`
    across the domain for each, so every equivalence class is visited once.
    """
    for others in itertools.product(domain, repeat=width - 1):
        probes = (others[:position] + (value,) + others[position:] for value in domain)
        if len({probe in fires for probe in probes}) > 1:
            return True
    return False


def describe_all(
    layers: list[Layer],
    *,
    input_lo: int = 0,
    input_hi: int = 1,
    max_evaluations: int = 4096,
    hidden_only: bool = True,
) -> list[NeuronFeature]:
    """Characterise every neuron, in `(layer, neuron)` order.

    `hidden_only` skips the final layer, whose output is linear and signed —
    "fires" is not meaningful there, the same stance
    `interp.cluster_neurons_by_activation` takes. Set it to `False` to get
    the final layer described on the raw `> 0` rule anyway.
    """
    if not layers:
        raise ValueError("layers must be non-empty")
    limit = len(layers) - 1 if hidden_only else len(layers)
    return [
        describe_neuron(
            layers,
            i,
            k,
            input_lo=input_lo,
            input_hi=input_hi,
            max_evaluations=max_evaluations,
        )
        for i in range(limit)
        for k in range(layers[i].out_features)
    ]
