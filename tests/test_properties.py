"""Randomised differential tests over generated nets.

Every other test file pins behaviour on hand-built fixtures. Those fixtures
share a blind spot: they use small, tidy weights (`1 << i` and friends), so
they never probe the magnitudes where integer arithmetic stops behaving like
arithmetic. This file generates nets instead, and checks the *properties* the
rest of the code rests on rather than specific values.

It earns its place: it found a real overflow bug in `fuse_linear_layers` that
all the fixture tests passed straight over — see `notes/04-reduce.md`.

Seeds are fixed, so a failure here is reproducible rather than a flake. If you
change a generator, expect the trial counts to shift and re-check runtime;
these are deliberately sized to keep the whole suite a few seconds.

Run just this file with:

    uv run pytest tests/test_properties.py -q
"""

from __future__ import annotations

import itertools
import random

import numpy as np
import pytest

from theorematic import Layer, evaluate
from theorematic.errors import IntegerOverflowError
from theorematic.features import ALWAYS, NEVER, SOMETIMES, UNKNOWN, describe_all
from theorematic.net import preact_bounds
from theorematic.reduce import reduce, reduce_with_bounds

OVERFLOW = "overflow"


def outcome(layers: list[Layer], x: np.ndarray):
    """The forward pass result, or a sentinel when it refuses to run.

    `evaluate` raises rather than wrapping past int64, so "what this net does
    at x" has three possible answers, not two. Equivalence between two nets
    means agreeing on which of the three.
    """
    try:
        return evaluate(layers, x)
    except IntegerOverflowError:
        return OVERFLOW


def same_outcome(a, b) -> bool:
    if a is OVERFLOW or b is OVERFLOW:
        return a is b
    return bool(np.array_equal(a, b))


# Weight magnitudes worth generating. 1e9 is the interesting one: two of them
# multiply to ~1e18, which is the same order as int64's ceiling, so fusion
# lands right on the boundary the guards exist to police.
SMALL_WEIGHTS = (2, 3, 8)
LARGE_WEIGHTS = (10**2, 10**4, 10**6, 10**9)


def random_net(rng: random.Random, n_in: int, depth: int, width: int, wmax: int) -> list[Layer]:
    """A random net with a single output, so `evaluate` returns one number."""
    layers: list[Layer] = []
    cur = n_in
    for i in range(depth):
        out = 1 if i == depth - 1 else rng.randint(1, width)
        W = np.array(
            [[rng.randint(-wmax, wmax) for _ in range(cur)] for _ in range(out)], dtype=np.int64
        )
        b = np.array([rng.randint(-wmax, wmax) for _ in range(out)], dtype=np.int64)
        layers.append(Layer(W=W, b=b))
        cur = out
    return layers


def splice_identity_layers(rng: random.Random, layers: list[Layer]) -> list[Layer]:
    """Insert identity layers at random interior positions.

    Random weights are never exactly `W=I, b=0`, so a purely random generator
    leaves `drop_identity_layers` untested — measured at 3 hits in 600 nets,
    which is not coverage. Identity layers are not an artificial case either:
    `construct.parallel` emits them as padding whenever it aligns branches of
    unequal depth, which is exactly where the pass earns its keep.

    Position 0 is included deliberately. `drop_identity_layers` must *not*
    remove a leading identity, because the raw input can be negative and the
    ReLU after it is therefore load-bearing. That rule is only observable when
    the input box actually reaches below zero — see `generated_cases`.
    """
    out = list(layers)
    for _ in range(rng.randint(0, 2)):
        if len(out) < 2:
            break
        position = rng.randint(0, len(out) - 1)
        width = out[position].in_features if position == 0 else out[position - 1].out_features
        identity = Layer(W=np.eye(width, dtype=np.int64), b=np.zeros(width, dtype=np.int64))
        out.insert(position, identity)
    return out


def input_box(n_in: int, lo: int, hi: int) -> list[np.ndarray]:
    return [np.array(p, dtype=np.int64) for p in itertools.product(range(lo, hi + 1), repeat=n_in)]


def generated_cases(seed: int, trials: int, weights: tuple[int, ...]):
    """Yield `(net, lo, hi, box)` tuples from a fixed seed.

    Some boxes reach below zero. Every hand-built fixture is binary or
    otherwise non-negative, so without this the whole project would never test
    a negative input — and several invariants (a leading ReLU mattering, bound
    propagation across a sign change) are invisible until one appears.
    """
    rng = random.Random(seed)
    for _ in range(trials):
        n_in = rng.randint(1, 3)
        lo = rng.choice([0, 0, 0, -1, -2])
        hi = lo + rng.choice([1, 1, 2, 3])
        net = random_net(rng, n_in, rng.randint(1, 4), 4, rng.choice(weights))
        yield splice_identity_layers(rng, net), lo, hi, input_box(n_in, lo, hi)


# --- what the bounds promise -------------------------------------------------


def test_preact_bounds_bracket_every_reachable_preactivation():
    """The soundness claim the whole bound-driven layer depends on."""
    for net, lo, hi, box in generated_cases(20260729, 600, SMALL_WEIGHTS):
        bounds = preact_bounds(net, lo, hi)
        for x in box:
            h = x
            for layer, (z_lo, z_hi) in zip(net, bounds):
                z = layer.W @ h + layer.b
                assert np.all(z_lo <= z), (z, z_lo)
                assert np.all(z <= z_hi), (z, z_hi)
                h = np.maximum(z, 0)


# --- what the rewrites promise -----------------------------------------------


def test_structural_reduce_preserves_evaluate():
    for net, lo, hi, box in generated_cases(11, 600, SMALL_WEIGHTS):
        small = reduce(net)
        for x in box:
            assert np.array_equal(evaluate(small, x), evaluate(net, x))


def test_bound_driven_reduce_preserves_evaluate_inside_the_box():
    for net, lo, hi, box in generated_cases(12, 600, SMALL_WEIGHTS):
        small = reduce_with_bounds(net, lo, hi)
        for x in box:
            assert np.array_equal(evaluate(small, x), evaluate(net, x))


@pytest.mark.parametrize("seed", [7, 1234, 99])
def test_bound_driven_reduce_survives_overflow_scale_weights(seed):
    """The regression this file was written for.

    Unguarded, weights at 1e9 made ~8% of reductions non-equivalent: fusion
    multiplies weights past int64 and numpy wrapped without a word. `evaluate`
    now raises instead of wrapping, so the invariant is that both nets agree on
    *which* outcome they produce, refusal included. A wrapped fused net returns
    a number where the original refuses, so this still catches the old bug.

    Also asserts fusion still fires, so a future over-cautious guard cannot
    pass this test by declining to do anything at all.
    """
    fired = 0
    for net, lo, hi, box in generated_cases(seed, 500, LARGE_WEIGHTS):
        small = reduce_with_bounds(net, lo, hi)
        if len(small) != len(net):
            fired += 1
        for x in box:
            assert same_outcome(outcome(small, x), outcome(net, x)), [l.W.tolist() for l in net]
    assert fired > 20, f"fusion fired only {fired} times; guards may be over-refusing"


@pytest.mark.parametrize("seed", [7, 1234])
def test_feature_verdicts_never_describe_wrapped_arithmetic(seed):
    """A neuron whose net cannot be evaluated must not get a confident verdict.

    `describe_neuron` decides constants from float64 bounds and everything else
    by enumerating with `evaluate`. Those two disagree once int64 is exceeded,
    so the module has to notice and return `"unknown"` rather than describing
    arithmetic that never happens.
    """
    for net, lo, hi, box in generated_cases(seed, 300, LARGE_WEIGHTS):
        overflows = any(outcome(net, x) is OVERFLOW for x in box)
        for f in describe_all(net, input_lo=lo, input_hi=hi):
            if overflows:
                assert f.verdict == UNKNOWN, (f.layer, f.neuron, f.verdict)
                continue
            prefix = net[: f.layer + 1]
            fires = [bool(evaluate(prefix, x)[f.neuron] > 0) for x in box]
            if f.verdict == NEVER:
                assert not any(fires)
            elif f.verdict == ALWAYS:
                assert all(fires)


# --- what the feature probes promise -----------------------------------------


def test_feature_verdicts_match_brute_force():
    for net, lo, hi, box in generated_cases(13, 400, SMALL_WEIGHTS):
        for f in describe_all(net, input_lo=lo, input_hi=hi):
            prefix = net[: f.layer + 1]
            fires = [bool(evaluate(prefix, x)[f.neuron] > 0) for x in box]
            if f.verdict == NEVER:
                assert not any(fires)
            elif f.verdict == ALWAYS:
                assert all(fires)
            elif f.verdict == SOMETIMES:
                assert any(fires) and not all(fires)


def test_coordinates_outside_support_never_change_a_preactivation():
    """`support` is exact as an exclusion — that is the half we rely on."""
    for net, lo, hi, box in generated_cases(14, 400, SMALL_WEIGHTS):
        n_in = net[0].in_features
        for f in describe_all(net, input_lo=lo, input_hi=hi):
            outside = [c for c in range(n_in) if c not in set(f.support)]
            if not outside:
                continue
            prefix = net[: f.layer + 1]
            base = np.full(n_in, lo, dtype=np.int64)
            reference = evaluate(prefix, base)[f.neuron]
            for coord in outside:
                for value in range(lo, hi + 1):
                    probe = base.copy()
                    probe[coord] = value
                    assert evaluate(prefix, probe)[f.neuron] == reference
