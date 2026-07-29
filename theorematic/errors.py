"""Shared exception types.

Kept in their own module so any technique (ILP, SAT, …) can raise and any
test can catch the same type without creating sibling-module imports.
"""

from __future__ import annotations


class VerificationError(AssertionError):
    """A solver returned an input that does not actually drive the net to the
    requested target.

    Indicates a bug in the *encoding* — undersized big-M, wrong bit-blast
    width, a mistake in bound propagation — not a user error. Failing loudly
    is the point: every inversion technique re-runs the forward pass and
    raises this on mismatch.
    """


class IntegerOverflowError(ArithmeticError):
    """A forward pass exceeded the range of the integer dtype it runs in.

    `numpy` wraps silently on integer overflow, which is the worst possible
    behaviour for this project: every technique here reasons about a net's
    arithmetic using something *other* than a numpy forward pass — interval
    bounds in float64, an LP relaxation, z3's unbounded integers — and none of
    those wrap. Let a wrap go unnoticed and the reasoning describes a different
    function from the one `evaluate` computes.

    So `evaluate` detects it and raises. The right response is smaller weights
    or a shallower net, not a wider dtype: the project's premise is a discrete
    weight alphabet small enough to reason about exactly.
    """
