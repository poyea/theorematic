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
