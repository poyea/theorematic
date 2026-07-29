# theorematic

Reverse-engineering integer-weighted ReLU networks: hand-built circuits,
inversion by ILP and SAT, equivalence-preserving reduction, and per-neuron
interpretability probes.

## Setup

Uses [uv](https://docs.astral.sh/uv/).

```
uv sync
```

## The tour

The fastest way to see the whole stack. Builds a 3-bit `a < b` comparator and
walks it through every module: construct, evaluate, visualize, invert with
both solvers, verify, reduce, interpret, extract per-neuron features.

```
uv run python examples/tour.py
```

Heatmap and activation PNGs land in `out/tour/`.

## Tests

```
uv run pytest                              # everything, ~10s
uv run pytest -q                           # quiet
uv run pytest tests/test_reduce.py         # one module
uv run pytest -k overflow                  # by name
uv run pytest --durations=10               # find the slow ones
```

### Property tests

`tests/test_properties.py` is different in kind from the rest. The other
files pin behaviour on hand-built fixtures; this one generates random nets and
checks the properties the code rests on: that `preact_bounds` really brackets
every reachable pre-activation, that both `reduce` drivers preserve
`evaluate`, that feature verdicts match brute force.

It is worth running on its own when touching `net.py`, `reduce.py`, or
`features.py`:

```
uv run pytest tests/test_properties.py -q
```

Seeds are fixed, so failures reproduce instead of flaking. These tests found a
real overflow bug in layer fusion that every fixture test passed over. The
fixtures use tidy weights like `1 << i`, so they never reach the magnitudes
where int64 arithmetic wraps.

The generator deliberately produces two things no fixture does: identity layers
(random weights are never exactly `W=I`, so the pass that removes them was
otherwise untested) and **negative** input boxes (every fixture is binary, so
without these the project had no test with a negative input at all).

If you add a rewrite, add a property here too, and check the property has
teeth by breaking the code on purpose and confirming this file goes red.

## Poking at it directly

Ask what each neuron computes. On a one-hot mux the circuit decomposes into
independent AND gates, recovered from the weights alone:

```
uv run python -c "from theorematic.features import describe_all; \
from theorematic.fixtures import one_hot_mux; \
[print(f) for f in describe_all(one_hot_mux(3))]"
```

Bound-driven reduction. A spike detector for `x == 7` provably collapses to a
single layer once you declare the input cannot reach 7:

```
uv run python -c "from theorematic.reduce import reduce_with_bounds; \
from theorematic.fixtures import equality_spike; \
print([l.W.shape for l in reduce_with_bounds(equality_spike(7), 0, 6)])"
```

Inversion. Find an input driving the output to 1, via MILP or SMT. The two
usually return different valid preimages:

```
uv run python -c "from theorematic.ilp import invert; \
from theorematic.fixtures import n_bit_less_than; \
print(invert(n_bit_less_than(3), target=[1], input_lo=0, input_hi=1))"
```

Visualise any fixture:

```
uv run python -c "from theorematic.visualize import network_heatmaps; \
from theorematic.fixtures import block_diagonal_net; \
network_heatmaps(block_diagonal_net([3,2,4]), 'out')"
```

## Formatting

```
uv run black theorematic tests examples
```

## License

MIT
