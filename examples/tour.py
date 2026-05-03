"""End-to-end tour of theorematic.

    uv run python examples/tour.py

Picks one fixture — a 3-bit "less than" comparator — and walks it through
every module the project currently provides: construct, evaluate, visualize,
invert, verify. Reading this is the fastest way to see how the pieces fit
together.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from theorematic import evaluate
from theorematic.fixtures import n_bit_less_than
from theorematic.ilp import invert
from theorematic.visualize import activation_flow, network_heatmaps, weight_stats


def int_to_bits(value: int, n: int) -> np.ndarray:
    """LSB-first bit decomposition; matches the input layout the bit fixtures expect."""
    return np.array([(value >> i) & 1 for i in range(n)])


def bits_to_int(bits: np.ndarray, n: int) -> int:
    return int(sum(int(bits[i]) << i for i in range(n)))


def main() -> None:
    n = 3
    net = n_bit_less_than(n)

    print(f"=== n_bit_less_than(n={n}): output=1 iff a < b for two {n}-bit ints ===")
    print(f"network has {len(net)} layers:")
    for i, layer in enumerate(net):
        print(f"  layer {i}: W{layer.W.shape}  b{layer.b.shape}")

    # 1. Forward pass — concrete behaviour on three telling samples.
    print("\n-- forward pass --")
    for a, b in [(2, 5), (5, 5), (5, 2)]:
        x = np.concatenate([int_to_bits(a, n), int_to_bits(b, n)])
        y = int(evaluate(net, x)[0])
        print(f"  a={a:>2}, b={b:>2}  ->  output={y}   (a < b is {a < b})")

    # 2. Visualization — one PNG per layer + scalar stats.
    out_dir = Path("out/tour")
    paths = network_heatmaps(net, out_dir)
    print(f"\n-- visualization --\n  wrote {len(paths)} heatmaps to {out_dir.as_posix()}/")
    for i, layer in enumerate(net):
        s = weight_stats(layer)
        print(
            f"  layer {i}: {int(s['shape_out'])}x{int(s['shape_in'])}  "
            f"density={s['density']:.2f}  unique-values={int(s['unique_values'])}"
        )

    # 2b. Activation flow — show which neurons fire for a concrete sample.
    sample_a, sample_b = 2, 5
    x_sample = np.concatenate([int_to_bits(sample_a, n), int_to_bits(sample_b, n)])
    flow_path = out_dir / "activation_flow.png"
    activation_flow(net, x_sample, path=flow_path)
    print(f"  activation flow for a={sample_a}, b={sample_b} -> {flow_path.as_posix()}")

    # 3. Inversion — ask the MILP solver: find ANY (a, b) bit pattern with a < b.
    print("\n-- inversion (ILP) --")
    r = invert(net, target=[1], input_lo=0, input_hi=1)
    assert r.feasible, f"unexpectedly infeasible: {r.status}"
    a_recovered = bits_to_int(r.x[:n], n)
    b_recovered = bits_to_int(r.x[n:], n)
    print(f"  recovered preimage:  a={a_recovered}, b={b_recovered}  (bits={r.x.tolist()})")

    # 4. Verify — invert() already does this internally, but we make the check
    # visible here so the round-trip is part of the narrative.
    out = int(evaluate(net, r.x)[0])
    print(f"  evaluate(net, x) = {out}")
    assert out == 1
    assert a_recovered < b_recovered
    print("  OK -- preimage verified: a < b and the net outputs 1")


if __name__ == "__main__":
    main()
