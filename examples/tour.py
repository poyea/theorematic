"""End-to-end tour of theorematic.

    uv run python examples/tour.py

Picks one fixture — a 3-bit "less than" comparator — and walks it through
every module the project currently provides: construct, evaluate, visualize,
invert, verify, reduce, interpret, extract features. Reading this is the
fastest way to see how the pieces fit together.
"""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np

from theorematic import evaluate
from theorematic.features import describe_all
from theorematic.fixtures import equality_spike, n_bit_less_than, one_hot_mux
from theorematic.ilp import invert as invert_ilp
from theorematic.interp import activation_profile, cluster_neurons_by_activation
from theorematic.reduce import reduce, reduce_with_bounds
from theorematic.sat import invert as invert_sat
from theorematic.visualize import activation_flow, network_heatmaps, weight_stats


def int_to_bits(value: int, n: int) -> np.ndarray:
    """LSB-first bit decomposition; matches the input layout the bit fixtures expect."""
    return np.array([(value >> i) & 1 for i in range(n)])


def bits_to_int(bits: np.ndarray, n: int) -> int:
    return int(sum(int(bits[i]) << i for i in range(n)))


def describe(layers: list) -> str:
    plural = "" if len(layers) == 1 else "s"
    return f"{len(layers)} layer{plural}: " + " ".join(
        f"{l.W.shape[0]}x{l.W.shape[1]}" for l in layers
    )


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

    # 3. Inversion — ask two solvers the same question and compare.
    print("\n-- inversion --")
    for name, invert in (("ILP (pulp/CBC)", invert_ilp), ("SAT (z3)", invert_sat)):
        r = invert(net, target=[1], input_lo=0, input_hi=1)
        assert r.feasible, f"{name} unexpectedly infeasible: {r.status}"
        a_recovered = bits_to_int(r.x[:n], n)
        b_recovered = bits_to_int(r.x[n:], n)
        out = int(evaluate(net, r.x)[0])
        print(
            f"  {name:<16}  a={a_recovered}, b={b_recovered}  " f"output={out}  (status={r.status})"
        )
        assert out == 1
        assert a_recovered < b_recovered
    print("  OK -- both solvers returned valid preimages")

    # 4. Reduction — structural first, then with a declared input domain.
    print("\n-- reduction --")
    print(f"  original            {describe(net)}")
    print(f"  reduce              {describe(reduce(net))}")
    print(f"  reduce_with_bounds  {describe(reduce_with_bounds(net, 0, 1))}")
    print("  a hand-built fixture is already tight, so both are no-ops -- as they should be.")

    # The payoff shows on a circuit with slack. A spike detector for x == 7
    # cannot fire when the input is restricted below 7, and bound propagation
    # proves it: the whole detector collapses to one linear layer.
    spike = equality_spike(7)
    collapsed = reduce_with_bounds(spike, 0, 6)
    print(f"\n  equality_spike(7)   {describe(spike)}")
    print(f"  under x in [0, 6]   {describe(collapsed)}")
    for x_val in range(7):
        assert np.array_equal(
            evaluate(collapsed, np.array([x_val])), evaluate(spike, np.array([x_val]))
        )
    print("  OK -- equivalent for every x in the declared domain (and only there)")

    # 5. Interpretation — sweep the whole input space and ask who fires.
    print("\n-- interpretation --")
    sweep = np.array(
        [
            np.concatenate([int_to_bits(a, n), int_to_bits(b, n)])
            for a, b in itertools.product(range(1 << n), repeat=2)
        ]
    )
    profile = activation_profile(net, sweep)
    print(f"  swept all {len(sweep)} inputs; activation rate per neuron:")
    for i, rates in enumerate(profile):
        kind = "output" if i == len(net) - 1 else "hidden"
        print(f"    layer {i} ({kind}): {np.round(rates, 2).tolist()}")

    clusters = cluster_neurons_by_activation(net, sweep)
    print(f"  hidden neurons fall into {len(clusters)} activation cluster(s):")
    for members in clusters.values():
        print(f"    {sorted(members)}")
    print("  neurons sharing a cluster fire on exactly the same inputs -- i.e. redundant.")

    # 6. Per-neuron features — what does each neuron compute, in terms of bits?
    print("\n-- per-neuron features --")
    for f in describe_all(net, input_lo=0, input_hi=1):
        fires = "-" if f.firing_inputs is None else str(len(f.firing_inputs))
        print(
            f"    neuron ({f.layer},{f.neuron}): {f.verdict:<9} "
            f"support={list(f.support)} fires_on={fires} patterns"
        )
    # Cross-check the count against the combinatorics rather than asserting a
    # magic number: pairs with a < b out of 2^n values is C(2^n, 2).
    values = 1 << n
    ordered_pairs = values * (values - 1) // 2
    first = describe_all(net, input_lo=0, input_hi=1)[0]
    assert first.firing_inputs is not None, "support small enough to enumerate at n=3"
    print(
        f"  neuron (0,0) fires on {len(first.firing_inputs)} of {len(sweep)} patterns; "
        f"C({values},2) = {ordered_pairs} is the count of pairs with a < b."
    )
    assert len(first.firing_inputs) == ordered_pairs

    # Support is where the projection back to input bits gets interesting: a
    # mux neuron touches only its own data/select pair, so the whole circuit
    # decomposes into independent gates.
    print("\n  one_hot_mux(3) -- same probe, a circuit that decomposes:")
    for f in describe_all(one_hot_mux(3), input_lo=0, input_hi=1):
        patterns = "-" if f.firing_inputs is None else [list(p) for p in f.firing_inputs]
        print(f"    neuron ({f.layer},{f.neuron}): support={list(f.support)} fires_on={patterns}")
    print("  each neuron sees one (data, select) pair and fires only on (1, 1) -- an AND gate.")


if __name__ == "__main__":
    main()
