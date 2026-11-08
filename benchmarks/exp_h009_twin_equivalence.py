#!/usr/bin/env python3
"""H009: is the compiled twin of the distributional trainers the same algorithm?

H008 ran DQN and Rainbow at the 1M benchmark's own reference point -- 64
environments, batch 32, replay ratio 8, one million transitions -- using the
`_torchcompile.py` twins.  DQN reproduced the benchmark almost exactly (68.24
against its published 68.0, and that agreement is the evidence that this
harness and its transition-axis scoring are sound).  Rainbow did not: 3.08
against a published 25.8, an 8.4x shortfall at settings that are identical on
paper.

The benchmark ran the *eager* files.  So either the twins diverge from their
originals, or something about this harness disagrees with the benchmark for
Rainbow specifically and not for DQN.  Running both variants at the same
reference point separates those.

    eager reproduces ~25.8, compiled does not  -> the twin is broken, and every
                                                  compiled-twin result for that
                                                  trainer in this repo is suspect
    both land near 3                           -> the benchmark's own number is
                                                  what needs explaining
    both land near 25.8                        -> something else in H007/H008
                                                  differs and must be found

Three trainers whose compiled twins carry the most weight elsewhere in the repo,
each at the exact published operating point.
"""

from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from exp_runner import run_experiment

BUDGET = 1_000_000
ENVS = 64
BATCH = 32
RATIO = 8.0
ALGOS = ("rainbow", "iqn", "fqf")
REFERENCE = {"rainbow": 25.8, "iqn": 18.9, "fqf": 17.8}


def specs():
    out = []
    for algo in ALGOS:
        for variant in ("eager", "torchcompile"):
            spec = {
                "name": f"{algo}-{variant}-ref",
                "algorithm": algo,
                "variant": variant,
                "num_envs": ENVS,
                "total_timesteps": BUDGET,
                "seed": 1,
                "batch_size": BATCH,
                "replay_ratio": RATIO,
                "planned_updates": int(BUDGET * RATIO / BATCH),
                "reference_1m": REFERENCE[algo],
                "flags": ["--batch-size", BATCH, "--replay-ratio", RATIO],
            }
            out.append(spec)
    # eager first: it is the arm that carries the published claim.
    return sorted(out, key=lambda s: (s["algorithm"], s["variant"]))


if __name__ == "__main__":
    results = run_experiment("h009_twin_equivalence", specs(), timeout=9000.0,
                             dry_run="--dry-run" in sys.argv)
    print()
    by = {}
    for r in results:
        if r.get("score") is not None:
            by.setdefault(r["algorithm"], {})[r["variant"]] = r["score"]
    print(f"{'algorithm':10s} {'published':>10s} {'eager':>9s} {'compiled':>9s} {'twin gap':>9s}")
    for algo in ALGOS:
        row = by.get(algo, {})
        e, c = row.get("eager"), row.get("torchcompile")
        gap = f"{e/max(c,1e-9):8.1f}x" if e and c else "        -"
        print(f"{algo:10s} {REFERENCE[algo]:10.1f} "
              f"{e if e is not None else float('nan'):9.2f} "
              f"{c if c is not None else float('nan'):9.2f} {gap}")
