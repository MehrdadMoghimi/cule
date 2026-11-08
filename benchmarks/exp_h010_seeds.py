#!/usr/bin/env python3
"""H010: replicate the rescue, and settle the non-monotone middle column.

H007 found that batch 32 beats batch 512 for all five collapsed trainers, which
was the registered prediction and is the result F011 rests on.  It also found
something a monotone mechanism cannot produce: batch 128 scored worse than BOTH
32 and 512 for four of the five (IQN 0.54 against 6.88 and 1.38).  Either the
seed-to-seed spread at these score levels is comparable to the effect, in which
case F011 is over-claimed, or batch size does something non-monotone here.

Three seeds at each of the three batch sizes, on the two trainers with the
largest H007 effects.  The question the design answers is not "which batch is
best" but "is the batch-32-over-batch-512 gap larger than the spread between
seeds of the same configuration".
"""

from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import statistics
from exp_runner import run_experiment

BUDGET = 1_000_000
ENVS = 1024
RATIO = 8.0
BATCHES = (32, 128, 512)
SEEDS = (1, 2, 3)
ALGOS = ("iqn", "fqf")
BUFFER = 131_072


def specs():
    out = []
    for algo in ALGOS:
        for batch in BATCHES:
            for seed in SEEDS:
                out.append({
                    "name": f"{algo}-b{batch}-s{seed}",
                    "algorithm": algo,
                    "variant": "torchcompile",
                    "num_envs": ENVS,
                    "total_timesteps": BUDGET,
                    "seed": seed,
                    "batch_size": batch,
                    "replay_ratio": RATIO,
                    "planned_updates": int(BUDGET * RATIO / batch),
                    "flags": ["--batch-size", batch, "--replay-ratio", RATIO,
                              "--buffer-size", BUFFER],
                })
    # One full seed sweep of the cheapest cell first, so an interrupted run still
    # has a spread estimate rather than three quarters of one condition.
    return sorted(out, key=lambda s: (-s["batch_size"], s["algorithm"], s["seed"]))


if __name__ == "__main__":
    results = run_experiment("h010_seeds", specs(), timeout=7200.0,
                             dry_run="--dry-run" in sys.argv)
    print()
    cells: dict = {}
    for r in results:
        if r.get("score") is not None:
            cells.setdefault((r["algorithm"], r["batch_size"]), []).append(r["score"])
    print(f"{'algorithm':10s} {'batch':>6s} {'n':>3s} {'mean':>7s} {'min':>7s} {'max':>7s} {'spread':>7s}")
    for algo in ALGOS:
        for batch in BATCHES:
            v = cells.get((algo, batch), [])
            if not v:
                continue
            print(f"{algo:10s} {batch:6d} {len(v):3d} {statistics.mean(v):7.2f} "
                  f"{min(v):7.2f} {max(v):7.2f} {max(v)-min(v):7.2f}")
    print()
    for algo in ALGOS:
        lo, hi = cells.get((algo, 512), []), cells.get((algo, 32), [])
        mid = cells.get((algo, 128), [])
        if lo and hi:
            effect = statistics.mean(hi) - statistics.mean(lo)
            noise = max((max(v) - min(v)) for v in (lo, hi, mid) if v)
            print(f"{algo}: b32-over-b512 effect {effect:+.2f}, largest within-cell spread {noise:.2f} "
                  f"-> {'SURVIVES' if effect > noise else 'DOES NOT survive'} the seed noise")
