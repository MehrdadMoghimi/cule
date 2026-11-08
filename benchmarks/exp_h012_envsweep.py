#!/usr/bin/env python3
"""H012: firm up the environment-count penalty, which currently rests on six runs.

F010 measured a 4.6x cost for DQN from 64 to 1,024 environments with batch size,
replay ratio, optimizer steps and transition budget all pinned.  F014 then showed
that the same penalty is NOT identifiable in the 241-run pooled regression, because
every highenv run is at 1,024 environments and every 1M-benchmark run is below it,
so environment count and protocol are nearly collinear there.  The controlled
experiment is the only evidence, and it is one seed at three counts.

This widens it to five counts and two seeds, and adds a second trainer that is
known to work at scale (IB-DQN, which was the one algorithm to IMPROVE when the
highenv sweep moved it to 1,024 environments).  If the penalty is real it should
appear for both; if it appears only for DQN it is a property of that trainer
rather than of parallelism.
"""

from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import statistics
from exp_runner import run_experiment

BUDGET = 1_000_000
BATCH = 32
RATIO = 8.0
ENV_COUNTS = (64, 128, 256, 512, 1024)
SEEDS = (1, 2)
ALGOS = ("dqn", "ibdqn")
BUFFER = 131_072


def specs():
    out = []
    for algo in ALGOS:
        for envs in ENV_COUNTS:
            for seed in SEEDS:
                out.append({
                    "name": f"{algo}-n{envs}-s{seed}",
                    "algorithm": algo, "variant": "torchcompile",
                    "num_envs": envs, "total_timesteps": BUDGET, "seed": seed,
                    "batch_size": BATCH, "replay_ratio": RATIO,
                    "planned_updates": int(BUDGET * RATIO / BATCH),
                    "updates_between_vector_steps": int(RATIO * envs / BATCH),
                    "flags": ["--batch-size", BATCH, "--replay-ratio", RATIO,
                              "--buffer-size", BUFFER],
                })
    # Seed 1 across the whole ladder first: a partial run still gives the shape.
    return sorted(out, key=lambda s: (s["seed"], s["algorithm"], -s["num_envs"]))


if __name__ == "__main__":
    results = run_experiment("h012_envsweep", specs(), timeout=9000.0,
                             dry_run="--dry-run" in sys.argv)
    print()
    cells: dict = {}
    for r in results:
        if r.get("score") is not None:
            cells.setdefault((r["algorithm"], r["num_envs"]), []).append(r["score"])
    print(f"{'algorithm':10s}" + "".join(f"{('n%d'%n):>11s}" for n in ENV_COUNTS))
    for algo in ALGOS:
        row = "".join(
            (f"{statistics.mean(cells[(algo,n)]):11.2f}" if (algo, n) in cells else f"{'-':>11s}")
            for n in ENV_COUNTS)
        print(f"{algo:10s}{row}")
        spreads = [max(v) - min(v) for n in ENV_COUNTS
                   if len((v := cells.get((algo, n), []))) > 1]
        if spreads:
            print(f"{'':10s}  largest within-cell seed spread: {max(spreads):.2f}")
