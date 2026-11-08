#!/usr/bin/env python3
"""H007: does a smaller batch rescue the trainers that collapsed at 1,024 envs?

This is the direct test of the prediction registered against F006.  Five
trainers that score 18-26 in the 1M protocol at 64 environments fall to ~2 at
1,024 environments in the highenv sweep -- Rainbow, IQN, FQF, DER, M-IQN.  F006
attributes that to the ladder raising batch size from the published 32 to 512,
which at a fixed replay ratio divides the optimizer-step count by 16.

If that is the cause, running them at 1,024 environments with the PUBLISHED
batch size should recover them, with no other change.  The comparison is held
at the published replay ratio of 8 and the 1M transition budget throughout, so
the only thing that moves is batch size -- and therefore optimizer steps.

    batch  32 -> 250,000 optimizer steps  (what they get in the 1M protocol)
    batch 512 ->  15,625 optimizer steps  (what the ladder gave them)

Prediction: the batch-32 column beats the batch-512 column for every one of the
five, and at least one clears the random-play band of 5 that the highenv report
records them under.  A flat result refutes F006 for these trainers and points
at the environment count itself rather than at the batch size.

Scored on transitions, so it is valid while the GPU is shared.
"""

from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from exp_runner import run_experiment

ENVS = 1024
BUDGET = 1_000_000
RATIO = 8.0
BATCHES = (32, 128, 512)
# The five that fell from a real score to the random-play band, plus DER which
# fell from an already-poor 5.3 and is the control for "was never good anyway".
ALGOS = ("rainbow", "iqn", "fqf", "miqn", "der")
BUFFER = 131_072


def specs():
    out = []
    for algo in ALGOS:
        for batch in BATCHES:
            out.append({
                "name": f"{algo}-b{batch}",
                "algorithm": algo,
                "variant": "torchcompile",
                "num_envs": ENVS,
                "total_timesteps": BUDGET,
                "seed": 1,
                "batch_size": batch,
                "replay_ratio": RATIO,
                "planned_updates": int(BUDGET * RATIO / batch),
                "flags": ["--batch-size", batch, "--replay-ratio", RATIO,
                          "--buffer-size", BUFFER],
            })
    # Cheapest cell of every algorithm first, so a partial run still has one
    # complete column to compare rather than one complete row.
    return sorted(out, key=lambda s: (-s["batch_size"], s["algorithm"]))


if __name__ == "__main__":
    results = run_experiment("h007_rescue", specs(), timeout=7200.0,
                             dry_run="--dry-run" in sys.argv)
    print()
    by = {}
    for r in results:
        if r.get("score") is not None:
            by.setdefault(r["algorithm"], {})[r["batch_size"]] = r["score"]
    print(f"{'algorithm':12s}" + "".join(f"{('b%d'%b):>10s}" for b in BATCHES) + f"{'best/worst':>12s}")
    for algo in ALGOS:
        row = by.get(algo, {})
        cells = "".join(f"{row[b]:10.2f}" if b in row else f"{'-':>10s}" for b in BATCHES)
        vals = [v for v in row.values() if v]
        ratio = f"{max(vals)/max(min(vals),1e-9):11.1f}x" if len(vals) > 1 else " " * 12
        print(f"{algo:12s}{cells}{ratio}")
