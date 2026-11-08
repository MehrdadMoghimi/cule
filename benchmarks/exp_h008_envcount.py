#!/usr/bin/env python3
"""H008: with batch and optimizer steps both pinned, does environment count still hurt?

H005 moves optimizer steps (confounded with batch), H006 moves batch with steps
pinned.  Neither touches the third variable that separates this repo's two
Breakout protocols: the 1M benchmark runs at 64-256 environments and the highenv
sweep at 1,024.

Here everything else is nailed down -- the published batch of 32, the published
replay ratio of 8, a 1M transition budget -- so every configuration performs
exactly 250,000 optimizer steps on exactly 1,000,000 transitions, and the only
thing that changes is how many environments those transitions came from.

What remains different at 1,024 environments is the *staleness* of the data: the
trainer takes 256 gradient steps between one vector step and the next, so the
policy that generated the batch is up to 256 updates behind the one being
trained, against 16 at 64 environments.  If score falls with environment count
here, parallelism itself carries a cost that no amount of batch or step tuning
recovers, and F001's protocol inversion has a third contributing cause.  If
score is flat, environment count is free once the step count is right -- which
would be the useful engineering result, since environments are what CuLE makes
cheap.

Two trainers: DQN (1-step, the workhorse) and Rainbow (3-step, distributional,
noisy nets -- the one that collapsed hardest, 25.8 -> 2.0).
"""

from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from exp_runner import run_experiment

BUDGET = 1_000_000
BATCH = 32          # the published value, held fixed
RATIO = 8.0         # the published value, held fixed
ENV_COUNTS = (64, 256, 1024)
ALGOS = ("dqn", "rainbow")
BUFFER = 131_072


def specs():
    out = []
    for algo in ALGOS:
        for envs in ENV_COUNTS:
            out.append({
                "name": f"{algo}-n{envs}",
                "algorithm": algo,
                "variant": "torchcompile",
                "num_envs": envs,
                "total_timesteps": BUDGET,
                "seed": 1,
                "batch_size": BATCH,
                "replay_ratio": RATIO,
                "planned_updates": int(BUDGET * RATIO / BATCH),
                "updates_between_vector_steps": int(RATIO * envs / BATCH),
                "flags": ["--batch-size", BATCH, "--replay-ratio", RATIO,
                          "--buffer-size", BUFFER],
            })
    return sorted(out, key=lambda s: (s["algorithm"], -s["num_envs"]))


if __name__ == "__main__":
    results = run_experiment("h008_envcount", specs(), timeout=7200.0,
                             dry_run="--dry-run" in sys.argv)
    print()
    by = {}
    for r in results:
        if r.get("score") is not None:
            by.setdefault(r["algorithm"], {})[r["num_envs"]] = r
    print(f"{'algorithm':10s}" + "".join(f"{('n%d'%n):>12s}" for n in ENV_COUNTS)
          + f"{'staleness':>28s}")
    for algo in ALGOS:
        row = by.get(algo, {})
        cells = "".join(f"{row[n]['score']:12.2f}" if n in row else f"{'-':>12s}"
                        for n in ENV_COUNTS)
        stale = " ".join(str(int(RATIO*n/BATCH)) for n in ENV_COUNTS)
        print(f"{algo:10s}{cells}    updates between steps: {stale}")
