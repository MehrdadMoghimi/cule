#!/usr/bin/env python3
"""H019: is F010's environment penalty really a replay-buffer-horizon penalty?

H008 held the replay buffer at 131,072 transitions across every environment
count and attributed the resulting 4.6x drop to staleness from parallelism.  But
a buffer of fixed size holds a DIFFERENT amount of history depending on how fast
it is filled:

    64 envs -> 131,072 / 64   = 2,048 vector steps of behaviour-policy history
   256 envs ->                =   512
  1024 envs ->                =   128

So the cell that scored best also saw sixteen times deeper history, and the two
explanations -- "the behaviour policy is 256 updates stale" and "the buffer only
remembers the last 128 vector steps" -- are not separated by that design.

Two runs separate them.  Shrink the buffer at the low environment counts so that
history measured in vector steps is equal to the 1,024-environment cell's:

    64 envs, buffer  8,192  -> 128 vector steps, same as 1024 envs at 131,072
   256 envs, buffer 32,768  -> 128 vector steps

Everything else is H008's: DQN, batch 32, replay ratio 8, 1M transitions,
250,000 optimizer steps.

    scores collapse toward 14.69  -> buffer horizon is the mechanism and F010
                                     names the wrong cause
    scores stay near 68 and 60    -> the horizon is not what mattered; the
                                     staleness of the behaviour policy is, and
                                     F010 stands as written
    something in between          -> both contribute, and the split is readable
                                     off the two numbers

Cheap and decisive: two runs against three already in hand.
"""

from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from exp_runner import run_experiment

BUDGET = 1_000_000
BATCH, RATIO = 32, 8.0
HORIZON = 128          # vector steps of history, matching the 1024-env cell
CELLS = ((64, 64 * HORIZON), (256, 256 * HORIZON))
H008 = {64: 68.24, 256: 60.05, 1024: 14.69}


def specs():
    return [{
        "name": f"dqn-n{envs}-buf{buf}",
        "algorithm": "dqn", "variant": "torchcompile",
        "num_envs": envs, "total_timesteps": BUDGET, "seed": 1,
        "batch_size": BATCH, "replay_ratio": RATIO, "buffer_size": buf,
        "history_vector_steps": buf // envs,
        "planned_updates": int(BUDGET * RATIO / BATCH),
        "flags": ["--batch-size", BATCH, "--replay-ratio", RATIO, "--buffer-size", buf],
    } for envs, buf in CELLS]


if __name__ == "__main__":
    results = run_experiment("h019_buffer_horizon", specs(), timeout=9000.0,
                             dry_run="--dry-run" in sys.argv)
    print()
    print(f"{'envs':>6s} {'buffer':>9s} {'history':>9s} {'score':>8s} "
          f"{'H008 (buf 131072)':>19s} {'attributable to horizon':>25s}")
    for r in results:
        if r.get("score") is None:
            continue
        n = r["num_envs"]
        deep = H008[n]
        shallow_ref = H008[1024]
        # of the gap between this count and 1024 in H008, how much does shrinking
        # the buffer to the same history reproduce?
        share = (deep - r["score"]) / max(deep - shallow_ref, 1e-9)
        print(f"{n:6d} {r['buffer_size']:9d} {r['history_vector_steps']:9d} "
              f"{r['score']:8.2f} {deep:19.2f} {share:24.0%}")
    print("\n100% = the whole environment-count penalty is really buffer horizon;")
    print("  0% = the horizon was irrelevant and F010 stands as written.")
