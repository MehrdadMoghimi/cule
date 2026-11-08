#!/usr/bin/env python3
"""H013: find the environment count that Hadamax-PQN actually wants.

Hadamax-PQN is the most sample-efficient trainer in this repository -- rank 1 of
35 at a million transitions, score 371 -- and rank 13 under the wall-clock
protocol at score 26.8.  The gap is not throughput alone: at 1,024 environments
it collected 1.62M transitions, MORE than the 1.00M that earned it 371, and
scored 13.9x worse.

F015 says an on-policy trainer is paid in fresh transitions (+1.82x per
doubling) rather than in gradient steps, and F016 says each of Hadamax's
transitions costs 16.9x the arithmetic of PQN's.  Both point the same way: its
optimum should sit at a far lower environment count than the ladder's fixed
1,024, which is the one value the ladder never varied.

Design mirrors H008.  Rollout length, minibatch SIZE and epoch count are pinned,
so every cell performs the same 7,812 optimizer steps on the same 1,000,000
transitions and only the environment count moves.  `--num-minibatches` is
recomputed per cell to hold the minibatch at 512 -- without that it would scale
with the rollout and the comparison would move two things at once.

Scored on transitions, so it is valid on a shared GPU.  Throughput is recorded
alongside, which is what turns the resulting curve into a wall-clock prediction:
the best five-minute operating point is the count maximising score at
sps x 300 transitions, not the count maximising sps.
"""

from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from exp_runner import run_experiment

BUDGET = 1_000_000
NUM_STEPS = 32
MINIBATCH = 512
EPOCHS = 4
ENV_COUNTS = (32, 64, 128, 256, 512, 1024)
ALGOS = ("hadamax", "pqn")     # PQN is the control: the trainer that likes 1024


def specs():
    out = []
    for algo in ALGOS:
        for envs in ENV_COUNTS:
            rollout = envs * NUM_STEPS
            num_minibatches = max(1, round(rollout / MINIBATCH))
            out.append({
                "name": f"{algo}-n{envs}",
                "algorithm": algo, "variant": "torchcompile",
                "num_envs": envs, "total_timesteps": BUDGET, "seed": 1,
                "num_steps": NUM_STEPS, "minibatch": MINIBATCH,
                "update_epochs": EPOCHS,
                "planned_updates": int(BUDGET * EPOCHS / MINIBATCH),
                "flags": ["--num-steps", NUM_STEPS,
                          "--num-minibatches", num_minibatches,
                          "--update-epochs", EPOCHS],
            })
    return sorted(out, key=lambda s: (s["algorithm"], -s["num_envs"]))


if __name__ == "__main__":
    results = run_experiment("h013_hadamax_envs", specs(), timeout=9000.0,
                             dry_run="--dry-run" in sys.argv)
    print()
    by = {}
    for r in results:
        if r.get("score") is not None:
            by.setdefault(r["algorithm"], {})[r["num_envs"]] = r
    for algo in ALGOS:
        row = by.get(algo, {})
        if not row:
            continue
        print(f"\n{algo}")
        print(f"  {'envs':>6s} {'score@1M':>9s} {'sps':>8s} {'5-min transitions':>18s}")
        for n in ENV_COUNTS:
            if n not in row:
                continue
            r = row[n]
            sps = r["train_sps"] or 0
            print(f"  {n:6d} {r['score']:9.2f} {sps:8d} {sps*300/1e6:17.2f}M")
