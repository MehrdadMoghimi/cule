#!/usr/bin/env python3
"""H016: measure each trainer's parallelism sensitivity as a single exponent.

H008 measured, for DQN, how much score is lost per doubling of the environment
count when batch size, replay ratio, transition budget and optimizer-step count
are all held fixed: 0.55 in log2 per doubling.  H013 measured the same quantity
for Hadamax-PQN and got 1.55 -- 2.8x as sensitive.  That difference is larger
than most of the differences the repository's benchmarks report BETWEEN
algorithms, and it is not a number anyone publishes.

If it varies systematically it is a property worth naming: how much parallelism
a trainer tolerates before the staleness of its own data costs more than the
throughput is worth.  It also has a direct use -- combined with F019's slopes it
predicts each trainer's wall-clock-optimal environment count without searching
for it.

Three counts per trainer so the exponent is a fitted slope rather than a
two-point difference, and the fit residual says whether a single exponent is
even the right description.  Everything else is pinned:

    off-policy  batch 128, replay ratio 8  -> 62,500 optimizer steps in every cell
    on-policy   num_steps 32, minibatch 512, 4 epochs -> 7,812 in every cell

Batch 128 rather than the published 32 is a deliberate compromise: at 32 the
replay trainers need 250,000 updates per cell and the experiment would take most
of a day.  The exponent is measured at a fixed batch, so it remains a valid
comparison across trainers even if its absolute level shifts with that choice.
"""

from __future__ import annotations

import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from breakout_algorithms import BY_KEY, ON_POLICY
from exp_runner import run_experiment

BUDGET = 1_000_000
ENV_COUNTS = (64, 256, 1024)
BATCH, RATIO = 128, 8.0
NUM_STEPS, MINIBATCH, EPOCHS = 32, 512, 4
ALGOS = ("pqn", "sem_pqn", "hadamax", "dqn", "ibdqn", "c51", "rainbow", "iqn")


def specs():
    out = []
    for key in ALGOS:
        algo = BY_KEY[key]
        for envs in ENV_COUNTS:
            if algo.family == ON_POLICY:
                flags = ["--num-steps", NUM_STEPS,
                         "--num-minibatches", max(1, round(envs * NUM_STEPS / MINIBATCH)),
                         "--update-epochs", EPOCHS]
                updates = int(BUDGET * EPOCHS / MINIBATCH)
                hint = NUM_STEPS
            else:
                flags = ["--batch-size", BATCH, "--replay-ratio", RATIO,
                         "--buffer-size", 131_072]
                updates = int(BUDGET * RATIO / BATCH)
                hint = None
            out.append({
                "name": f"{key}-n{envs}",
                "algorithm": key, "variant": "torchcompile", "num_envs": envs,
                "total_timesteps": BUDGET, "seed": 1, "num_steps": hint,
                "planned_updates": updates, "family": algo.family, "flags": flags,
            })
    # one complete trainer at a time, cheapest count last, so a partial run yields
    # whole rows rather than fragments of every row
    return sorted(out, key=lambda s: (ALGOS.index(s["algorithm"]), -s["num_envs"]))


if __name__ == "__main__":
    results = run_experiment("h016_staleness_exponent", specs(), timeout=9000.0,
                             dry_run="--dry-run" in sys.argv)
    print()
    cells: dict = {}
    for r in results:
        if r.get("score") is not None:
            cells.setdefault(r["algorithm"], {})[r["num_envs"]] = max(r["score"], 0.5)
    rows = []
    for key in ALGOS:
        c = cells.get(key, {})
        if len(c) < 2:
            continue
        ns = sorted(c)
        # slope of log2(score) against -log2(envs): positive = hurt by parallelism
        xs = [-math.log2(n) for n in ns]
        ys = [math.log2(c[n]) for n in ns]
        mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
        num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        den = sum((x - mx) ** 2 for x in xs) or 1e-9
        slope = num / den
        pred = [my + slope * (x - mx) for x in xs]
        resid = max(abs(y - p) for y, p in zip(ys, pred))
        rows.append((slope, key, c, resid))
    print(f"{'trainer':10s} {'family':11s}" + "".join(f"{('n%d'%n):>10s}" for n in ENV_COUNTS)
          + f"{'exponent':>10s} {'max resid':>10s}")
    for slope, key, c, resid in sorted(rows, reverse=True):
        cellstr = "".join(f"{c[n]:10.2f}" if n in c else f"{'-':>10s}" for n in ENV_COUNTS)
        print(f"{key:10s} {BY_KEY[key].family:11s}{cellstr}{slope:10.2f} {resid:10.2f}")
    print("\nexponent = log2 score lost per doubling of environment count, "
          "at fixed data and fixed optimizer steps.")
    print("reference: DQN 0.55 at batch 32 (H008), Hadamax 1.55 (H013).")
