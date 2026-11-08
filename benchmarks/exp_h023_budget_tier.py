#!/usr/bin/env python3
"""H023: does the value of fresh data relative to more replay depend on the budget?

Journal hypothesis: H023

H021 measured, for DQN across six games at 250,000 transitions and 62,500
optimizer steps, mean slopes of +0.88 in log2 per doubling of FRESH TRANSITIONS
at fixed compute, against +0.44 per doubling of REPLAY at fixed fresh data.  A
new transition is worth about twice a re-swept one, and all six per-game
exchange rates R came out at or below 1.28.

A word on coordinates, because this is easy to get wrong and I did.  F012
reports the opposite ordering -- +2.30 per doubling of optimizer steps against
+0.70 per doubling of samples consumed, R = 3.3 -- and it is tempting to read
H021 as failing to reproduce it.  It is not.  F012 traverses its step axis by
SHRINKING THE BATCH at fixed samples consumed, and holds fresh transitions
pinned at 1M in both of its arms; its two axes are (optimizer steps, samples
consumed) at constant data.  H021's arms move a pair F012 never varied:

    +data    fresh transitions x2, optimizer steps FIXED, samples FIXED
             (budget doubled and replay ratio halved, which cancel exactly)
    +steps   fresh transitions FIXED, replay ratio x2, so optimizer steps and
             samples consumed both x2 -- with batch pinned these two are
             perfectly collinear and no experiment of this shape can separate
             them, which is fine, because turning up the replay ratio is what a
             practitioner actually does.

So H021 answers "is it better to collect another transition or to re-sweep the
ones I have", and F012 answers "given a fixed pile of sampled data, is it better
to take many small steps or few large ones".  Both are worth knowing and neither
bears on the other.

The question here is whether H021's answer moves with the budget.  The mechanism
would be unsurprising: early in a run the buffer is small and the policy poor, so
a fresh transition carries something no amount of re-sweeping can synthesise;
later, data is plentiful and what limits the agent is how much it has extracted
from what it already holds.  If that is right, the advantage of fresh data
should shrink as the base budget grows, and where it crosses over is a concrete
number that says which resource to buy at which point in a run.

This repeats H021's three-cell design at a base budget of 500,000 -- exactly
double on both axes, so the tier is a clean doubling of the operating point
rather than a new experiment.  Same six games, same trainer, env count, batch
size and buffer, so the only thing that moves is where on the surface the
derivative is taken.

REGISTERED PREDICTION: slope_data falls and slope_steps rises relative to H021,
so mean R rises above H021's 0.29 (median over the six games, whose mean is
dragged by MsPacman's noise-dominated cell).  If instead the slopes are
unchanged, the exchange rate is a property of the operating point only through
the game and trainer, and the crossover story is wrong.
"""

from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from exp_runner import run_experiment
from exp_h021_currency_by_game import GAMES, DQN, ARMS

BASE_BUDGET = 500_000


def specs(seeds=(1,)):
    out = []
    for seed in seeds:
        tag = "" if seed == 1 else f"-s{seed}"
        for game in GAMES:
            for arm, data_mult, step_mult in ARMS:
                budget = BASE_BUDGET * data_mult
                ratio = DQN["ratio"] * step_mult / data_mult
                out.append({
                    "name": f"dqn-{game.lower()}-{arm}{tag}",
                    "algorithm": "dqn", "variant": "torchcompile",
                    "env_id": f"{game}NoFrameskip-v4", "num_envs": DQN["envs"],
                    "total_timesteps": budget, "seed": seed,
                    "game": game, "arm": arm, "family": "off_policy",
                    "base_budget": BASE_BUDGET,
                    "data_mult": data_mult, "step_mult": step_mult,
                    "planned_updates": int(budget * ratio / DQN["batch"]),
                    "flags": ["--batch-size", DQN["batch"], "--replay-ratio", ratio,
                              "--buffer-size", DQN["buffer"]],
                })
    return out


def _opt(name, default=None):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


if __name__ == "__main__":
    run_experiment("h023_budget_tier",
                   specs(tuple(int(x) for x in _opt("--seeds", "1").split(","))),
                   timeout=2400.0, startup_cap=300.0, stats_interval=16,
                   closing=0.25, dry_run="--dry-run" in sys.argv)
