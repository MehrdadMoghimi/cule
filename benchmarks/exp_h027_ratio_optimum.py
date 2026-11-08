#!/usr/bin/env python3
"""H027: where does the best constant replay ratio sit, and does it move with budget?

Journal hypothesis: registered at run time.

F029 and F036 measure the marginal value of replay by doubling it from 8 to 16:
+0.47 in log2 at a 250,000-transition budget and +0.28 at 500,000.  Both are
positive, so at neither budget is 16 past the peak, and the two numbers taken
together say something mildly counter-intuitive -- the marginal value of MORE
replay is HIGHER at the smaller budget, not lower.  That is the opposite of the
reading the crossover invites, which is that a small budget is data-limited and
should therefore want less replay.

Both readings cannot be right, and the reason they conflict is that the two
slopes were not measured at the same number of optimizer steps: doubling the
ratio at 250k transitions moves 62,500 -> 125,000 updates, and at 500k it moves
125,000 -> 250,000.  A slope evaluated at a different place on a concave curve
is not evidence about where that curve peaks.

The way to settle it is to stop taking derivatives and just find the maximum.
Five constant ratios spanning a 16x range, at two budgets, on three games:

    ratios   1, 2, 4, 8, 16          (32 would exceed the submission guard at
                                      this env count and is left out)
    budgets  250,000 and 1,000,000 transitions
    games    Breakout, Qbert, Assault

REGISTERED PREDICTION: the argmax ratio is the SAME or HIGHER at 1,000,000 than
at 250,000 in a majority of the three games.  This is a prediction against the
intuitive reading of the crossover and in favour of what the measured slopes
actually say; if the argmax comes out lower at the larger budget, the slopes in
F029 and F036 are being over-interpreted and the crossover should be described
only as a statement about marginal values at a fixed operating point, never as a
recommendation about how much to replay.
"""

from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from exp_runner import run_experiment
from exp_h021_currency_by_game import DQN

GAMES = ("Breakout", "Qbert", "Assault")
RATIOS = (1.0, 2.0, 4.0, 8.0, 16.0)
BUDGETS = (250_000, 1_000_000)


def specs(seeds=(1,)):
    out = []
    for seed in seeds:
        tag = "" if seed == 1 else f"-s{seed}"
        for budget in BUDGETS:
            for game in GAMES:
                for ratio in RATIOS:
                    out.append({
                        "name": f"dqn-{game.lower()}-b{budget // 1000}k-r{ratio:g}{tag}",
                        "algorithm": "dqn", "variant": "torchcompile",
                        "env_id": f"{game}NoFrameskip-v4", "num_envs": DQN["envs"],
                        "total_timesteps": budget, "seed": seed,
                        "game": game, "arm": f"r{ratio:g}", "budget": budget,
                        "ratio": ratio, "family": "off_policy",
                        "planned_updates": int(budget * ratio / DQN["batch"]),
                        "flags": ["--batch-size", DQN["batch"], "--replay-ratio", ratio,
                                  "--buffer-size", DQN["buffer"]],
                    })
    return out


def _opt(name, default=None):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


if __name__ == "__main__":
    run_experiment("h027_ratio_optimum",
                   specs(tuple(int(x) for x in _opt("--seeds", "1").split(","))),
                   timeout=3600.0, startup_cap=300.0, stats_interval=16,
                   closing=0.25, dry_run="--dry-run" in sys.argv)
