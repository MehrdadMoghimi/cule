#!/usr/bin/env python3
"""H024: if fresh data is worth more early and replay more later, a rising

Journal hypothesis: H022 (the journal id and the file number differ: ids are assigned in registration order)
replay ratio should beat a constant one at the same total cost.

F029 measured, for DQN across six games at 250,000 transitions, that doubling
fresh data is worth +1.09 in log2 against +0.47 for doubling replay.  F030
repeated it at 500,000 and found the first number collapses to +0.20 while the
second holds at +0.28 -- the value of a fresh transition decays with budget and
the value of a re-swept one does not, so the two cross somewhere in between.

Every trainer in this repository, and as far as I know every published Atari
agent, holds the replay ratio CONSTANT for the whole run.  If the crossover is
real that is the wrong shape: it over-replays when data is the scarce resource
and under-replays when compute is.  The prescription that follows is concrete --
start low and ramp up -- and it is falsifiable, which is the point of running it.

The comparison is controlled by construction.  A linear ramp from r0 to r1
performs a total number of optimizer steps proportional to its mean (r0+r1)/2,
so a 2->14 ramp and a constant 8 do the SAME total work on the SAME total data
over the same 1,000,000 transitions; they differ only in when the work is done.
Nothing about the budget, the buffer, the batch or the network changes.

    const8      ratio 8 throughout                     (the usual practice)
    ramp_up     2 -> 14, mean 8                        (what F030 predicts)
    ramp_down   14 -> 2, mean 8                        (the control)

ramp_down is what makes this a two-sided test.  Without it, a win for ramp_up
would be consistent with "any non-constant schedule helps", which is a much
weaker and less interesting claim.  If the crossover story is right the order
must be ramp_up > const8 > ramp_down; if schedules merely add useful noise,
ramp_down should also beat const8.

REGISTERED PREDICTION: ramp_up beats const8 in a majority of the six games, and
ramp_down is the worst of the three in a majority.  The total optimizer-step
counts are recorded per run and should agree across arms to within the
discretisation of the update accumulator, which is the check that the arms
really did cost the same.
"""

from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from exp_runner import run_experiment
from exp_h021_currency_by_game import GAMES, DQN

BUDGET = 1_000_000
MEAN_RATIO = 8.0
ARMS = {
    "const8":    (MEAN_RATIO, None),
    "ramp_up":   (2.0, 14.0),
    "ramp_down": (14.0, 2.0),
}

# Seed-1 outcome, for the record: the prediction above was exactly backwards.
# ramp_down won 4 of 6 games and ramp_up 0 of 6, mean 1.14x and 0.83x of const8.
# See F031 -- the slopes measure the marginal value of the LAST doubling under a
# flat schedule, which cannot see that an early gradient step improves the
# behaviour policy and so improves every transition collected afterwards.


def specs(seeds=(1,)):
    out = []
    for seed in seeds:
        tag = "" if seed == 1 else f"-s{seed}"
        for game in GAMES:
            for arm, (r0, r1) in ARMS.items():
                flags = ["--batch-size", DQN["batch"], "--replay-ratio", r0,
                         "--buffer-size", DQN["buffer"]]
                if r1 is not None:
                    flags += ["--replay-ratio-final", r1]
                out.append({
                    "name": f"dqn-{game.lower()}-{arm}{tag}",
                    "algorithm": "dqn", "variant": "torchcompile",
                    "env_id": f"{game}NoFrameskip-v4", "num_envs": DQN["envs"],
                    "total_timesteps": BUDGET, "seed": seed,
                    "game": game, "arm": arm, "family": "off_policy",
                    "ratio_start": r0, "ratio_end": r1 if r1 is not None else r0,
                    "planned_updates": int(BUDGET * MEAN_RATIO / DQN["batch"]),
                    "flags": flags,
                })
    return out


def _opt(name, default=None):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


if __name__ == "__main__":
    run_experiment("h024_ratio_schedule",
                   specs(tuple(int(x) for x in _opt("--seeds", "1").split(","))),
                   timeout=2400.0, startup_cap=300.0, stats_interval=16,
                   closing=0.25, dry_run="--dry-run" in sys.argv)
