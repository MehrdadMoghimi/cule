#!/usr/bin/env python3
"""H025: how far does front-loading the replay ratio pay?

Journal hypothesis: registered at run time.

F031 found that a replay ratio decaying 14->2 beats a constant 8 by 1.14x on
average across six games, and beats the mirror-image 2->14 ramp by 1.38x, at
total optimizer steps equal to within 0.01%.  That establishes a direction.  It
does not say whether 14->2 is near the best available or merely the first thing
tried, and the difference matters for using it: a prescription worth stating has
to say how steep.

Every arm here holds the mean ratio at 8, so all of them cost the same as the
constant-8 default and as each other; they differ only in the steepness of the
decay.  The extreme arm ends at zero, meaning the agent stops learning entirely
for the final stretch and coasts on the policy it already has -- if that still
beats a constant ratio, the compounding argument in F031 is much stronger than
one ramp can show, and if it collapses, the useful range has an upper edge that
the intermediate arms locate.

    const8    8 throughout          the default
    ramp11    11 -> 5               mild
    ramp14    14 -> 2               F031's arm, for continuity
    ramp16    16 -> 0               the maximum possible decay at mean 8

REGISTERED PREDICTION: score rises monotonically with ramp steepness up to
ramp14 and falls at ramp16, because an agent that stops updating entirely cannot
track a replay buffer that keeps turning over.  If instead ramp16 is best, the
effect is not about scheduling compute at all but about late updates being
actively harmful -- a different and more surprising claim, and one that would
point at the buffer's recency rather than at policy compounding.
"""

from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from exp_runner import run_experiment
from exp_h021_currency_by_game import GAMES, DQN

BUDGET = 1_000_000
ARMS = {
    "const8": (8.0, None),
    "ramp11": (11.0, 5.0),
    "ramp14": (14.0, 2.0),
    "ramp16": (16.0, 0.0),
}


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
                    "ramp_span": (r0 - (r1 if r1 is not None else r0)),
                    "flags": flags,
                })
    return out


def _opt(name, default=None):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


if __name__ == "__main__":
    run_experiment("h025_ramp_strength",
                   specs(tuple(int(x) for x in _opt("--seeds", "1").split(","))),
                   timeout=2400.0, startup_cap=300.0, stats_interval=16,
                   closing=0.25, dry_run="--dry-run" in sys.argv)
