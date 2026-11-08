#!/usr/bin/env python3
"""H026: does the replay-schedule result hold for off-policy trainers other than DQN?

Journal hypothesis: H025.

F034 established, at three seeds and identical total optimizer steps, that moving
replay LATER in training is worse in six games out of six for DQN (mean 0.85x of
a constant ratio, sign test p = 0.031), while moving it earlier is not reliably
better.  Every number in it is DQN.  The proposed mechanism -- an early gradient
step improves the behaviour policy and therefore the quality of every transition
collected afterwards, so its value compounds in a way a late step's does not --
is a property of off-policy learning in general and says nothing specific about
a one-step TD target.  If it is right the effect should appear in any trainer
that fills its own replay buffer.  If it is really about DQN's particular
bootstrap, it should not.

Three more off-policy trainers, all at DQN's operating point (128 envs, batch 32,
buffer 100k, mean replay ratio 8, 1,000,000 transitions) so the arms are directly
comparable with F034:

    C51      distributional, categorical
    QR-DQN   distributional, quantile
    IB-DQN   information-bottleneck regularised

Three games rather than six, chosen to span DQN's own range of outcomes rather
than to sample Atari: Breakout, where the falling ramp helped most (+0.36);
Assault, where it helped moderately (+0.17); Qbert, where DQN preferred the
constant ratio outright (-0.13).

REGISTERED PREDICTION: the rising ramp is worse than the constant in a clear
majority of the nine trainer-game cells, reproducing F034's negative half.  On
the positive half I predict nothing, because F034 could not establish it for DQN
either.

POWER, worked out after the first trainer came back flat and before reading
anything into it.  DQN's per-game standard errors at three seeds average 0.15,
so a single seed carries about 0.27 in log2.  Three games at one seed give the
per-trainer mean a standard error of 0.153, against a DQN-sized effect of -0.232
-- 1.5 sigma, which cannot separate "no effect" from "the same effect as DQN".
Three games at three seeds give 0.089, or 2.6 sigma, which can.  So one seed per
trainer is a screen and not a test, and a flat result from it must not be read
as a refutation.  Seeds 2 and 3 are the experiment.
"""

from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from exp_runner import run_experiment

BUDGET = 1_000_000
ENVS, BATCH, BUFFER = 128, 32, 100_000
GAMES = ("Breakout", "Assault", "Qbert")
ALGOS = ("c51", "qrdqn", "ibdqn")
ARMS = {"const8": (8.0, None), "ramp_up": (2.0, 14.0), "ramp_down": (14.0, 2.0)}


def specs(seeds=(1,)):
    out = []
    for seed in seeds:
        tag = "" if seed == 1 else f"-s{seed}"
        for algo in ALGOS:
            for game in GAMES:
                for arm, (r0, r1) in ARMS.items():
                    flags = ["--batch-size", BATCH, "--replay-ratio", r0,
                             "--buffer-size", BUFFER]
                    if r1 is not None:
                        flags += ["--replay-ratio-final", r1]
                    out.append({
                        "name": f"{algo}-{game.lower()}-{arm}{tag}",
                        "algorithm": algo, "variant": "torchcompile",
                        "env_id": f"{game}NoFrameskip-v4", "num_envs": ENVS,
                        "total_timesteps": BUDGET, "seed": seed,
                        "game": game, "arm": arm, "family": "off_policy",
                        "ratio_start": r0, "ratio_end": r1 if r1 is not None else r0,
                        "flags": flags,
                    })
    return out


def _opt(name, default=None):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


if __name__ == "__main__":
    run_experiment("h026_schedule_generality",
                   specs(tuple(int(x) for x in _opt("--seeds", "1").split(","))),
                   timeout=3600.0, startup_cap=300.0, stats_interval=16,
                   closing=0.25, dry_run="--dry-run" in sys.argv)
