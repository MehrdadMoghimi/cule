#!/usr/bin/env python3
"""H021: is the gradient-steps/fresh-data exchange rate a property of the

Journal hypothesis: H021
algorithm, or of the game?

F012 measured, for DQN on Breakout, that score rises +2.30 in log2 per doubling
of optimizer steps and only +0.70 per doubling of sampled transitions -- an
exchange rate of 3.3 to 1 in favour of gradient steps.  F015 found the opposite
sign for on-policy trainers and used that to explain the protocol inversion
(F001).  Both numbers come from one game.

F026 is the reason to doubt they travel.  It refuted a model that took a single
staleness exponent measured on DQN and applied it to seven other trainers; the
exponent turned out to be trainer-specific with a 3.2x spread.  Nobody has asked
the same question of the game axis, and `game_descriptors.py` now gives a
concrete reason to expect trouble there too: under a random policy Breakout
revisits coarse states far more than anything else measured, at 61 novel states
per 1,000 steps against 299-991 for the other seventeen games.  Breakout is the
most redundant game in the set by an order of magnitude, and redundancy is
exactly the condition under which replaying old data should beat collecting new.
So the prior is that F012's 3.3:1 is close to an upper bound rather than a
typical value.

DESIGN.  Per (game, trainer), three cells sharing a base:

    base    D transitions,  S optimizer steps
    +data   2D transitions, S optimizer steps   (replay ratio / epochs halved)
    +steps  D transitions,  2S optimizer steps  (replay ratio / epochs doubled)

giving slope_data and slope_steps as log2 ratios against the shared base, and
R = slope_steps / slope_data.

Scores are taken RELATIVE TO THE RANDOM-POLICY RETURN measured by
`game_descriptors.py`.  This is not cosmetic: Krull pays a random agent 1,634
and MsPacman 236, so a log ratio of raw returns would be mostly a ratio of
baselines, compressed by a different factor in every game.  Breakout, at 1.3,
is the one game where that mistake is invisible -- another way in which
measuring only there is misleading.

TRAINERS.  DQN off-policy, Hadamax on-policy.  PQN was the obvious on-policy
choice and is the wrong one: its own 1M curve sits at 2.0 at 250k transitions
and 4.2 at 500k, against a random baseline of 1.3, so every cell would be noise.
Hadamax reaches 20.0 by 125k and 47.1 by 250k.  It is also the most
protocol-inverted trainer in the repository -- first on the 1M ranking, near the
bottom on wall clock -- which makes it the interesting on-policy arm anyway.

Each trainer runs at the environment count its 1M benchmark row used (DQN 128,
Hadamax 32) rather than at a shared value, because F010/F020 make score strongly
env-count dependent and trainer-specific.  Env count is pinned WITHIN a trainer,
which is what the slopes require; holding it across trainers would buy nothing
and would put Hadamax two doublings away from where it is known to work.  The
Breakout base cells double as a validity check: they should land near the 1M
reference curve at the same transition count.
"""

from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from exp_runner import run_experiment

# Six games spanning the novelty axis measured by `game_descriptors.py`
# (novel states per 1,000 random steps in brackets), restricted to games whose
# score moves within a few hundred thousand transitions -- which rules out Pong,
# Freeway and Enduro however interesting their descriptors are.
GAMES = {
    "Breakout":      61.4,
    "Qbert":        298.8,
    "SpaceInvaders": 584.3,
    "MsPacman":     677.1,
    "Krull":        920.2,
    "Assault":      984.3,
}

D = 250_000

DQN = dict(envs=128, batch=32, ratio=8.0, buffer=100_000)
HADAMAX = dict(envs=32, num_steps=32, minibatches=4, epochs=4)

ARMS = (("base", 1, 1), ("data", 2, 1), ("steps", 1, 2))


def specs(arms=None, seeds=(1,), algos=None):
    arms = [a for a in ARMS if arms is None or a[0] in arms]
    out = []
    for seed in seeds:
        # Seed 1 keeps the original unsuffixed names so the first pass stays cached.
        tag = "" if seed == 1 else f"-s{seed}"
        for game in GAMES:
            short = game.lower()
            for arm, data_mult, step_mult in arms:
                budget = D * data_mult
                ratio = DQN["ratio"] * step_mult / data_mult
                out.append({
                    "name": f"dqn-{short}-{arm}{tag}",
                    "algorithm": "dqn", "variant": "torchcompile",
                    "env_id": f"{game}NoFrameskip-v4", "num_envs": DQN["envs"],
                    "total_timesteps": budget, "seed": seed,
                    "game": game, "arm": arm, "family": "off_policy",
                    "data_mult": data_mult, "step_mult": step_mult,
                    "planned_updates": int(budget * ratio / DQN["batch"]),
                    "flags": ["--batch-size", DQN["batch"], "--replay-ratio", ratio,
                              "--buffer-size", DQN["buffer"]],
                })
                # On-policy: optimizer steps are budget * epochs * minibatches /
                # (envs * num_steps), so update_epochs moves steps at fixed data the
                # same way the replay ratio does.  EPOCHS is 4 rather than 1 so the
                # +data arm stays an integer after halving.
                epochs = int(HADAMAX["epochs"] * step_mult / data_mult)
                rollouts = budget / (HADAMAX["envs"] * HADAMAX["num_steps"])
                out.append({
                    "name": f"hadamax-{short}-{arm}{tag}",
                    "algorithm": "hadamax", "variant": "torchcompile",
                    "env_id": f"{game}NoFrameskip-v4", "num_envs": HADAMAX["envs"],
                    "total_timesteps": budget, "seed": seed,
                    "game": game, "arm": arm, "family": "on_policy",
                    "data_mult": data_mult, "step_mult": step_mult,
                    "num_steps": HADAMAX["num_steps"],
                    "planned_updates": int(rollouts * epochs * HADAMAX["minibatches"]),
                    "flags": ["--num-steps", HADAMAX["num_steps"],
                              "--num-minibatches", HADAMAX["minibatches"],
                              "--update-epochs", epochs],
                })
    if algos is not None:
        out = [s for s in out if s["algorithm"] in algos]
    return out


def _opt(name, default=None):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


if __name__ == "__main__":
    arms = _opt("--arms")
    algos = _opt("--algos")
    run_experiment("h021_currency_by_game",
                   specs(arms.split(",") if arms else None,
                         tuple(int(x) for x in _opt("--seeds", "1").split(",")),
                         algos.split(",") if algos else None),
                   timeout=1800.0, startup_cap=300.0, stats_interval=16,
                   closing=0.25, dry_run="--dry-run" in sys.argv)
