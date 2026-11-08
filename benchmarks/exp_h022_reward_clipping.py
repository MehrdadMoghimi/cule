#!/usr/bin/env python3
"""H022: when Atari reward clipping helps, is it buying information or scale?

Journal hypothesis: H024 (the journal id and the file number differ: ids are assigned in registration order)

Every Atari agent in this repository clips the learning reward to its sign, as
Mnih et al. 2015 did, and reports the raw game score.  The justification usually
given is that it makes one learning rate work across games with wildly different
reward magnitudes.  That is a claim about OPTIMISATION SCALE.  A different claim
is often implied in the same breath -- that clipping discards the distinction
between a 25-point and a 1,000-point event, and so costs the agent information.
Both are plausible and they are rarely separated, because in most games the two
move together: big rewards and varied rewards go hand in hand.

`game_descriptors.py` measures the two axes apart, under a random policy:

    reward_scale   mean magnitude of the nonzero rewards -- how much clipping
                   changes the SIZE of the TD target
    reward_cv      coefficient of variation of those magnitudes -- how much
                   clipping destroys, 0 meaning it is information-lossless

and there are games at both off-diagonal corners, which is what makes this
testable:

    Assault   cv 0.00, scale 21.0   clipping loses NOTHING, shrinks targets 21x
    Amidar    cv 1.45, scale  2.2   clipping loses a LOT, barely changes scale
    Breakout  cv 0.10, scale  1.0   clipping is very nearly the identity map

The two accounts therefore make opposite predictions, registered here before the
runs:

    IF the benefit is scale    -- unclipped loses in proportion to reward_scale,
                                  so it should be hurt most on Assault (21x) and
                                  Qbert (30x) and hardly at all on Amidar (2.2x).
    IF the benefit is negative
    information                -- unclipped should WIN where reward_cv is high
                                  (Amidar 1.45, Alien 1.34, Qbert 1.27) and tie
                                  where it is zero (Assault, Frostbite).
    Breakout is the null: clipping there is nearly the identity, so the two arms
    must come out equal.  If they do not, the run-to-run noise is larger than
    the effect and nothing else here can be believed.

One trainer (DQN), one seed, 250,000 transitions, at exactly H021's base
operating point so the clipped arm is directly comparable with that experiment.
Scores are raw game score either way -- `cule_env` accumulates episode returns
from the unclipped reward regardless of this flag, so the two arms are scored on
the same axis.
"""

from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from exp_runner import run_experiment

# (reward_cv, reward_scale) from game_descriptors.jsonl, for the record.
GAMES = {
    "Breakout":      (0.10,  1.0),   # null: clipping is nearly the identity
    "Amidar":        (1.45,  2.2),   # high loss, low scale   -> discriminating
    "Assault":       (0.00, 21.0),   # no loss,   high scale  -> discriminating
    "Krull":         (0.37, 12.7),
    "MsPacman":      (0.86, 10.3),
    "SpaceInvaders": (1.11, 16.0),
    "Qbert":         (1.27, 30.0),
    "Alien":         (1.34, 10.5),
    # Added after the first pass.  The eight games above gave a correlation of
    # -0.859 between the harm from dropping clipping and the RATE at which
    # reward accumulates, against -0.531 for the magnitude of a single reward
    # and +0.368 for magnitude variety -- but on eight points with no individual
    # game separated from zero, which is not enough to prefer one predictor over
    # another.  These five spread the reward-rate axis from 0.94 down to 0.15
    # and add two more games where clipping is information-lossless but changes
    # the scale a great deal (Asterix 50x, BeamRider 44x), which is the corner
    # that separates the two accounts.
    "Asterix":       (0.07, 50.3),   # no loss, the largest scale in the set
    "Gopher":        (0.82, 26.1),
    "BeamRider":     (0.01, 44.0),   # no loss, high scale
    "Frostbite":     (0.00, 10.0),
    "Seaquest":      (0.08, 20.1),
}

BUDGET = 250_000
ENVS, BATCH, RATIO, BUFFER = 128, 32, 8.0, 100_000


def specs(seeds=(1,)):
    out = []
    for seed in seeds:
        tag = "" if seed == 1 else f"-s{seed}"
        for game, (cv, scale) in GAMES.items():
            for arm, flag in (("clip", []), ("raw", ["--no-clip-rewards"])):
                out.append({
                    "name": f"dqn-{game.lower()}-{arm}{tag}",
                    "algorithm": "dqn", "variant": "torchcompile",
                    "env_id": f"{game}NoFrameskip-v4", "num_envs": ENVS,
                    "total_timesteps": BUDGET, "seed": seed,
                    "game": game, "arm": arm, "reward_cv": cv, "reward_scale": scale,
                    "flags": ["--batch-size", BATCH, "--replay-ratio", RATIO,
                              "--buffer-size", BUFFER] + flag,
                })
    return out


def _opt(name, default=None):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


if __name__ == "__main__":
    run_experiment("h022_reward_clipping",
                   specs(tuple(int(x) for x in _opt("--seeds", "1").split(","))),
                   timeout=1800.0, startup_cap=300.0, stats_interval=16,
                   closing=0.25, dry_run="--dry-run" in sys.argv)
