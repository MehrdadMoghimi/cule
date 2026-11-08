#!/usr/bin/env python3
"""H015: does any of this hold on a game other than Breakout?

Every finding so far -- the protocol inversion (F001), the exchange rate between
gradient steps and data (F012), the environment-count penalty (F010), the
family-currency split (F015) -- was measured on BreakoutNoFrameskip-v4 and
nothing else.  Breakout is a single, atypical game: dense reward, short episodes
early, and a scoring range that rewards a policy which learns one narrow skill.
A mechanism that only holds there is a fact about Breakout, not about
reinforcement learning at 1,024 environments.

The cheapest test that could falsify the central claim is the environment-count
contrast, run unchanged on three more games.  Batch, replay ratio, transition
budget and optimizer-step count are all pinned exactly as in H008; only the game
and the environment count move.

    DQN  (off-policy)  expected to LOSE heavily at 1,024 -- it lost 4.6x on Breakout
    PQN  (on-policy)   expected to lose far less, or gain, per F015

Pong is included deliberately even though its score range is [-21, 21] and no
ratio can be formed on it: it is the game where "learning" means climbing from
-21, so a mechanism that reverses there would be visible in the sign alone.
Scores are reported as differences as well as ratios for that reason.
"""

from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from exp_runner import run_experiment

BUDGET = 1_000_000
BATCH = 32
RATIO = 8.0
NUM_STEPS = 32
MINIBATCH = 512
EPOCHS = 4
ENV_COUNTS = (64, 1024)
GAMES = ("PongNoFrameskip-v4", "SpaceInvadersNoFrameskip-v4", "QbertNoFrameskip-v4")
SHORT = {g: g.replace("NoFrameskip-v4", "").lower() for g in GAMES}


def specs():
    out = []
    for game in GAMES:
        for envs in ENV_COUNTS:
            out.append({
                "name": f"dqn-{SHORT[game]}-n{envs}",
                "algorithm": "dqn", "variant": "torchcompile", "env_id": game,
                "num_envs": envs, "total_timesteps": BUDGET, "seed": 1,
                "game": SHORT[game], "batch_size": BATCH, "replay_ratio": RATIO,
                "planned_updates": int(BUDGET * RATIO / BATCH),
                "flags": ["--batch-size", BATCH, "--replay-ratio", RATIO,
                          "--buffer-size", 131_072],
            })
            out.append({
                "name": f"pqn-{SHORT[game]}-n{envs}",
                "algorithm": "pqn", "variant": "torchcompile", "env_id": game,
                "num_envs": envs, "total_timesteps": BUDGET, "seed": 1,
                "game": SHORT[game], "num_steps": NUM_STEPS,
                "planned_updates": int(BUDGET * EPOCHS / MINIBATCH),
                "flags": ["--num-steps", NUM_STEPS,
                          "--num-minibatches", max(1, round(envs * NUM_STEPS / MINIBATCH)),
                          "--update-epochs", EPOCHS],
            })
    # both counts of one game before moving on, so a partial run still contrasts
    return sorted(out, key=lambda s: (s["game"], s["algorithm"], -s["num_envs"]))


if __name__ == "__main__":
    results = run_experiment("h015_crossgame", specs(), timeout=9000.0,
                             dry_run="--dry-run" in sys.argv)
    print()
    cells = {(r["algorithm"], r["game"], r["num_envs"]): r["score"]
             for r in results if r.get("score") is not None}
    print(f"{'trainer':8s} {'game':14s} {'n=64':>9s} {'n=1024':>9s} {'diff':>9s} {'ratio':>8s}")
    for algo in ("dqn", "pqn"):
        for game in GAMES:
            g = SHORT[game]
            lo, hi = cells.get((algo, g, 64)), cells.get((algo, g, 1024))
            if lo is None or hi is None:
                continue
            ratio = f"{lo/hi:7.2f}x" if hi > 0 and lo > 0 else "      -"
            print(f"{algo:8s} {g:14s} {lo:9.2f} {hi:9.2f} {lo-hi:+9.2f} {ratio}")
    print("\nBreakout reference (H008, same protocol): dqn 68.24 -> 14.69 (4.6x), "
          "pqn not measured at batch 32")
