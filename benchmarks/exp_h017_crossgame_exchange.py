#!/usr/bin/env python3
"""H017: does the gradient-steps-versus-data exchange rate hold on other games?

F012 is the central quantitative result: with sampled data pinned, score rises
+2.30 in log2 per doubling of optimizer steps; with optimizer steps pinned, it
rises +0.70 per doubling of sampled data.  One doubling of gradient steps is
worth about 3.3 doublings of data.

Both arms were measured on Breakout and only Breakout.  Breakout is dense-reward
with a scoring range that rewards one narrow skill, and there is no reason in
advance why the exchange rate should be a constant of reinforcement learning
rather than a fact about that game.

This reruns H005's arm -- the steep one, the one the conclusion rests on -- on
two more games.  Total sampled data is constant at 8.0M in every cell (1M
transitions at replay ratio 8), and batch size moves 32 -> 2048 so optimizer
steps move 64x.  If the slope comes back near +2.30 on all three, the exchange
rate is a property of the method.  If it is flat on Qbert, F012 is about
Breakout.

Pong is included as the sign test: its score range is [-21, 21], so a slope
there is measured against a floor of -21 rather than 0, and the analysis reports
the shift from the random-play floor rather than a ratio.
"""

from __future__ import annotations

import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from exp_runner import run_experiment

BUDGET = 1_000_000
ENVS = 1024
RATIO = 8.0
BATCHES = (32, 128, 512, 2048)
GAMES = {"pong": "PongNoFrameskip-v4",
         "qbert": "QbertNoFrameskip-v4",
         "spaceinvaders": "SpaceInvadersNoFrameskip-v4"}
FLOOR = {"pong": -21.0, "qbert": 0.0, "spaceinvaders": 0.0}


def specs():
    out = []
    for short, game in GAMES.items():
        for batch in BATCHES:
            out.append({
                "name": f"dqn-{short}-b{batch}",
                "algorithm": "dqn", "variant": "torchcompile", "env_id": game,
                "num_envs": ENVS, "total_timesteps": BUDGET, "seed": 1,
                "game": short, "batch_size": batch, "replay_ratio": RATIO,
                "planned_updates": int(BUDGET * RATIO / batch),
                "flags": ["--batch-size", batch, "--replay-ratio", RATIO,
                          "--buffer-size", 131_072],
            })
    # cheapest cells first (big batch = few updates), one game at a time
    return sorted(out, key=lambda s: (s["game"], -s["batch_size"]))


if __name__ == "__main__":
    results = run_experiment("h017_crossgame_exchange", specs(), timeout=9000.0,
                             dry_run="--dry-run" in sys.argv)
    print()
    cells: dict = {}
    for r in results:
        if r.get("score") is not None:
            cells.setdefault(r["game"], {})[r["planned_updates"]] = r["score"]
    print(f"{'game':15s}" + "".join(f"{('b%d'%b):>10s}" for b in BATCHES)
          + f"{'slope':>9s} {'R^2':>7s}")
    for short in GAMES:
        c = cells.get(short, {})
        if len(c) < 3:
            continue
        us = sorted(c)
        xs = [math.log2(u) for u in us]
        # measured above the game's random-play floor, so Pong is comparable
        ys = [c[u] - FLOOR[short] for u in us]
        mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
        den = sum((x - mx) ** 2 for x in xs) or 1e-9
        slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den
        pred = [my + slope * (x - mx) for x in xs]
        ss_res = sum((y - p) ** 2 for y, p in zip(ys, pred))
        ss_tot = sum((y - my) ** 2 for y in ys) or 1e-9
        row = "".join(f"{c[int(BUDGET*RATIO/b)]:10.2f}"
                      if int(BUDGET * RATIO / b) in c else f"{'-':>10s}" for b in BATCHES)
        print(f"{short:15s}{row}{slope:9.2f} {1-ss_res/ss_tot:7.3f}")
    print("\nslope = score gained per doubling of optimizer steps, above the random-play floor,")
    print("with total sampled data pinned at 8.0M in every cell.")
    print("Breakout reference (H005): +2.30 per doubling, R^2 0.977.")
