#!/usr/bin/env python3
"""H014: rerun the wall-clock ranking with the one knob the ladder never varied.

`sweep_highenv.py` fixed the environment count at the highest value each trainer
could run -- 1,024 for all of these -- and searched everything else.  F010 then
measured a 4.6x cost for DQN in moving 64 -> 1,024 with batch size and optimizer
steps both pinned, and F015 showed that on-policy trainers are paid in fresh
transitions while off-policy trainers are paid in gradient steps, so the two
families should not want the same count.  Neither could have been discovered by
that sweep, because the count was the one thing it held constant.

This rerun changes exactly one thing per trainer.  Every hyperparameter is the
ladder's own winning value, the budget is the same 300-second window, the score
is the same closing-40%-of-window mean -- so a row here is directly comparable
with a row of that report.  Only `--num-envs` moves, and the two derived
quantities that would otherwise move with it are pinned:

  on-policy   `--num-minibatches` is recomputed so the minibatch SIZE stays at
              the winner's value; otherwise it would scale with the rollout and
              the gradient step would change size along with the count.
  off-policy  the replay RATIO is passed directly rather than the ladder's
              updates-per-vector-step, which is defined against 1,024 envs and
              would silently change the samples drawn per transition.

Eight trainers: the four that top the wall-clock table (PQN, SEM-PQN, DQN,
IB-DQN), the one that tops the sample-efficiency table and sits 13th here
(Hadamax-PQN), and three that the 1M protocol rates highly and this one does not
(C51, SAC, BBF).

Two arms for the replay trainers, because holding the ratio fixed answers only
half the question.  At 64 environments CuLE steps at 4,840 transitions/sec
against 68,294 at 1,024 (F019), so a low-count run leaves most of the card idle
-- and a replay trainer can spend that idle time on gradient steps instead.

    keep-ratio  the ladder's replay ratio, unchanged.  Lowering the count then
                costs data AND optimizer steps together, and F019 predicts this
                arm is roughly flat for off-policy trainers: the staleness the
                lower count buys back is almost exactly cancelled.
    keep-steps  the ratio scaled by 1024/n, so optimizer steps per second are
                held at what the 1,024-environment configuration achieved.  Only
                fresh data is given up, and only staleness is bought.  F019's
                slopes make this the arm that should win: roughly +1.39 in log2
                for DQN at 64 environments, a 2.6x gain.

That second arm is the operating point the ladder could not reach, because it
fixed the environment count and expressed cadence as updates per vector step --
a quantity defined against 1,024 environments.

WALL-CLOCK SCORED: this must have the GPU to itself.
"""

from __future__ import annotations

import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pathlib import Path
from breakout_algorithms import BY_KEY, ON_POLICY
from exp_runner import run_experiment

ROOT = Path(__file__).resolve().parent.parent
STATE = ROOT / "benchmark_results" / "artifacts" / "breakout" / "highenv" / "state.json"
WINDOW = 300.0
ENV_COUNTS = (64, 256, 1024)
ALGOS = ("pqn", "sem_pqn", "dqn", "ibdqn", "hadamax", "c51", "sac", "bbf")
# ratio multipliers for the replay trainers: 1.0 keeps samples-per-transition,
# "steps" keeps optimizer steps per second at the 1024-env value.
ARMS = ("keep-ratio", "keep-steps")


def specs():
    state = json.loads(STATE.read_text())
    out = []
    for key in ALGOS:
        entry = state[key]
        best = entry["best"]
        knobs = best["knobs"]
        base_envs = entry["num_envs"]
        algo = BY_KEY[key]
        for envs in ENV_COUNTS:
            flags: list = []
            if algo.family == ON_POLICY:
                num_steps = knobs.get("num_steps") or 16
                minibatch = knobs.get("minibatch") or 512
                flags += ["--num-steps", num_steps,
                          "--num-minibatches", max(1, round(envs * num_steps / minibatch))]
                if knobs.get("update_epochs"):
                    flags += ["--update-epochs", knobs["update_epochs"]]
                num_steps_hint = num_steps
            else:
                num_steps_hint = None
            flags += list(knobs.get("extra", ()))
            if algo.family != ON_POLICY:
                batch = knobs.get("batch_size")
                # the ladder's cadence is per vector step at ITS env count; the
                # ratio is the quantity that means the same thing at any count
                base_ratio = knobs["updates_per_step"] * batch / base_envs
                for arm in ARMS:
                    if arm == "keep-steps" and envs == base_envs:
                        continue          # identical to keep-ratio at the base count
                    scale = 1.0 if arm == "keep-ratio" else base_envs / envs
                    off = list(flags) + ["--batch-size", batch,
                                         "--replay-ratio", base_ratio * scale]
                    if knobs.get("buffer_size"):
                        off += ["--buffer-size", knobs["buffer_size"]]
                    out.append({
                        "name": f"{key}-n{envs}-{arm}",
                        "algorithm": key, "variant": best["variant"], "num_envs": envs,
                        "total_timesteps": entry["total_timesteps"], "seed": 1,
                        "lr_mult": knobs.get("lr_mult", 1.0), "num_steps": None,
                        "arm": arm, "replay_ratio": base_ratio * scale,
                        "ladder_score": best["confirm_score"], "ladder_envs": base_envs,
                        "flags": off,
                    })
                continue
            out.append({
                "name": f"{key}-n{envs}",
                "arm": "on-policy",
                "algorithm": key,
                "variant": best["variant"],
                "num_envs": envs,
                "total_timesteps": entry["total_timesteps"],
                "seed": 1,
                "lr_mult": knobs.get("lr_mult", 1.0),
                "num_steps": num_steps_hint,
                "ladder_score": best["confirm_score"],
                "ladder_envs": base_envs,
                "flags": flags,
            })
    # every trainer at 1024 first: that column reproduces the published report and
    # is the check that this harness agrees with it before anything is claimed.
    return sorted(out, key=lambda s: (-s["num_envs"], ALGOS.index(s["algorithm"])))


if __name__ == "__main__":
    results = run_experiment("h014_freed_envs", specs(), timeout=1800.0,
                             wall_clock=WINDOW, dry_run="--dry-run" in sys.argv)
    print()
    by = {}
    for r in results:
        if r.get("score") is None:
            continue
        cell = by.setdefault(r["algorithm"], {})
        # report the better arm per count, and print the arms separately below
        cell[r["num_envs"]] = max(cell.get(r["num_envs"], 0.0), r["score"])
    print("\nby arm:")
    for r in sorted((r for r in results if r.get("score") is not None),
                    key=lambda r: (r["algorithm"], -r["num_envs"], r.get("arm", ""))):
        print(f"  {r['algorithm']:9s} n={r['num_envs']:5d} {r.get('arm','-'):11s} "
              f"ratio={r.get('replay_ratio') or float('nan'):8.3f}  score {r['score']:8.2f}  "
              f"sps {r['train_sps'] or 0:6d}")
    print(f"{'trainer':10s}" + "".join(f"{('n%d'%n):>10s}" for n in ENV_COUNTS)
          + f"{'ladder':>9s}{'best':>9s}{'gain':>8s}")
    rows = []
    for algo in ALGOS:
        row = by.get(algo, {})
        if not row:
            continue
        ladder = next(r["ladder_score"] for r in results if r["algorithm"] == algo)
        best_n, best_v = max(row.items(), key=lambda kv: kv[1])
        rows.append((best_v, algo, best_n))
        cells = "".join(f"{row[n]:10.2f}" if n in row else f"{'-':>10s}" for n in ENV_COUNTS)
        print(f"{algo:10s}{cells}{ladder:9.2f}{best_v:9.2f}{best_v/max(ladder,1e-9):7.2f}x")
    print("\ncorrected wall-clock ranking:")
    for i, (v, algo, n) in enumerate(sorted(rows, reverse=True), 1):
        print(f"  {i:2d}. {algo:10s} {v:8.2f}  at {n} envs")
