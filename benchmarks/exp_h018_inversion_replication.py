#!/usr/bin/env python3
"""H018: does the protocol inversion itself reproduce on a second game?

F001 is the finding everything else hangs off: the same ten trainers rank almost
oppositely under a 1M-transition published-hyperparameter protocol and a
five-minute 1,024-environment tuned protocol, sharing one of five top slots, with
Spearman +0.504.  F015 explains it -- on-policy trainers are paid in fresh
transitions and off-policy in gradient steps, and the two protocols supply
opposite currencies.

Both were measured on Breakout.  The explanation makes a prediction that a
second game can falsify cleanly: run the same ten trainers under both protocols
on SpaceInvaders and the rank correlation should again be middling, the top of
the two tables should again be nearly disjoint, and the trainers that climb
should again be the on-policy ones.

    sample arm  each trainer at ITS published operating point from the 1M
                benchmark -- its own environment count, batch size and replay
                ratio -- for 1,000,000 transitions, scored on the transition axis.
    clock arm   each trainer at the winning configuration the Breakout ladder
                found for it, at 1,024 environments, for a 300-second window,
                scored on the wall-clock axis.

The clock arm reuses Breakout-tuned hyperparameters on a different game, which
is a real limitation and is exactly what the original protocol does when it
reports a single number per algorithm.  The comparison is between protocols, not
between tunings, and both arms inherit the same handicap.

SpaceInvaders rather than Pong: a [-21, 21] range makes ranks unstable near the
floor, where most of these trainers will sit after a million transitions.

MIXED SCORING: the clock arm is wall-clock scored and needs the GPU to itself.
"""

from __future__ import annotations

import sys, os, csv, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pathlib import Path
from breakout_algorithms import BY_KEY, ON_POLICY
from exp_runner import run_experiment

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "benchmark_results" / "artifacts" / "breakout"
GAME = "SpaceInvadersNoFrameskip-v4"
BUDGET = 1_000_000
WINDOW = 300.0
CLOCK_ENVS = 1024
ALGOS = ("pqn", "sem_pqn", "hadamax", "ppo", "dqn", "ibdqn", "c51", "rainbow", "iqn", "sac")


def published() -> dict[str, dict]:
    out = {}
    for r in csv.DictReader((ART / "1m" / "summary.csv").open()):
        out[r["algorithm"]] = r
    return out


def sample_specs():
    pub = published()
    out = []
    for key in ALGOS:
        row = pub[key]
        algo = BY_KEY[key]
        envs = int(row["num_envs"])
        flags: list = []
        if algo.family != ON_POLICY and row["batch_size"] and row["replay_ratio"]:
            flags += ["--batch-size", int(float(row["batch_size"])),
                      "--replay-ratio", float(row["replay_ratio"])]
        out.append({
            "name": f"{key}-sample",
            "algorithm": key, "variant": "eager", "env_id": GAME,
            "num_envs": envs, "total_timesteps": BUDGET, "seed": 1,
            "arm": "sample", "flags": flags,
        })
    return out


def clock_specs():
    state = json.loads((ART / "highenv" / "state.json").read_text())
    out = []
    for key in ALGOS:
        entry = state[key]
        best = entry["best"]
        knobs = best["knobs"]
        algo = BY_KEY[key]
        flags: list = []
        hint = None
        if algo.family == ON_POLICY:
            ns = knobs.get("num_steps") or 16
            mb = knobs.get("minibatch") or 512
            flags += ["--num-steps", ns,
                      "--num-minibatches", max(1, round(CLOCK_ENVS * ns / mb))]
            if knobs.get("update_epochs"):
                flags += ["--update-epochs", knobs["update_epochs"]]
            hint = ns
        else:
            batch = knobs["batch_size"]
            ratio = knobs["updates_per_step"] * batch / entry["num_envs"]
            flags += ["--batch-size", batch, "--replay-ratio", ratio]
            if knobs.get("buffer_size"):
                flags += ["--buffer-size", knobs["buffer_size"]]
        flags += list(knobs.get("extra", ()))
        out.append({
            "name": f"{key}-clock",
            "algorithm": key, "variant": best["variant"], "env_id": GAME,
            "num_envs": CLOCK_ENVS, "total_timesteps": entry["total_timesteps"],
            "seed": 1, "lr_mult": knobs.get("lr_mult", 1.0), "num_steps": hint,
            "arm": "clock", "flags": flags,
        })
    return out


def report(sample, clock):
    from scipy.stats import spearmanr, mannwhitneyu
    s = {r["algorithm"]: r["score"] for r in sample if r.get("score") is not None}
    w = {r["algorithm"]: r["score"] for r in clock if r.get("score") is not None}
    shared = sorted(set(s) & set(w))
    if len(shared) < 4:
        print(f"\nonly {len(shared)} trainers finished both arms; no ranking yet")
        return
    rho, p = spearmanr([s[k] for k in shared], [w[k] for k in shared])
    rs = {k: i + 1 for i, k in enumerate(sorted(shared, key=lambda k: -s[k]))}
    rw = {k: i + 1 for i, k in enumerate(sorted(shared, key=lambda k: -w[k]))}
    print(f"\nSpaceInvaders, n={len(shared)}   Spearman {rho:+.3f} (p={p:.4f})")
    print("  Breakout reference (F001): +0.504\n")
    print(f"  {'trainer':10s} {'family':11s} {'sample':>9s} {'#':>3s} {'clock':>9s} {'#':>3s} {'move':>6s}")
    for k in sorted(shared, key=lambda k: rw[k]):
        print(f"  {k:10s} {BY_KEY[k].family:11s} {s[k]:9.2f} {rs[k]:3d} "
              f"{w[k]:9.2f} {rw[k]:3d} {rs[k]-rw[k]:+6d}")
    groups: dict = {}
    for k in shared:
        groups.setdefault(BY_KEY[k].family, []).append(rs[k] - rw[k])
    for fam, v in sorted(groups.items()):
        print(f"  mean rank change {fam:11s} {sum(v)/len(v):+5.1f}  (n={len(v)})")
    if len(groups.get("on_policy", [])) >= 3 and len(groups.get("off_policy", [])) >= 3:
        u, pu = mannwhitneyu(groups["on_policy"], groups["off_policy"], alternative="greater")
        print(f"  on-policy climbs more: U={u:.0f}, p={pu:.4f}   "
              f"(Breakout: p=0.0016)")


if __name__ == "__main__":
    dry = "--dry-run" in sys.argv
    sample = run_experiment("h018_inversion_sample", sample_specs(),
                            timeout=9000.0, dry_run=dry)
    clock = run_experiment("h018_inversion_clock", clock_specs(),
                           timeout=1800.0, wall_clock=WINDOW, dry_run=dry)
    if not dry:
        report(sample, clock)
