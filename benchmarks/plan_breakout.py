#!/usr/bin/env python3
"""Turn `calibration.jsonl` into the benchmark's operating point per algorithm.

Cost model
----------
For a replay-based trainer the wall clock of a fixed-budget run separates into a
collect term and a learn term::

    seconds(S, N, r) = S * t_collect(N)  +  S * (r / batch) * t_update

`t_collect(N)` comes from the probes that ran with the learner switched off;
`t_update` is recovered from the probes that ran it, by subtracting the collect
term measured at the same environment count.

The model's useful property is that **the learn term does not contain N**.
Environment count buys throughput only on the collect term, so it can be chosen
for speed without touching the gradient cadence that decides whether the
algorithm learns at all -- and the cadence can be chosen for learning without
regard to how many environments are running.

Choices
-------
* `replay_ratio` is the algorithm's **published** cadence (`paper_replay_ratio`
  in the registry: batch size divided by update period, so 32/4 = 8 for the DQN
  line).  It is reduced only when the learn term alone would blow the per-run
  wall-clock budget, and then only down a power-of-two ladder, with the
  reduction recorded in the plan so the report can flag it.
* `num_envs` is the smallest swept count whose collect term has stopped
  mattering -- 15% of the run by default.  Going beyond that spends environment
  memory and pushes the replay data further off-policy for no wall-clock return.
* On-policy and streaming trainers have no separable learner, so their choice is
  simply the swept count with the best measured throughput.
* Trainers with a `pinned_envs` design point (BTR, R2D2) keep it: their
  environment count is part of the published algorithm.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from breakout_algorithms import ALGORITHMS, BY_KEY, OFF_POLICY, ON_POLICY, ROOT

RESULTS = ROOT / "benchmark_results" / "artifacts" / "breakout"

# Share of total run time above which extra environments stop being worth it.
COLLECT_SHARE_TARGET = 0.15
# Never drop below this: CuLE is latency-bound with a handful of environments.
MIN_ENVS = 64
# Floors on how much learning the budget must still buy (see plan_throughput_only).
MIN_POLICY_UPDATES = 200  # on-policy rollouts over the whole run
MIN_STREAM_UPDATES = 100_000  # streaming parameter updates over the whole run


def load_calibration(path: Path) -> dict:
    """{(algorithm, variant): {"collect": {N: sps}, "learn": [...], "train": {N: sps}}}"""
    data: dict = defaultdict(
        lambda: {"collect": {}, "learn": [], "train": {}, "failed": []}
    )
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        key = (rec["algorithm"], rec["variant"])
        if rec.get("status") != "ok":
            data[key]["failed"].append({
                "probe": rec["probe"], "num_envs": rec["num_envs"],
                "status": rec["status"], "tail": rec.get("tail", "")[-300:],
            })
            continue
        result = rec["result"]
        if rec["probe"] == "collect":
            data[key]["collect"][rec["num_envs"]] = result["sps"]
        elif rec["probe"] == "learn":
            data[key]["learn"].append({
                "num_envs": rec["num_envs"], "replay_ratio": rec["replay_ratio"],
                "sps": result["sps"], "ups": result["ups"],
                "peak_cuda_memory_mb": result.get("peak_cuda_memory_mb"),
            })
        else:
            data[key]["train"][rec["num_envs"]] = result["sps"]
    return data


def update_cost(algo, measurements: dict) -> tuple[float | None, list[dict]]:
    """Seconds per gradient update, averaged over the learner probes.

    The number of updates is taken from the probe's *measured* `ups`/`sps`
    rather than from the nominal replay ratio.  Those two disagree whenever a
    probe's window is short enough that the buffer is still filling -- a trainer
    at ratio 32 cannot serve 256 minibatches per vector step until it holds
    enough transitions to sample them -- and dividing by a nominal count the run
    never reached would understate the per-update cost, which is exactly the
    error that makes a run overshoot its predicted wall clock.
    """
    estimates = []
    for probe in measurements["learn"]:
        collect_sps = measurements["collect"].get(probe["num_envs"])
        if not collect_sps or not probe["sps"] or not probe["ups"]:
            continue
        learner_seconds_per_transition = 1.0 / probe["sps"] - 1.0 / collect_sps
        measured_updates_per_transition = probe["ups"] / probe["sps"]
        if measured_updates_per_transition <= 0 or learner_seconds_per_transition <= 0:
            continue
        nominal = (probe["replay_ratio"] / algo.batch_size) if probe["replay_ratio"] else None
        estimates.append({
            "replay_ratio": probe["replay_ratio"],
            "t_update": learner_seconds_per_transition / measured_updates_per_transition,
            "measured_ups": probe["ups"],
            "measured_updates_per_transition": measured_updates_per_transition,
            "nominal_updates_per_transition": nominal,
        })
    if not estimates:
        return None, []
    # The heavier probe is the more representative one: its window spends
    # proportionally less time on buffer warm-up.
    return max(e["t_update"] for e in estimates), estimates


def ratio_ladder(published: float) -> list[float]:
    """Powers of two down from the published cadence, with a floor of 1."""
    ladder, ratio = [], float(published)
    while ratio >= 1.0:
        ladder.append(ratio)
        ratio /= 2.0
    if not ladder or ladder[-1] != 1.0:
        ladder.append(1.0)
    return ladder


def plan_fixed_cadence(algo, measurements: dict, steps: int, variant: str) -> dict:
    """Trainers whose cadence cannot exceed one update per vector step.

    `endpoint_ddqn` gates its learner on `vector_step % train_frequency == 0`, so
    at N environments with `train_frequency=1` its replay ratio is batch/N and
    nothing can raise it.  Preserving the published ratio therefore *chooses* the
    environment count rather than leaving it free: take the largest N that still
    reaches it.
    """
    # Only probes that ran with the trainer's own cadence mean anything here; a
    # probe that passed an explicit `--replay-ratio` measured a different
    # configuration and must not be mistaken for this one.
    probes = {p["num_envs"]: p for p in measurements["learn"] if p["replay_ratio"] is None}
    if not probes:
        return {"error": "no learner probes at the trainer's own cadence",
                "failed": measurements["failed"]}
    published = float(algo.paper_replay_ratio)
    if algo.use_shipped_cadence:
        # BTR and R2D2 ship the published vectorized configuration already, so
        # the environment count is pinned and the cadence is whatever the file
        # does at that count.
        num_envs = algo.pinned_envs if algo.pinned_envs in probes else min(probes)
        achieved = published
    else:
        feasible = [n for n in probes if algo.batch_size / n >= published]
        num_envs = max(feasible) if feasible else min(probes)
        achieved = algo.batch_size / num_envs
    return {
        "variant": variant,
        "num_envs": num_envs,
        "replay_ratio": achieved,
        "paper_replay_ratio": published,
        "replay_ratio_reduced": achieved < published,
        "buffer_size": min(1_000_000, steps) if algo.has_buffer_size else None,
        "predicted_seconds": steps / probes[num_envs]["sps"],
        "predicted_collect_seconds": None,
        "predicted_learn_seconds": None,
        "train_sps": {str(n): p["sps"] for n, p in probes.items()},
        "cadence_note": ("shipped vectorized configuration; --replay-ratio not overridden"
                         if algo.use_shipped_cadence
                         else "at most one update per vector step (--train-frequency 1)"),
    }


def plan_off_policy(algo, measurements: dict, budget: float, steps: int,
                    variant: str) -> dict:
    if not algo.has_replay_ratio or algo.use_shipped_cadence:
        return plan_fixed_cadence(algo, measurements, steps, variant)
    collect = measurements["collect"]
    if not collect:
        return {"error": "no successful collect probes", "failed": measurements["failed"]}
    t_update, estimates = update_cost(algo, measurements)
    if t_update is None:
        # Without a learner cost the model degenerates to collect-only, which
        # would silently pick the largest environment count and a cadence nobody
        # verified was affordable.  Refuse rather than plan a fiction.
        return {"error": "no usable learner probes; cannot cost the gradient cadence",
                "failed": measurements["failed"]}

    published = float(algo.paper_replay_ratio)
    chosen_ratio, reduced = published, False
    for ratio in ratio_ladder(published):
        if steps * (ratio / algo.batch_size) * t_update <= budget * (1 - COLLECT_SHARE_TARGET):
            chosen_ratio = ratio
            break
        reduced = True
    else:
        chosen_ratio = 1.0
    learn_seconds = steps * (chosen_ratio / algo.batch_size) * t_update

    # Environment count: a pinned design point wins; otherwise the smallest
    # swept count whose collect term has stopped mattering.
    if algo.pinned_envs and algo.pinned_envs in collect:
        num_envs = algo.pinned_envs
    else:
        num_envs = None
        for candidate in sorted(collect):
            if candidate < MIN_ENVS:
                continue
            collect_seconds = steps / collect[candidate]
            if collect_seconds <= COLLECT_SHARE_TARGET * (collect_seconds + learn_seconds):
                num_envs = candidate
                break
        if num_envs is None:  # collect always dominates: take the fastest measured
            num_envs = max(collect, key=lambda n: collect[n])

    collect_seconds = steps / collect[num_envs]
    peak_mb = max((p.get("peak_cuda_memory_mb") or 0) for p in measurements["learn"]) \
        if measurements["learn"] else 0
    return {
        "variant": variant,
        "num_envs": num_envs,
        "replay_ratio": chosen_ratio,
        "paper_replay_ratio": published,
        "replay_ratio_reduced": reduced,
        "buffer_size": min(1_000_000, steps),
        "predicted_seconds": collect_seconds + learn_seconds,
        "predicted_collect_seconds": collect_seconds,
        "predicted_learn_seconds": learn_seconds,
        "t_update_seconds": t_update,
        "collect_sps": {str(n): s for n, s in collect.items()},
        "update_estimates": estimates,
        "probe_peak_cuda_mb": peak_mb,
    }


def plan_throughput_only(algo, measurements: dict, steps: int, variant: str) -> dict:
    """On-policy and streaming: fastest environment count the budget still allows.

    Throughput alone would pick the largest count swept, but for these families
    the environment count also sets how many *parameter updates* the budget buys.
    An on-policy rollout is `num_envs * num_steps` transitions, so 1024
    environments at the shipped 128 steps would leave seven policy updates in a
    1M-transition run; a streaming trainer performs one bounded update per vector
    step, so N environments divide its update count by N.  Both therefore get an
    explicit floor on updates, and the fastest count that clears it wins.
    """
    train = measurements["train"]
    if not train:
        return {"error": "no successful throughput probes", "failed": measurements["failed"]}

    if algo.family == ON_POLICY:
        def updates(n: int) -> float:
            return steps / (n * algo.num_steps)
        floor = MIN_POLICY_UPDATES
    else:
        def updates(n: int) -> float:
            return steps / n
        floor = MIN_STREAM_UPDATES

    allowed = [n for n in train if updates(n) >= floor]
    if algo.pinned_envs and algo.pinned_envs in train:
        num_envs = algo.pinned_envs
    elif allowed:
        num_envs = max(allowed, key=lambda n: train[n])
    else:  # budget cannot afford the floor anywhere: take the smallest swept
        num_envs = min(train)
    config = {
        "variant": variant,
        "num_envs": num_envs,
        "predicted_seconds": steps / train[num_envs],
        "train_sps": {str(n): s for n, s in train.items()},
    }
    if algo.family == ON_POLICY:
        config["num_steps"] = algo.num_steps
        if algo.minibatch_size:
            batch = num_envs * algo.num_steps
            config["rollout_batch"] = batch
            config["minibatch_size"] = algo.minibatch_size
            config["num_minibatches"] = max(1, round(batch / algo.minibatch_size))
            config["update_epochs"] = algo.update_epochs
    return config


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--calibration", type=Path, default=RESULTS / "calibration.jsonl")
    parser.add_argument("--output", type=Path, default=RESULTS / "plan.json")
    parser.add_argument("--total-timesteps", type=int, default=1_000_000)
    parser.add_argument("--budget-hours", type=float, default=1.5,
                        help="per-run wall-clock target; the replay ratio is reduced "
                             "only when the learner alone would exceed it")
    parser.add_argument("--variant", default="torchcompile",
                        choices=["eager", "torchcompile"])
    args = parser.parse_args()

    measurements = load_calibration(args.calibration)
    budget = args.budget_hours * 3600.0
    configs, order, problems = {}, [], {}

    other = "eager" if args.variant == "torchcompile" else "torchcompile"
    for algo in ALGORITHMS:
        # `sac_atari.py` and `ppo_atari_lstm.py` have no `--benchmark` flag, so
        # they can only be probed through their twin.  The environment-count
        # choice still transfers -- it is driven by the collect term, which the
        # compiled learner does not touch -- so fall back to the twin's
        # measurements while still running the variant that was asked for.
        data = measurements.get((algo.key, args.variant))
        calibrated_with = args.variant
        if not data or not (data["collect"] or data["train"] or data["learn"]):
            data = measurements.get((algo.key, other))
            calibrated_with = other
        if not data:
            problems[algo.key] = "no calibration probes"
            continue
        if algo.family == OFF_POLICY:
            config = plan_off_policy(algo, data, budget, args.total_timesteps, args.variant)
        else:
            config = plan_throughput_only(algo, data, args.total_timesteps, args.variant)
        if "error" in config:
            problems[algo.key] = config
            continue
        config["calibrated_with"] = calibrated_with
        configs[algo.key] = config
        order.append(algo.key)

    # Cheapest first: a broken config surfaces early instead of behind a long run.
    order.sort(key=lambda k: configs[k]["predicted_seconds"])

    plan = {
        "total_timesteps": args.total_timesteps,
        "budget_hours": args.budget_hours,
        "collect_share_target": COLLECT_SHARE_TARGET,
        "order": order,
        "configs": configs,
        "problems": problems,
    }
    args.output.write_text(json.dumps(plan, indent=1, sort_keys=True))

    total = sum(configs[k]["predicted_seconds"] for k in order)
    print(f"{len(order)} algorithms planned, predicted total {total / 3600:.1f} h\n")
    header = (f"{'algorithm':14s} {'variant':13s} {'envs':>5s} {'ratio':>7s} "
              f"{'paper':>6s} {'collect':>9s} {'learn':>9s} {'total':>9s}")
    print(header)
    print("-" * len(header))
    for key in order:
        c = configs[key]
        ratio, paper = c.get("replay_ratio"), c.get("paper_replay_ratio")
        collect, learn = c.get("predicted_collect_seconds"), c.get("predicted_learn_seconds")
        print(f"{key:14s} {c['variant']:13s} {c['num_envs']:5d} "
              f"{(f'{ratio:g}' if ratio else '-'):>7s} "
              f"{(f'{paper:g}' if paper else '-'):>6s} "
              f"{(f'{collect / 60:.1f}m' if collect else '-'):>9s} "
              f"{(f'{learn / 60:.1f}m' if learn else '-'):>9s} "
              f"{c['predicted_seconds'] / 60:8.1f}m"
              f"{' *' if c.get('replay_ratio_reduced') else ''}")
    print("\n* replay ratio reduced below the published cadence to fit the budget")
    for key, why in problems.items():
        print(f"PROBLEM {key}: {why if isinstance(why, str) else why.get('error')}")


if __name__ == "__main__":
    main()
