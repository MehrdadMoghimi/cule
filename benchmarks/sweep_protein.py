#!/usr/bin/env python3
"""Cost-aware Bayesian search over one algorithm's hyperparameters.

This is the *depth* pass that `sweep_highenv.py` is not.  That sweep spends nine
to eleven trials on each of thirty-five algorithms, walking a hand-written
ladder of candidates; nine trials is below the number at which any model-based
optimizer beats the ladder, so the ladder is the right tool there.  This file
takes the handful of algorithms the ladder ranked highest and spends a hundred
or more trials on each, which is the regime CARBS was built for.

The optimizer is Protein (`protein.py`, vendored from PufferLib), a reworked
CARBS: a Gaussian process over the score, a second one over the cost, and a
search restricted to the neighbourhood of the score-versus-cost Pareto front.

What makes it the right optimizer for this repository
-----------------------------------------------------
The score here is already a point on a cost-score curve rather than a scalar:
`sweep_highenv.py` grades a configuration by the return it reaches after a fixed
number of *seconds* of training.  Fixing that number, as the ladder does, throws
away the only axis CARBS knows how to exploit, so here the training window is
itself a searched dimension -- `train_seconds`, log-uniform from 30 s to the cap
-- and it is the dimension declared as Protein's `cost_param`.  The consequences
are the point of the exercise:

* Cheap exploration is automatic.  Protein opens at ~30 s trials and expands
  along the cost axis only where the score GP says the neighbourhood is worth
  paying for, so the early trials cost a tenth of a ladder rung.
* The score GP sees `train_seconds` as an input, so it learns each
  configuration's return-versus-time curve rather than one point on it, and can
  extrapolate a 300 s ranking from 40 s evidence.  That is the "tune as you
  scale" claim, and it is what a successive-halving rung cannot do: ASHA can
  only discard, never extrapolate.
* Cost is measured as *wall clock including startup*, not as the requested
  window.  A `_torchcompile.py` twin therefore pays for its compile and
  graph-capture time out of the same budget the eager file does not spend, and
  the cost GP learns that tax instead of the protocol hiding it.

What is held fixed, and why
---------------------------
`--num-envs`, the eager/compiled variant, and `--total-timesteps` all come from
`sweep_highenv.py`'s `state.json` and are not searched.

The last of those matters.  Every annealing schedule in these trainers -- the
epsilon ramp, the learning-rate decay, the entropy coefficient -- is defined as
a fraction of `--total-timesteps`, so searching it would make a faster
configuration anneal more slowly and confound the comparison.  Holding it at the
value the ladder used also keeps every score here directly comparable with the
numbers in the ladder's `REPORT.md`.  The cost is that a configuration much
faster than the one that sized the budget can exhaust it before its window
elapses; that is detected rather than ignored, recorded as `budget_exhausted`,
and reported.  `--timestep-margin` raises the budget when it happens, at the
price of comparability with the baseline.

The variant is fixed because Protein searches a continuous box and has no
categorical dimension.  Eager versus compiled is not a knob on a continuum -- it
changes which flags exist and where the replay buffer lives -- so it is a
separate *study* instead, and `--variant both` runs the two and reports them
side by side.

Stages
------
    search    the Bayesian loop: suggest, run, observe, append to the JSONL.
              Resumable -- a restart replays every recorded observation into a
              fresh Protein before asking for the next suggestion.
    confirm   the best configuration re-run at the full window on further seeds.
              The search runs one seed at 30-300 s windows, so its winner is
              chosen partly on noise; this is the cheap defence, and it is a
              separate stage because it is only worth paying for once.

Outputs, under `benchmark_results/artifacts/breakout/protein/`:
    trials.jsonl      one record per configuration run, per study
    curves/<id>.csv   return-versus-time samples for every trial
    logs/<id>.log     complete trainer output
    REPORT.md         written by `--report`
"""

from __future__ import annotations

import argparse
import atexit
import json
import math
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from breakout_algorithms import BY_KEY, ENV_ID, OFF_POLICY, ON_POLICY, ROOT
from protein import Protein
import sweep_highenv as ladder
from sweep_highenv import (
    GPU_BUFFER,
    GPU_STORAGE_TWINS,
    HOST_BUFFER,
    Knobs,
    Trial,
    acquire_lock,
    append_trial,
    args_fields,
    build_command,
    has,
    lr_flags,
    run_trial,
    trainer_path,
)

RESULTS = ROOT / "benchmark_results" / "artifacts" / "breakout" / "protein"
LADDER_STATE = ROOT / "benchmark_results" / "artifacts" / "breakout" / "highenv" / "state.json"

# The floor of the cost axis.  Below roughly this, a 1,024-environment trainer
# has not finished enough episodes for the closing-40% average to mean anything,
# and the compiled twins have barely finished compiling.
MIN_WINDOW = 30.0


# --------------------------------------------------------------------------- #
# Search space
# --------------------------------------------------------------------------- #

def _log(lo, hi):
    return dict(distribution="log_normal", min=float(lo), max=float(hi), scale="auto")


def _pow2(lo, hi):
    return dict(distribution="uniform_pow2", min=int(lo), max=int(hi), scale="auto")


def _int(lo, hi):
    return dict(distribution="int_uniform", min=int(lo), max=int(hi), scale="auto")


def search_space(algo, variant: str, num_envs: int, max_window: float,
                 search_envs: bool = False) -> dict:
    """The dimensions this particular trainer actually exposes.

    Every dimension is gated on the flag existing in the trainer's `Args`, the
    same introspection `sweep_highenv.build_command` uses to decide what to
    emit.  A dimension that the trainer would silently ignore is worse than a
    missing one: the GP would spend its observations explaining a coordinate
    that does not affect the score.
    """
    path = trainer_path(algo, variant)
    space: dict = {
        "metric": "score",
        "goal": "maximize",
        "metric_distribution": "uniform",
        # Each observation is one training run costing minutes, so the failure
        # classifier -- which learns *where* configurations crash, typically the
        # large-batch large-buffer corner that runs the card out of memory -- is
        # worth its keep.  Protein enables it only at downsample 1.
        "downsample": 1,
        "early_stop_quantile": 0.3,
        # The cost dimension, and the reason this optimizer is not a scalar one.
        "train_seconds": _log(MIN_WINDOW, max_window),
    }

    if search_envs:
        # F010: with batch size and optimizer steps both pinned, moving 64 -> 1024
        # environments still costs DQN 4.6x, because the trainer takes 256 gradient
        # steps between vector steps at 1,024 against 16 at 64 and the batch it
        # trains on is that much staler.  Environment count is therefore a real
        # axis of the objective, not a throughput setting to be maximised, and a
        # search that fixes it is searching a slice through the optimum.
        space["num_envs"] = _pow2(64, num_envs)

    if lr_flags(path, 2.0):
        space["lr_mult"] = _log(0.125, 8.0)

    if algo.family == ON_POLICY:
        if has(path, "num_steps"):
            space["num_steps"] = _pow2(4, 128)
        if has(path, "num_minibatches"):
            # Bounded above by the smallest rollout the `num_steps` range can
            # produce, so that `num_minibatches` never rounds to 1 by accident.
            space["minibatch"] = _pow2(256, max(512, num_envs * 4))
        if has(path, "update_epochs"):
            space["update_epochs"] = _int(1, 8)
    elif algo.family == OFF_POLICY:
        if has(path, "batch_size"):
            # The floor is the published 32, not a round number.  The ladder
            # never went below 256 for 15 of the 17 replay trainers, and F006/F008
            # place the interesting region below that -- a search space that
            # excludes the answer cannot be argued with.
            space["batch_size"] = _pow2(32, 4096)
        if has(path, "replay_ratio") or has(path, "learner_updates_per_vector_step"):
            space["updates_per_step"] = _log(0.03125, 64.0)
        if has(path, "buffer_size"):
            # A GPU-resident transition costs ~56 KB against ~7 KB on the host,
            # so the two storages are not the same axis and cannot share bounds.
            on_device = variant == "torchcompile" and algo.key in GPU_STORAGE_TWINS
            space["buffer_size"] = (_pow2(8192, GPU_BUFFER) if on_device
                                    else _pow2(65536, HOST_BUFFER))
    # Streaming trainers hold neither a rollout nor a buffer; learning rate and
    # the window are the whole space, which is exactly why they are cheap.
    return space


KNOB_NAMES = ("batch_size", "updates_per_step", "buffer_size",
              "num_steps", "minibatch", "update_epochs", "lr_mult")
INTEGER_KNOBS = ("batch_size", "buffer_size", "num_steps", "minibatch", "update_epochs")


def to_knobs(hypers: dict, extra: tuple[str, ...]) -> tuple[Knobs, float]:
    """Protein's suggestion as a `Knobs` plus the window, in native Python types.

    The unnormalizers return numpy scalars, which survive as far as the
    `str(...)` in `build_command` and then reach argparse as `"3.0"` where an
    int was wanted.  Coercing here also keeps the JSONL serializable.
    """
    fields = {}
    for name in KNOB_NAMES:
        if name in hypers:
            value = hypers[name]
            fields[name] = int(round(float(value))) if name in INTEGER_KNOBS else float(value)
    return Knobs(extra=extra, **fields), float(hypers["train_seconds"])


# --------------------------------------------------------------------------- #
# Studies
# --------------------------------------------------------------------------- #

def study_id(algorithm: str, variant: str) -> str:
    return f"{algorithm}__{variant}"


def load_ladder_state() -> dict:
    if not LADDER_STATE.exists():
        raise SystemExit(
            f"{LADDER_STATE} not found -- this search takes its environment count, "
            f"variant and timestep budget from the ladder sweep.  Run\n"
            f"    python benchmarks/sweep_highenv.py --stages feasibility screen\n"
            f"first, or pass --num-envs/--variant/--total-timesteps explicitly."
        )
    return json.loads(LADDER_STATE.read_text())


def default_algorithms(state: dict, count: int) -> list[str]:
    """The ladder's top finishers, which is the only reason to spend a depth pass."""
    ranked = sorted(
        ((entry["best"]["confirm_score"] or 0.0, key) for key, entry in state.items()
         if entry.get("best")),
        reverse=True,
    )
    return [key for _, key in ranked[:count]]


def load_observations(path: Path, sid: str) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        if line.strip():
            record = json.loads(line)
            if record.get("study") == sid:
                out.append(record)
    return out


def replay(protein: Protein, records: list[dict]) -> None:
    """Rebuild a Protein's posterior from the JSONL so a restart resumes."""
    for record in records:
        protein.observe(record["hypers"], record["score"] or 0.0, record["cost"],
                        is_failure=record["is_failure"])
    # `observe` does not advance the suggestion counter, and the counter is what
    # decides whether the next suggestion comes from the Sobol sequence or the
    # GP.  Without this a resumed study restarts its random phase from zero.
    protein.suggestion_idx = len(records)


def make_protein(algo, variant: str, num_envs: int, args) -> Protein:
    config = search_space(algo, variant, num_envs, args.max_window, args.search_envs)
    print(f"  dimensions: {', '.join(k for k in config if isinstance(config[k], dict))}")
    return Protein(
        config,
        cost_param="train_seconds",
        max_suggestion_cost=args.max_cost,
        num_random_samples=args.random_samples,
        use_gpu=False,  # the GP is tiny and the card is busy training
    )


def observe_record(sid: str, algo, variant: str, num_envs: int, hypers: dict,
                   trial: Trial, total: int, seed: int, result: dict, cmd: list[str],
                   suggestion: dict) -> dict:
    record = ladder.record_trial(algo, trial, total, seed, result, cmd)
    score = result["score"]
    failed = result["status"] != "ok" or score is None
    # A run that ends because it ran out of timesteps rather than out of window
    # scored on a shorter window than it was charged for.  It is still a valid
    # observation -- the score and the cost are both real -- but the flag is what
    # makes a systematically truncated study visible instead of merely puzzling.
    exhausted = (not failed and result["trained_seconds"] < 0.9 * trial.window
                 and result["returncode"] == 0)
    record.update({
        "study": sid,
        "hypers": {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                   for k, v in hypers.items()},
        "cost": result["wall_seconds"],
        "is_failure": failed,
        "budget_exhausted": exhausted,
        "predicted": {k: (float(v) if isinstance(v, (int, float, np.floating)) else str(v))
                      for k, v in suggestion.items()},
    })
    return record


def run_study(algo, variant: str, args, state: dict, trials_path: Path) -> None:
    entry = state[algo.key]
    num_envs = args.num_envs or entry["num_envs"]
    total = int(entry["total_timesteps"] * args.timestep_margin)
    # Flags the ladder found necessary but that are not on a continuum -- BBF's
    # epsilon schedule, RND's normalization warmup -- ride along unsearched.
    extra = tuple(entry.get("best", {}).get("knobs", {}).get("extra", ()))
    sid = study_id(algo.key, variant)

    done = load_observations(trials_path, sid)
    print(f"\n{algo.label} [{variant}] n={num_envs} total_timesteps={total:,}"
          + (f" extra={' '.join(extra)}" if extra else ""))
    protein = make_protein(algo, variant, num_envs, args)
    if done:
        replay(protein, done)
        best = max((r["score"] or 0.0) for r in done)
        print(f"  resumed from {len(done)} observations, best so far {best:.2f}")

    path = trainer_path(algo, variant)
    for index in range(len(done), args.trials):
        hypers, suggestion = protein.suggest(None)
        knobs, window = to_knobs(hypers, extra)
        window = max(MIN_WINDOW, min(window, args.max_window))
        # `build_command` converts a cadence into --replay-ratio using the trial's
        # environment count, so the suggested count has to reach the Trial rather
        # than only the command line.
        trial_envs = int(hypers.get("num_envs", num_envs))
        trial = Trial(algo.key, "protein", variant, trial_envs, knobs, window,
                      label=f"p{index:03d}")
        cmd = build_command(algo, trial, total, args.seed)

        bar_scale = trial_envs
        if algo.family == ON_POLICY:
            bar_scale *= knobs.num_steps or args_fields(path).get("num_steps") or 1

        print(f"  {index:3d}/{args.trials} n{trial_envs:<5d} {window:5.0f}s "
              f"{'-'.join(knobs.label_parts()) or 'defaults':44s} ", end="", flush=True)
        if args.dry_run:
            print("\n      " + " ".join(cmd[1:]))
            continue

        result = run_trial(cmd, bar_scale, window,
                           args.results / "logs" / f"{sid}__{trial.label}.log",
                           args.results / "curves" / f"{sid}__{trial.label}.csv",
                           args.startup_cap, args.stats_interval)
        record = observe_record(sid, algo, variant, trial_envs, hypers, trial, total,
                                args.seed, result, cmd, suggestion)
        append_trial(trials_path, record)
        protein.observe(hypers, record["score"] or 0.0, record["cost"],
                        is_failure=record["is_failure"])

        mark = "!" if record["is_failure"] else ("~" if record["budget_exhausted"] else " ")
        print(f"{mark}score={('%.2f' % record['score']) if record['score'] is not None else '  -  '} "
              f"cost={record['cost']:5.0f}s sps={result['train_sps'] or 0:>6} "
              f"[{result['status']}]", flush=True)
        if record["is_failure"]:
            print("      " + result["tail"][-300:].replace("\n", "\n      "), flush=True)


# --------------------------------------------------------------------------- #
# Confirmation
# --------------------------------------------------------------------------- #

def confirm_study(algo, variant: str, args, state: dict, trials_path: Path) -> None:
    """Re-run the search winner at the full window on further seeds.

    One seed and a window as short as 30 s is enough to rank neighbourhoods; it
    is not enough to separate two configurations whose scores differ by less
    than Breakout's run-to-run spread.  A GP-guided search makes that worse than
    a ladder does, because it deliberately concentrates its trials wherever the
    noise happened to look good, so the winner is the observation most likely to
    be a lucky one.  Re-running only the winner, only at the end, costs a few
    trials and is the difference between a number and a claim.
    """
    sid = study_id(algo.key, variant)
    records = [r for r in load_observations(trials_path, sid)
               if not r["is_failure"] and r["score"] is not None]
    if not records:
        print(f"  {sid}: nothing to confirm")
        return
    winner = max(records, key=lambda r: r["score"])
    entry = state[algo.key]
    num_envs = args.num_envs or entry["num_envs"]
    total = int(entry["total_timesteps"] * args.timestep_margin)
    extra = tuple(entry.get("best", {}).get("knobs", {}).get("extra", ()))
    knobs, _ = to_knobs(winner["hypers"], extra)
    num_envs = int(winner["hypers"].get("num_envs", num_envs))
    path = trainer_path(algo, variant)

    print(f"\n{algo.label} [{variant}] confirming {winner['label']} "
          f"(search score {winner['score']:.2f} at {winner['window']:.0f}s)")
    for seed in range(args.seed, args.seed + args.confirm_seeds):
        label = f"confirm-s{seed}"
        trial = Trial(algo.key, "confirm", variant, num_envs, knobs,
                      args.max_window, label=label)
        cmd = build_command(algo, trial, total, seed)
        bar_scale = num_envs
        if algo.family == ON_POLICY:
            bar_scale *= knobs.num_steps or args_fields(path).get("num_steps") or 1
        print(f"  seed {seed} ", end="", flush=True)
        if args.dry_run:
            print("\n      " + " ".join(cmd[1:]))
            continue
        result = run_trial(cmd, bar_scale, args.max_window,
                           args.results / "logs" / f"{sid}__{label}.log",
                           args.results / "curves" / f"{sid}__{label}.csv",
                           args.startup_cap, args.stats_interval)
        record = observe_record(sid, algo, variant, num_envs, winner["hypers"], trial,
                                total, seed, result, cmd, {})
        record["stage"] = "confirm"
        append_trial(trials_path, record)
        print(f"score={('%.2f' % record['score']) if record['score'] is not None else '  -  '} "
              f"cost={record['cost']:5.0f}s [{result['status']}]", flush=True)


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #

def pareto(records: list[dict]) -> list[dict]:
    """Cheapest-first non-dominated points of the cost-score cloud."""
    usable = sorted((r for r in records if not r["is_failure"] and r["score"] is not None),
                    key=lambda r: r["cost"])
    front, best = [], -math.inf
    for record in usable:
        if record["score"] > best:
            front.append(record)
            best = record["score"]
    return front


def describe(record: dict) -> str:
    knobs = record.get("knobs", {})
    parts = [f"envs={record['num_envs']}"] if record.get("num_envs") else []
    for name, fmt in (("batch_size", "batch={}"), ("updates_per_step", "updates/step={:.3g}"),
                      ("buffer_size", "buffer={}"), ("num_steps", "num_steps={}"),
                      ("minibatch", "minibatch={}"), ("update_epochs", "epochs={}")):
        if knobs.get(name):
            parts.append(fmt.format(knobs[name]))
    if knobs.get("lr_mult", 1.0) != 1.0:
        parts.append(f"lr x{knobs['lr_mult']:.3g}")
    return ", ".join(parts) or "defaults"


def write_report(args, state: dict, trials_path: Path) -> None:
    records = [json.loads(line) for line in trials_path.read_text().splitlines() if line.strip()]
    studies: dict[str, list[dict]] = {}
    for record in records:
        studies.setdefault(record["study"], []).append(record)

    lines = [
        "# Cost-aware Bayesian search on Breakout",
        "",
        "Protein (a reworked CARBS) over each algorithm's hyperparameters at the",
        "environment count `sweep_highenv.py` fixed, with the training window",
        f"searched from {MIN_WINDOW:.0f} s to {args.max_window:.0f} s as the cost",
        "dimension.  Score is the mean training episodic return over the closing",
        "40% of the window.  CuLE GPU backend, `" + ENV_ID + "`.",
        "",
        "The last column is the ladder sweep's own five-minute result for the same",
        "algorithm.  It is only comparable to a row whose window reached 300 s -- a",
        "search that stopped short of the cost cap is a cheaper number, not a worse one.",
        "",
        "| study | trials | GPU-hours | best score | window | configuration | ladder best (300s) |",
        "|---|---:|---:|---:|---:|---|---:|",
    ]
    for sid, group in sorted(studies.items()):
        search = [r for r in group if r["stage"] == "protein"]
        usable = [r for r in search if not r["is_failure"] and r["score"] is not None]
        hours = sum(r["cost"] for r in group) / 3600.0
        algorithm = sid.split("__")[0]
        baseline = (state.get(algorithm, {}).get("best") or {}).get("confirm_score")
        if not usable:
            lines.append(f"| {sid} | {len(search)} | {hours:.2f} | - | - | all failed | "
                         f"{baseline if baseline is not None else '-'} |")
            continue
        best = max(usable, key=lambda r: r["score"])
        lines.append(
            f"| {sid} | {len(search)} | {hours:.2f} | {best['score']:.2f} | "
            f"{best['window']:.0f}s | {describe(best)} | "
            f"{('%.2f' % baseline) if baseline is not None else '-'} |"
        )

    for sid, group in sorted(studies.items()):
        search = [r for r in group if r["stage"] == "protein"]
        confirms = [r for r in group if r["stage"] == "confirm"
                    and not r["is_failure"] and r["score"] is not None]
        front = pareto(search)
        if not front:
            continue
        lines += ["", f"## {sid}", "",
                  "Cost-score Pareto front -- the cheapest configuration that reached each",
                  "score.  A flat tail means the extra seconds bought nothing, which is the",
                  "signal that the window, not the configuration, was the binding constraint.",
                  "", "| cost | window | score | configuration |", "|---:|---:|---:|---|"]
        for record in front:
            lines.append(f"| {record['cost']:.0f}s | {record['window']:.0f}s | "
                         f"{record['score']:.2f} | {describe(record)} |")
        failures = [r for r in search if r["is_failure"]]
        truncated = [r for r in search if r.get("budget_exhausted")]
        notes = []
        if failures:
            notes.append(f"{len(failures)} of {len(search)} trials failed")
        if truncated:
            notes.append(f"{len(truncated)} exhausted `--total-timesteps` before the window "
                         f"(raise `--timestep-margin`)")
        if notes:
            lines += ["", "Notes: " + "; ".join(notes) + "."]
        if confirms:
            scores = [r["score"] for r in confirms]
            lines += ["", f"Confirmation at {args.max_window:.0f}s over {len(scores)} seeds: "
                          f"mean {sum(scores) / len(scores):.2f}, "
                          f"spread {min(scores):.2f}-{max(scores):.2f}.  "
                          f"Command:", "", "```", " ".join(
                              str(part) for part in confirms[0]["command"][1:]), "```"]

    report = args.results / "REPORT.md"
    report.write_text("\n".join(lines) + "\n")
    print(f"wrote {report}")


# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--algorithms", nargs="+",
                        help="default: the top --top finishers of the ladder sweep")
    parser.add_argument("--top", type=int, default=5)
    parser.add_argument("--variant", default="ladder",
                        choices=["ladder", "eager", "torchcompile", "both"],
                        help="'ladder' uses whichever variant won the ladder sweep")
    parser.add_argument("--stages", nargs="+", default=["search"],
                        choices=["search", "confirm", "report"])
    parser.add_argument("--trials", type=int, default=100,
                        help="total observations per study, resume included")
    parser.add_argument("--random-samples", type=int, default=12,
                        help="Sobol trials before the GP takes over")
    parser.add_argument("--max-window", type=float, default=300.0,
                        help="top of the cost axis, and the confirmation window")
    parser.add_argument("--max-cost", type=float, default=420.0,
                        help="wall-clock ceiling Protein will suggest, startup included")
    parser.add_argument("--confirm-seeds", type=int, default=3)
    parser.add_argument("--timestep-margin", type=float, default=1.0,
                        help="scales the ladder's --total-timesteps; >1 stops fast "
                             "configurations exhausting it, at the cost of comparability")
    parser.add_argument("--num-envs", type=int, default=None,
                        help="override the ladder's environment count (the ceiling "
                             "of the search when --search-envs is set)")
    parser.add_argument("--search-envs", action="store_true",
                        help="search the environment count from 64 up to the ladder's "
                             "value instead of pinning it (see F010)")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--startup-cap", type=float, default=300.0)
    parser.add_argument("--stats-interval", type=int, default=32)
    parser.add_argument("--results", type=Path, default=RESULTS)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    for sub in ("logs", "curves"):
        (args.results / sub).mkdir(parents=True, exist_ok=True)
    trials_path = args.results / "trials.jsonl"

    state = load_ladder_state()
    keys = args.algorithms or default_algorithms(state, args.top)
    unknown = [k for k in keys if k not in BY_KEY or k not in state]
    if unknown:
        raise SystemExit(f"unknown or unmeasured algorithms: {', '.join(unknown)}")

    if "report" in args.stages and len(args.stages) == 1:
        write_report(args, state, trials_path)
        return

    if not args.dry_run:
        lock = acquire_lock(args.results)
        atexit.register(lambda: lock.unlink(missing_ok=True))

    started = time.perf_counter()
    for key in keys:
        algo = BY_KEY[key]
        if args.variant == "ladder":
            variants = [state[key]["best"]["variant"]]
        elif args.variant == "both":
            variants = ["eager"] + (["torchcompile"] if algo.torchcompile else [])
        else:
            variants = [args.variant]
        for variant in variants:
            if variant == "torchcompile" and not algo.torchcompile:
                print(f"{key}: no compiled twin, skipping")
                continue
            if "search" in args.stages:
                run_study(algo, variant, args, state, trials_path)
            if "confirm" in args.stages:
                confirm_study(algo, variant, args, state, trials_path)

    print(f"\ntotal {(time.perf_counter() - started) / 3600:.2f} GPU-hours")
    if trials_path.exists():
        write_report(args, state, trials_path)


if __name__ == "__main__":
    main()
