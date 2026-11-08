#!/usr/bin/env python3
"""Run fixed-transition-budget experiments and score them by transitions, not seconds.

`sweep_highenv.py` and `sweep_protein.py` both grade a configuration by the
return it reaches after a fixed number of *seconds*.  That is the right question
when the goal is to pick an operating point, and the wrong one when the goal is
to isolate a mechanism, for two reasons:

* a wall-clock score is only valid on an otherwise idle GPU, so no such
  experiment can share the card with anything else, and
* holding seconds fixed lets the variable under test change the amount of data
  the learner sees, which is exactly the confound worth removing.

So this runner fixes `--total-timesteps`, lets each configuration take as long
as it takes, and scores the closing fraction of the *transition* axis.  A run
that is slowed by contention gets the same score as one that is not.

Records go to `benchmark_results/exploration/<experiment>/runs.jsonl`, keyed by
a name that is expected to be unique within the experiment; re-running skips
anything already recorded with an ok status, so an interrupted experiment
resumes by being launched again.
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from breakout_algorithms import BY_KEY, ENV_ID, ON_POLICY, ROOT
from sweep_highenv import PYTHON, args_fields, has, lr_flags, run_trial, trainer_path

BASE = ROOT / "benchmark_results" / "exploration"


def base_command(algorithm: str, variant: str, num_envs: int, total_timesteps: int,
                 seed: int, env_id: str = ENV_ID) -> list[str]:
    """The flags every trainer in this repo needs, and only those it defines."""
    algo = BY_KEY[algorithm]
    path = trainer_path(algo, variant)
    cmd = [PYTHON, str(path), "--env-backend", "cule", "--env-id", env_id,
           "--num-envs", str(num_envs), "--total-timesteps", str(total_timesteps),
           "--seed", str(seed)]
    if has(path, "cule_device"):
        cmd += ["--cule-device", "cuda"]
    if variant == "torchcompile":
        if has(path, "compile"):
            cmd.append("--compile")
        if has(path, "cudagraphs") and algo.has_cudagraphs:
            cmd.append("--cudagraphs")
    for name in algo.budget_flags:
        cmd += [name, str(total_timesteps)]
    return cmd


def score_by_seconds(curve: Path, window: float, closing: float = 0.4) -> dict:
    """Mean return over the closing fraction of the WALL-CLOCK window.

    This is `sweep_highenv.py`'s scoring rule, reproduced so that a run here is
    directly comparable with a row of that sweep's report.  It is only valid on
    an otherwise idle GPU -- unlike `score_by_transitions`, a slower run really
    does score worse, which is the point of the measure and also its fragility.
    """
    if not curve.exists():
        return {"score": None, "peak": None, "points": 0, "reached": 0}
    points = [(float(r["seconds"]), int(r["global_step"]), float(r["mean_return"]))
              for r in csv.DictReader(curve.open()) if r["mean_return"] not in ("", "nan")]
    inside = [p for p in points if p[0] <= window]
    if not inside:
        return {"score": None, "peak": None, "points": 0, "reached": 0}
    tail = [v for t, _, v in inside if t >= (1.0 - closing) * window]
    if not tail:
        tail = [v for _, _, v in inside[-3:]]
    return {"score": round(sum(tail) / len(tail), 3),
            "peak": max(v for _, _, v in inside),
            "points": len(inside),
            "reached": max(s for _, s, _ in inside)}


def score_by_transitions(curve: Path, budget: int, closing: float = 0.25) -> dict:
    """Mean return over the closing fraction of the transition axis.

    Scoring on transitions rather than on wall-clock samples is what makes two
    runs comparable when one of them was competing for the GPU.  `closing` is
    generous because a trainer that logs a 20-episode moving average produces
    few points, and a single point is a noisy estimate of where a run ended up.
    """
    if not curve.exists():
        return {"score": None, "peak": None, "points": 0, "reached": 0}
    points = [(int(r["global_step"]), float(r["mean_return"]))
              for r in csv.DictReader(curve.open()) if r["mean_return"] not in ("", "nan")]
    if not points:
        return {"score": None, "peak": None, "points": 0, "reached": 0}
    reached = max(p[0] for p in points)
    cut = reached * (1.0 - closing)
    tail = [v for s, v in points if s >= cut] or [points[-1][1]]
    return {"score": round(sum(tail) / len(tail), 3),
            "peak": max(v for _, v in points),
            "points": len(points),
            "reached": reached}


# A configuration authored here bugchecked the host on 2026-08-21 (Windows
# VIDEO_TDR_ERROR, 0x116).  Windows' display watchdog (`TdrDelay`, default 2 s)
# resets the adapter when a submission takes too long, and does not care that
# the work is legitimate; the guest died with it.
#
# Finding the right thing to cap took one wrong answer.  The obvious candidates
# -- optimizer updates per vector step, replayed samples per vector step,
# replay-buffer size -- are all IDENTICAL between the cell that crashed
# (64 envs, ratio 128, batch 512) and the H014 cell that ran for hours without
# incident (1024 envs, ratio 8, batch 512): both submit 16 updates and 8,192
# samples per vector step against the same buffer.  So none of them can be the
# discriminator.
#
# What differs is the rate.  Stepping 64 CuLE envs costs about a sixteenth of
# what stepping 1,024 costs, so at 64 envs those same 16 updates are demanded
# sixteen times more often, and the emulator work that used to break up the
# stream no longer does.  Learner work per second of emulator time is
# `ratio * envs / step_cost(envs)`, and CuLE's step cost is very nearly linear
# in `envs` up to 1,024 -- so it reduces to the replay ratio alone.
#
# That is also the physically meaningful reading: replay ratio IS the
# learner-to-emulator work ratio.  At 8 the GPU still returns to the emulator
# regularly; at 128 it is a solid wall of learner kernels with nothing to
# interrupt it.  So the ratio is the cap that matters.
#
# The secondary caps bound the WORK in one submission, not the number of
# launches.  Sixty-four back-to-back batch-32 updates is 2,048 samples and a few
# milliseconds -- launch-bound, not watchdog-bound -- so capping the update
# count tightly (an earlier version used 16, the largest value that happened to
# have been observed running) refuses sound small-batch science for no reason.
# `samples_per_vector_step` is the quantity with units of work; the update cap
# is left loose enough to catch only pathological launch storms.
#
# None of these are correctness limits.  A headless card would run all of it.
MAX_REPLAY_RATIO = 32.0
MAX_UPDATES_PER_VECTOR_STEP = 128.0
MAX_SAMPLES_PER_VECTOR_STEP = 8192.0
MAX_BUFFER_TRANSITIONS = 262_144


def submission_cost(spec: dict) -> dict:
    """How much learner work this spec submits, and how often."""
    flags = {}
    parts = [str(f) for f in spec.get("flags", [])]
    for i in range(len(parts) - 1):
        if parts[i].startswith("--"):
            flags[parts[i][2:].replace("-", "_")] = parts[i + 1]

    def num(key, default=None):
        try:
            return float(flags[key])
        except (KeyError, TypeError, ValueError):
            return default

    envs = float(spec["num_envs"])
    batch = num("batch_size")
    ratio = num("replay_ratio")
    explicit = num("learner_updates_per_vector_step")
    if explicit is not None:
        updates = explicit
        samples = updates * (batch or 32.0)
        ratio = samples / envs
    elif ratio is not None and batch:
        # A ramped ratio is judged by its peak, which is what the watchdog sees.
        ratio = max(ratio, num("replay_ratio_final", ratio) or ratio)
        samples = ratio * envs
        updates = samples / batch
    else:
        # On-policy: one pass of update_epochs x num_minibatches per rollout of
        # num_steps vector steps, so the per-vector-step demand is small by
        # construction and the effective ratio is just update_epochs.
        steps = num("num_steps", 32.0) or 32.0
        mbs = num("num_minibatches", 1.0) or 1.0
        epochs = num("update_epochs", 1.0) or 1.0
        updates = epochs * mbs / steps
        samples = epochs * envs
        ratio = epochs
    return {"replay_ratio_eff": round(ratio, 3),
            "updates_per_vector_step": round(updates, 3),
            "samples_per_vector_step": round(samples, 1),
            "buffer_transitions": int(num("buffer_size", 0.0) or 0.0)}


def guard_spec(spec: dict) -> str | None:
    """Return a refusal reason, or None if the spec is safe to launch."""
    c = submission_cost(spec)
    if c["replay_ratio_eff"] > MAX_REPLAY_RATIO:
        return (f"effective replay ratio {c['replay_ratio_eff']:.1f} exceeds the cap of "
                f"{MAX_REPLAY_RATIO:.0f} -- this is the axis that bugchecked the host")
    if c["updates_per_vector_step"] > MAX_UPDATES_PER_VECTOR_STEP:
        return (f"{c['updates_per_vector_step']:.1f} optimizer updates per vector step "
                f"exceeds the cap of {MAX_UPDATES_PER_VECTOR_STEP:.0f}")
    if c["samples_per_vector_step"] > MAX_SAMPLES_PER_VECTOR_STEP:
        return (f"{c['samples_per_vector_step']:.0f} replayed samples per vector step "
                f"exceeds the cap of {MAX_SAMPLES_PER_VECTOR_STEP:.0f}")
    if c["buffer_transitions"] > MAX_BUFFER_TRANSITIONS:
        return (f"replay buffer of {c['buffer_transitions']:,} transitions exceeds the cap "
                f"of {MAX_BUFFER_TRANSITIONS:,}")
    return None


def load_done(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    out = {}
    for line in path.read_text().splitlines():
        if line.strip():
            record = json.loads(line)
            out[record["name"]] = record
    return out


def run_experiment(experiment: str, specs: list[dict], timeout: float = 3600.0,
                   startup_cap: float = 300.0, stats_interval: int = 32,
                   closing: float = 0.25, dry_run: bool = False,
                   wall_clock: float | None = None) -> list[dict]:
    """Run every spec that has not already produced an ok record.

    A spec is {name, algorithm, variant, num_envs, total_timesteps, flags, seed,
    plus any extra keys worth recording, e.g. the variable under test}.
    """
    out_dir = BASE / experiment
    for sub in ("logs", "curves"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)
    runs_path = out_dir / "runs.jsonl"
    done = load_done(runs_path)

    results = []
    for index, spec in enumerate(specs, 1):
        name = spec["name"]
        if name in done and done[name].get("status") == "ok":
            print(f"[{index}/{len(specs)}] {name:44s} cached score={done[name]['score']}")
            results.append(done[name])
            continue

        refusal = guard_spec(spec)
        if refusal is not None:
            print(f"[{index}/{len(specs)}] {name:44s} REFUSED: {refusal}")
            record = {**{k: v for k, v in spec.items() if k != "flags"},
                      "flags": [str(f) for f in spec.get("flags", [])],
                      "experiment": experiment, "status": "refused",
                      "refusal": refusal, "score": None, "peak": None,
                      "completed": False, **submission_cost(spec)}
            with runs_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, sort_keys=True) + "\n")
            results.append(record)
            continue

        algo = BY_KEY[spec["algorithm"]]
        variant = spec.get("variant", "eager")
        path = trainer_path(algo, variant)
        cmd = base_command(spec["algorithm"], variant, spec["num_envs"],
                           spec["total_timesteps"], spec.get("seed", 1),
                           spec.get("env_id", ENV_ID))
        cmd += lr_flags(path, spec.get("lr_mult", 1.0))
        cmd += [str(part) for part in spec.get("flags", [])]

        bar_scale = spec["num_envs"]
        if algo.family == ON_POLICY:
            bar_scale *= spec.get("num_steps") or args_fields(path).get("num_steps") or 1

        print(f"[{index}/{len(specs)}] {name:44s} ", end="", flush=True)
        if dry_run:
            print("\n    " + " ".join(cmd[1:]))
            continue

        started = time.perf_counter()
        window = wall_clock if wall_clock else timeout
        result = run_trial(cmd, bar_scale, window,
                           out_dir / "logs" / f"{name}.log",
                           out_dir / "curves" / f"{name}.csv",
                           startup_cap, stats_interval)
        curve_path = out_dir / "curves" / f"{name}.csv"
        scored = (score_by_seconds(curve_path, wall_clock) if wall_clock
                  else score_by_transitions(curve_path, spec["total_timesteps"], closing))
        record = {
            **{k: v for k, v in spec.items() if k != "flags"},
            "flags": [str(f) for f in spec.get("flags", [])],
            "experiment": experiment,
            "command": cmd,
            "status": result["status"],
            "returncode": result["returncode"],
            "wall_seconds": round(time.perf_counter() - started, 1),
            "startup_seconds": result["startup_seconds"],
            "train_sps": result["train_sps"],
            "final_step": result["final_step"],
            "scoring": "wall_clock" if wall_clock else "transitions",
            "window": wall_clock,
            "completed": (result["trained_seconds"] >= 0.95 * wall_clock if wall_clock
                          else scored["reached"] >= 0.9 * spec["total_timesteps"]),
            "tail": result["tail"][-1500:],
            **submission_cost(spec),
            **scored,
        }
        with runs_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
        results.append(record)
        print(f"score={('%.2f' % record['score']) if record['score'] is not None else '  -  '} "
              f"peak={record['peak'] or 0:>6.1f} reached={scored['reached']/1e6:5.2f}M "
              f"sps={record['train_sps'] or 0:>6} {record['wall_seconds']:6.0f}s "
              f"[{record['status']}{'' if record['completed'] else ' TRUNCATED'}]", flush=True)
        if record["status"] != "ok":
            print("    " + record["tail"][-300:].replace("\n", "\n    "), flush=True)
    return results
