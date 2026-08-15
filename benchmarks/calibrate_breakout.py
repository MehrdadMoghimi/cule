#!/usr/bin/env python3
"""Throughput calibration for the Breakout learning benchmark.

The benchmark has one free knob -- `--num-envs` -- and it should pick it from
measurement rather than assertion.  This script measures the two cost terms that
decide a run's wall clock, separately, so the run time can be predicted for any
environment count:

    seconds(S, N) = S * t_collect(N)  +  S * (ratio / batch) * t_update

`t_collect(N)` is the per-transition cost of stepping N CuLE environments and
running the behaviour policy, measured with the learner switched off (a
`--learning-starts` above the benchmark window).  `t_update` is the cost of one
gradient step at the published batch size, recovered from a second probe that
runs the learner at a known replay ratio and subtracts the collect term.

The useful consequence, visible directly in the model, is that **the learn term
does not contain N**.  Environment count buys throughput only on the collect
term, so it can be chosen for speed without touching the gradient cadence that
decides whether the algorithm learns at all.

Every hyperparameter is passed explicitly rather than inherited, because the
`*_torchcompile.py` twins ship different defaults from their eager counterparts
(256 environments and batch 512 instead of 1 and 32).  A probe therefore measures
the same configuration whichever file runs it.

Results append to JSONL and the script is resumable: probes already recorded as
successful are skipped unless `--overwrite` is passed.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from breakout_algorithms import (
    ALGORITHMS, BY_KEY, ENV_ID, OFF_POLICY, ON_POLICY, STREAMING, ROOT,
)

RESULTS = ROOT / "benchmark_results" / "artifacts" / "breakout"
BENCHMARK_RE = re.compile(r"BENCHMARK_RESULT (\{.*\})")
PYTHON = os.environ.get("CULE_PYTHON", "/home/mehrdad96/anaconda3/envs/cule312/bin/python")

# CuLE is latency-bound below ~128 environments and the repo's scaling study puts
# the CuLE/EnvPool crossover near 512, so the sweep brackets the useful range.
ENV_SWEEP = (64, 128, 256, 512, 1024)
# On-policy environment count is bounded by the *budget*, not by throughput: the
# rollout batch is num_envs * num_steps, so 1024 environments at the shipped 128
# steps would spend 13% of a 1M-transition budget on a single rollout and leave
# seven policy updates for the whole run.  Sweep the range that keeps hundreds.
ON_POLICY_SWEEP = (8, 16, 32, 64, 128)
# Streaming applies one bounded update per vector step, so N environments divide
# the number of parameter updates by N.  The published algorithm is N=1.
STREAM_SWEEP = (1, 2, 4, 8, 16)

# A calibration probe keeps the replay buffer small: the sampling cost of a
# frame-efficient buffer does not depend on its capacity, and a 1M buffer would
# spend the whole probe filling RAM.
PROBE_BUFFER = 50_000

# A `learn` probe must be sized by the work it does, not by the transitions it
# collects.  At replay ratio r the trainer performs r*N/batch gradient steps per
# vector step, so a transition-sized window would run tens of thousands of
# updates for a heavy learner and time out.  Sizing by update count keeps every
# probe to the same amount of learner work regardless of the algorithm.
TARGET_UPDATES = 800
# Transitions a probe should cover when it is sized by data rather than by
# gradient work.
TARGET_TRANSITIONS = 120_000


def measure_iterations(num_envs: int, updates_per_vector_step: float | None,
                       transitions_per_iteration: int | None = None) -> tuple[int, int]:
    """Warmup/measure iterations for one probe.

    An "iteration" does not mean the same thing to every trainer, and getting
    that wrong is expensive.  A replay trainer counts single vector steps, but
    the on-policy files count *full training iterations* -- a whole
    `num_envs * num_steps` rollout plus its update -- so a window sized in
    vector steps would run 250 rollouts of 16,384 transitions and cover four
    times the entire benchmark budget in a single probe.  Pass
    `transitions_per_iteration` for those trainers so the window is sized by the
    data it actually consumes.
    """
    if updates_per_vector_step:
        measure = max(6, min(200, round(TARGET_UPDATES / updates_per_vector_step)))
    elif transitions_per_iteration:
        measure = max(3, min(60, round(TARGET_TRANSITIONS / transitions_per_iteration)))
    else:
        # The cap has to be generous here: a streaming trainer at one
        # environment performs one update per vector step, so a 200-step window
        # would time 200 updates and be dominated by process start-up.
        measure = max(20, min(2000, TARGET_TRANSITIONS // max(num_envs, 1)))
    return max(2, measure // 4), measure


# Below this many environments a GPU CuLE dispatch is pure launch overhead, so
# the CPU backend wins; this is the repo's own `resolve_cule_device` threshold.
CULE_CPU_THRESHOLD = 32


def cule_device_for(num_envs: int) -> str:
    return "cuda" if num_envs >= CULE_CPU_THRESHOLD else "cpu"


def variant_flags(algo, variant: str) -> list[str]:
    if variant == "eager":
        return []
    flags = ["--compile"]
    if algo.has_cudagraphs:
        flags.append("--cudagraphs")
    return flags


def on_policy_minibatch_flags(algo, num_envs: int) -> list[str]:
    """Hold the minibatch size at the trainer's default while envs scale.

    Raising `--num-envs` multiplies the rollout batch.  Left alone, the shipped
    `--num-minibatches` would multiply the minibatch size with it and quietly
    change what the learning rate means.  Growing the minibatch *count* instead
    keeps minibatch size, epoch count, and therefore gradient work per collected
    transition exactly as shipped.
    """
    if not algo.minibatch_size:
        return []
    minibatches = max(1, round(num_envs * algo.num_steps / algo.minibatch_size))
    flags = ["--num-minibatches", str(minibatches)]
    if algo.update_epochs is not None:
        flags += ["--update-epochs", str(algo.update_epochs)]
    return flags


def probe_variant(algo, variant: str, probe: str) -> str:
    """Which file actually runs a probe.

    A `collect` probe switches the learner off, and the learner is the only thing
    the twins compile, so running it through the eager file measures the same
    quantity while skipping a 60-120 s `torch.compile` warmup per probe.  The
    `learn` and `train` probes, where the compiled region is exactly what is
    being measured, always use the requested variant.
    """
    if probe == "collect":
        return "eager" if algo.has_benchmark_eager else variant
    return variant


def build_command(algo, variant: str, probe: str, num_envs: int,
                  replay_ratio: float | None) -> list[str]:
    variant = probe_variant(algo, variant, probe)
    path = algo.eager_path if variant == "eager" else algo.torchcompile_path
    updates_per_vector_step = None
    if probe == "learn" and replay_ratio and algo.batch_size:
        updates_per_vector_step = replay_ratio * num_envs / algo.batch_size
    # On-policy trainers count whole rollouts as iterations, not vector steps.
    per_iteration = num_envs * algo.num_steps if algo.family == ON_POLICY else None
    warmup, measure = measure_iterations(num_envs, updates_per_vector_step, per_iteration)

    cmd = [
        PYTHON, str(path),
        "--env-backend", "cule",
        "--env-id", ENV_ID,
        "--num-envs", str(num_envs),
        "--seed", "1",
        "--benchmark",
        "--benchmark-warmup-iterations", str(warmup),
        "--benchmark-measure-iterations", str(measure),
    ]
    if algo.has_cule_device:
        cmd += ["--cule-device", cule_device_for(num_envs)]
    if algo.has_batch_size and algo.batch_size:
        cmd += ["--batch-size", str(algo.batch_size)]
    cmd += variant_flags(algo, variant)

    if algo.family == OFF_POLICY:
        if algo.has_buffer_size:
            cmd += ["--buffer-size", str(PROBE_BUFFER)]
        if probe == "collect":
            cmd += ["--learning-starts", str(10 ** 9)]
        else:
            cmd += ["--learning-starts", "0"]
            if algo.has_replay_ratio and replay_ratio is not None:
                cmd += ["--replay-ratio", str(replay_ratio)]
    elif algo.family == ON_POLICY:
        cmd += ["--num-steps", str(algo.num_steps)]
        cmd += on_policy_minibatch_flags(algo, num_envs)
    cmd += algo.extra
    return cmd


def run_probe(cmd: list[str], timeout: float) -> dict:
    started = time.perf_counter()
    try:
        proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                              timeout=timeout, start_new_session=True)
    except subprocess.TimeoutExpired as exc:
        return {"status": "timeout", "wall_seconds": time.perf_counter() - started,
                "tail": (exc.stdout or "")[-2000:]}
    out = proc.stdout + proc.stderr
    match = BENCHMARK_RE.search(out)
    if proc.returncode != 0 or match is None:
        return {"status": "error", "returncode": proc.returncode,
                "wall_seconds": time.perf_counter() - started, "tail": out[-3000:]}
    return {"status": "ok", "wall_seconds": time.perf_counter() - started,
            "result": json.loads(match.group(1))}


def probe_plan(algo, variants: tuple[str, ...]) -> list[dict]:
    plan = []
    for variant in variants:
        if variant == "torchcompile" and not algo.torchcompile:
            continue
        if variant == "eager" and not algo.has_benchmark_eager:
            continue
        if variant == "torchcompile" and not algo.has_benchmark:
            continue
        if algo.env_sweep:
            sweep = algo.env_sweep
        elif algo.pinned_envs:
            sweep = (algo.pinned_envs,)
        elif algo.family == STREAMING:
            sweep = STREAM_SWEEP
        elif algo.family == ON_POLICY:
            sweep = ON_POLICY_SWEEP
        else:
            sweep = ENV_SWEEP
        if algo.family == OFF_POLICY:
            for n in sweep:
                plan.append({"probe": "collect", "num_envs": n,
                             "replay_ratio": None, "variant": variant})
            if not algo.has_replay_ratio or algo.use_shipped_cadence:
                # Cadence is fixed by the trainer (endpoint_ddqn) or is already
                # the published configuration (BTR, R2D2), so one learner probe
                # per swept count *is* the measurement -- no ratio to vary.
                for n in sweep:
                    plan.append({"probe": "learn", "num_envs": n,
                                 "replay_ratio": None, "variant": variant})
                continue
            # Two learner probes at one environment count.  Two points suffice:
            # the model is linear in the number of updates, and the second point
            # exists to check that linearity rather than to fit it.
            anchor = algo.pinned_envs or 256
            for ratio in (algo.paper_replay_ratio, algo.paper_replay_ratio / 4.0):
                plan.append({"probe": "learn", "num_envs": anchor,
                             "replay_ratio": ratio, "variant": variant})
        else:
            for n in sweep:
                plan.append({"probe": "train", "num_envs": n,
                             "replay_ratio": None, "variant": variant})
    return plan


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--algorithms", nargs="+", default=[a.key for a in ALGORITHMS])
    parser.add_argument("--variants", nargs="+", default=["torchcompile"],
                        choices=["eager", "torchcompile"])
    parser.add_argument("--output", type=Path, default=RESULTS / "calibration.jsonl")
    parser.add_argument("--timeout", type=float, default=1200.0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    done: set[tuple] = set()
    if args.output.exists() and not args.overwrite:
        for line in args.output.read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                if rec.get("status") == "ok":
                    done.add((rec["algorithm"], rec["variant"], rec["probe"],
                              rec["num_envs"], rec["replay_ratio"]))

    plan = [(BY_KEY[key], item)
            for key in args.algorithms
            for item in probe_plan(BY_KEY[key], tuple(args.variants))]

    print(f"{len(plan)} probes planned, {len(done)} already done", flush=True)
    for index, (algo, item) in enumerate(plan, start=1):
        signature = (algo.key, item["variant"], item["probe"], item["num_envs"],
                     item["replay_ratio"])
        if signature in done:
            continue
        cmd = build_command(algo, item["variant"], item["probe"],
                            item["num_envs"], item["replay_ratio"])
        print(f"[{index}/{len(plan)}] {algo.key}/{item['variant']}/{item['probe']}"
              f"/n{item['num_envs']}/r{item['replay_ratio']}", flush=True)
        record = {
            "algorithm": algo.key, "family": algo.family, "variant": item["variant"],
            "probe": item["probe"], "num_envs": item["num_envs"],
            "replay_ratio": item["replay_ratio"], "batch_size": algo.batch_size,
            "ran_variant": probe_variant(algo, item["variant"], item["probe"]),
            "command": cmd, "started_at": datetime.now(timezone.utc).isoformat(),
        }
        record.update(run_probe(cmd, args.timeout))
        with args.output.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
        if record["status"] == "ok":
            r = record["result"]
            # On-policy trainers report no `ups`: their gradient work is welded
            # to the rollout, so there is no separate update count to report.
            ups = r.get("ups")
            print(f"    sps={r['sps']:.0f} "
                  f"ups={f'{ups:.1f}' if ups is not None else '-'} "
                  f"mem={r.get('peak_cuda_memory_mb', 0):.0f}MB", flush=True)
        else:
            print(f"    {record['status']}: {record.get('tail', '')[-300:]}", flush=True)


if __name__ == "__main__":
    main()
