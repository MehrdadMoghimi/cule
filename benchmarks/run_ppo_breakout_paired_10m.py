#!/usr/bin/env python3
"""Run paired 10M-step compiled PPO Breakout experiments on CuLE and EnvPool.

For each selected environment count, both backends receive exactly the same PPO
hyperparameters.  Evaluation pauses training every 1M transitions and runs 64
deterministic, full-game CPU CuLE environments in parallel, giving both training
backends a common evaluation protocol.

The configurations preserve a useful rollout horizon while keeping optimizer
minibatches at 1,024 samples.  The 256-env setup is the healthy 4,096-batch
anchor from the earlier search; 512 and 1,024 use the successful 8,192-batch
regime; 2,048 scales to a 16,384 batch rather than shortening GAE to four steps.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import shlex
import sys
from types import SimpleNamespace
from typing import Any

from tqdm import tqdm

from search_torchcompile_breakout import (
    ROOT,
    append_jsonl,
    build_command,
    read_records,
    run_trial,
    trial_key,
    trial_paths,
    write_summary,
)


DEFAULT_OUTPUT_DIR = (
    ROOT / "benchmark_results" / "artifacts" / "ppo_breakout_paired_10m"
)

# Identical within each CuLE/EnvPool pair.  Batch and minibatch sizes are shown
# explicitly in the names so that artifacts remain self-describing.
CONFIGS: dict[int, dict[str, Any]] = {
    256: {
        "name": "n256_s16_b4096_mb1024_lr25",
        "num_envs": 256,
        "num_steps": 16,
        "num_minibatches": 4,
        "update_epochs": 4,
        "learning_rate": 2.5e-4,
    },
    512: {
        "name": "n512_s16_b8192_mb1024_lr35",
        "num_envs": 512,
        "num_steps": 16,
        "num_minibatches": 8,
        "update_epochs": 4,
        "learning_rate": 3.5e-4,
    },
    1024: {
        "name": "n1024_s8_b8192_mb1024_lr35",
        "num_envs": 1024,
        "num_steps": 8,
        "num_minibatches": 8,
        "update_epochs": 4,
        "learning_rate": 3.5e-4,
    },
    2048: {
        "name": "n2048_s8_b16384_mb1024_lr50",
        "num_envs": 2048,
        "num_steps": 8,
        "num_minibatches": 16,
        "update_epochs": 4,
        "learning_rate": 5.0e-4,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--env-counts",
        nargs="+",
        type=int,
        choices=sorted(CONFIGS),
        default=sorted(CONFIGS),
    )
    parser.add_argument(
        "--backends",
        nargs="+",
        choices=("cule", "envpool"),
        default=["cule", "envpool"],
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[1])
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--timeout-seconds", type=float, default=21_600)
    parser.add_argument(
        "--resume", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--progress", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be positive")
    return args


def build_trials(args: argparse.Namespace) -> list[dict[str, Any]]:
    trials = []
    # Keep the two backends adjacent so each matched pair finishes together.
    for num_envs in args.env_counts:
        for seed in args.seeds:
            for backend in args.backends:
                trials.append(
                    {
                        "algorithm": "ppo",
                        "backend": backend,
                        "config": dict(CONFIGS[num_envs]),
                        "seed": seed,
                        "total_timesteps": 10_000_000,
                    }
                )
    return trials


def launcher_args(args: argparse.Namespace) -> SimpleNamespace:
    output_dir = args.output_dir.resolve()
    return SimpleNamespace(
        python=args.python,
        env_id="Breakout-v5",
        evaluation_interval=1_000_000,
        evaluation_episodes=10,
        evaluation_seed=10_000,
        evaluation_max_episode_steps=18_000,
        progress=args.progress,
        log_dir=output_dir / "logs",
        curve_dir=output_dir / "curves",
    )


def main() -> None:
    args = parse_args()
    trials = build_trials(args)
    launch = launcher_args(args)
    output = args.output_dir.resolve() / "trials.jsonl"
    summary = args.output_dir.resolve() / "summary.csv"

    if args.dry_run:
        print(f"{len(trials)} paired PPO trials")
        for index, trial in enumerate(trials, 1):
            log_path, curve_path = trial_paths(trial, launch)
            command = build_command(trial, curve_path, launch)
            print(
                f"[{index:02d}] {trial['backend']} / "
                f"{trial['config']['num_envs']} envs / seed {trial['seed']}"
            )
            print("     " + shlex.join(command))
        return

    records = read_records(output)
    completed = (
        {trial_key(record) for record in records if record.get("status") == "ok"}
        if args.resume
        else set()
    )
    overall = tqdm(
        total=len(trials),
        desc="Paired PPO Breakout 10M",
        unit="run",
        position=0,
        disable=not args.progress,
    )

    for index, trial in enumerate(trials, 1):
        label = (
            f"ppo/{trial['backend']}/{trial['config']['num_envs']}env/"
            f"seed{trial['seed']}"
        )
        overall.set_postfix_str(label)
        if trial_key(trial) in completed:
            tqdm.write(f"[{index}/{len(trials)}] skip {label}")
            overall.update(1)
            continue

        log_path, curve_path = trial_paths(trial, launch)
        command = build_command(trial, curve_path, launch)
        tqdm.write(f"[{index}/{len(trials)}] run {label}")
        tqdm.write("  " + shlex.join(command))
        record = {
            **trial,
            "command": command,
            "curve_path": str(curve_path),
            "log_path": str(log_path),
            "started_at": datetime.now(timezone.utc).isoformat(),
        }
        record.update(
            run_trial(
                command,
                log_path,
                args.timeout_seconds,
                trial["total_timesteps"],
                label,
                args.progress,
            )
        )
        append_jsonl(output, record)
        records.append(record)
        write_summary(summary, records)

        if record["status"] == "ok":
            evaluation = record["evaluation"]
            tqdm.write(
                f"  final reward={evaluation['reward_mean']:.1f}, "
                f"training={evaluation['training_seconds']:.1f}s"
            )
        else:
            tqdm.write(f"  {record['status']}; see {log_path}")
        overall.update(1)

    overall.close()
    print(f"raw results: {output}")
    print(f"summary: {summary}")


if __name__ == "__main__":
    main()
