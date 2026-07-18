#!/usr/bin/env python3
"""Run a resumable 2M-transition Breakout search over six trainer/backend pairs.

The search compares compiled PQN, PPO, and Rainbow with CuLE and EnvPool.  It
uses curated, backend-specific configurations instead of a blind Cartesian
product: on-policy rollout length shrinks as environment count grows, and
Rainbow batch size, learning rate, and target refresh cadence change together.

Every trial is isolated in a child process.  The trainer writes a common
full-game CuLE evaluation, making the final reward comparable even though
training collection uses different backends.  Raw records are appended to
JSONL immediately and a ranked CSV is rebuilt after every trial.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
import time
from typing import Any

from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "benchmark_results" / "artifacts" / "torchcompile_breakout_search"
TEN_M_ARTIFACT_DIR = ROOT / "benchmark_results" / "artifacts" / "torchcompile_breakout_10m"
EVALUATION_RE = re.compile(r"EVALUATION_RESULT (\{.*\})")
TRAINING_PROGRESS_RE = re.compile(r"TRAINING_PROGRESS ([0-9]+)")
EVALUATION_START_RE = re.compile(r"EVALUATION_START ([0-9]+)")

TRAINERS = {
    "pqn": ROOT / "cleanrl" / "pqn_atari_envpool_torchcompile.py",
    "ppo": ROOT / "cleanrl" / "ppo_atari_envpool_torchcompile.py",
    "rainbow": ROOT / "cleanrl" / "rainbow_atari_torchcompile.py",
}


def on_policy(
    name: str,
    num_envs: int,
    num_steps: int,
    *,
    num_minibatches: int = 4,
    update_epochs: int = 4,
    learning_rate: float = 2.5e-4,
    q_lambda: float | None = None,
) -> dict[str, Any]:
    config: dict[str, Any] = {
        "name": name,
        "num_envs": num_envs,
        "num_steps": num_steps,
        "num_minibatches": num_minibatches,
        "update_epochs": update_epochs,
        "learning_rate": learning_rate,
    }
    if q_lambda is not None:
        config["q_lambda"] = q_lambda
    return config


def rainbow(
    name: str,
    num_envs: int,
    batch_size: int,
    *,
    replay_ratio: float = 1.0,
    learning_starts: int = 80_000,
) -> dict[str, Any]:
    # Keep roughly 262k replay samples between target refreshes.  With replay
    # ratio 1 this also keeps target cadence stable as batch size changes.
    target_network_frequency = max(1, round(262_144 / batch_size))
    # Conservative square-root batch scaling around Rainbow's 6.25e-5 rate.
    learning_rate = 6.25e-5 * math.sqrt(batch_size / 128)
    return {
        "name": name,
        "num_envs": num_envs,
        "batch_size": batch_size,
        "buffer_size": 100_000,
        "learning_starts": learning_starts,
        "learning_rate": learning_rate,
        "replay_ratio": replay_ratio,
        "target_network_frequency": target_network_frequency,
    }


# Four trials per pair (24 with one seed).  The first entry in each list is the
# conservative anchor; --quick selects only the first two entries.
SEARCH_SPACE: dict[tuple[str, str], list[dict[str, Any]]] = {
    ("ppo", "cule"): [
        on_policy("c128_s32_b4096", 128, 32),
        on_policy("c256_s16_b4096", 256, 16),
        on_policy("c512_s8_b4096", 512, 8),
        on_policy("c1024_s8_b8192_lr35", 1024, 8, num_minibatches=8, learning_rate=3.5e-4),
    ],
    ("ppo", "envpool"): [
        on_policy("e128_s32_b4096", 128, 32),
        on_policy("e64_s64_b4096", 64, 64),
        on_policy("e256_s16_b4096", 256, 16),
        on_policy("e512_s8_b4096", 512, 8),
    ],
    ("pqn", "cule"): [
        on_policy("c128_s64_b8192", 128, 64, q_lambda=0.65),
        on_policy("c256_s32_b8192", 256, 32, q_lambda=0.65),
        on_policy("c512_s16_b8192_lam80", 512, 16, q_lambda=0.80),
        on_policy("c640_s16_b10240_lam80", 640, 16, num_minibatches=5, q_lambda=0.80),
    ],
    ("pqn", "envpool"): [
        on_policy("e128_s32_b4096", 128, 32, q_lambda=0.65),
        on_policy("e32_s128_b4096", 32, 128, q_lambda=0.65),
        on_policy("e64_s64_b4096", 64, 64, q_lambda=0.65),
        on_policy("e256_s16_b4096_lam80", 256, 16, q_lambda=0.80),
    ],
    ("rainbow", "cule"): [
        rainbow("c128_b128", 128, 128),
        rainbow("c256_b128", 256, 128),
        rainbow("c256_b256", 256, 256),
        rainbow("c512_b512", 512, 512),
    ],
    ("rainbow", "envpool"): [
        rainbow("e128_b128", 128, 128),
        rainbow("e32_b64", 32, 64),
        rainbow("e64_b64", 64, 64),
        rainbow("e256_b256", 256, 256),
    ],
}

# Selected from the completed one-seed 2M search.  Rainbow/CuLE was a reward
# tie across all four candidates, so the fastest tied configuration is used.
# Rainbow remains provisional because none of its 2M candidates learned beyond
# near-random evaluation performance.
SELECTED_10M_CONFIGS = {
    ("ppo", "cule"): "c1024_s8_b8192_lr35",
    ("ppo", "envpool"): "e128_s32_b4096",
    ("pqn", "cule"): "c256_s32_b8192",
    ("pqn", "envpool"): "e64_s64_b4096",
    ("rainbow", "cule"): "c512_b512",
    ("rainbow", "envpool"): "e256_b256",
}


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")


def read_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSON in {path}:{line_number}: {error}") from error
    return records


def trial_key(trial: dict[str, Any]) -> str:
    identity = {
        "algorithm": trial["algorithm"],
        "backend": trial["backend"],
        "config": trial["config"],
        "seed": trial["seed"],
        "total_timesteps": trial["total_timesteps"],
    }
    return json.dumps(identity, sort_keys=True)


def build_trials(args: argparse.Namespace) -> list[dict[str, Any]]:
    trials = []
    for algorithm in args.algorithms:
        for backend in args.backends:
            configs = SEARCH_SPACE[(algorithm, backend)]
            if args.selected_10m:
                selected_name = SELECTED_10M_CONFIGS[(algorithm, backend)]
                configs = [config for config in configs if config["name"] == selected_name]
            elif args.quick:
                configs = configs[:2]
            for config in configs:
                if args.configs and config["name"] not in args.configs:
                    continue
                for seed in args.seeds:
                    trials.append(
                        {
                            "algorithm": algorithm,
                            "backend": backend,
                            "config": config,
                            "seed": seed,
                            "total_timesteps": args.total_timesteps,
                        }
                    )
    if args.limit is not None:
        trials = trials[: args.limit]
    return trials


def add_param(command: list[str], key: str, value: Any) -> None:
    flag = "--" + key.replace("_", "-")
    if isinstance(value, bool):
        command.append(flag if value else "--no-" + key.replace("_", "-"))
    else:
        command.extend((flag, str(value)))


def trial_paths(trial: dict[str, Any], args: argparse.Namespace) -> tuple[Path, Path]:
    stem = (
        f"{trial['algorithm']}__{trial['backend']}__{trial['config']['name']}"
        f"__seed{trial['seed']}__t{trial['total_timesteps']}"
    )
    return args.log_dir / f"{stem}.log", args.curve_dir / f"{stem}.csv"


def build_command(trial: dict[str, Any], curve_path: Path, args: argparse.Namespace) -> list[str]:
    algorithm = trial["algorithm"]
    backend = trial["backend"]
    config = trial["config"]
    exp_name = f"tc_search_{backend}_{config['name']}"
    command = [
        args.python,
        str(TRAINERS[algorithm]),
        "--compile",
        "--env-backend",
        backend,
        "--env-id",
        args.env_id,
        "--total-timesteps",
        str(trial["total_timesteps"]),
        "--seed",
        str(trial["seed"]),
        "--exp-name",
        exp_name,
        "--evaluation-interval",
        str(args.evaluation_interval),
        "--evaluation-episodes",
        str(args.evaluation_episodes),
        "--evaluation-seed",
        str(args.evaluation_seed),
        "--evaluation-max-episode-steps",
        str(args.evaluation_max_episode_steps),
        "--skip-initial-evaluation",
        "--learning-curve-path",
        str(curve_path),
    ]
    if args.progress:
        command.append("--emit-progress")
    if backend == "cule":
        command.extend(("--cule-device", "cuda"))
    for key, value in config.items():
        if key != "name":
            add_param(command, key, value)
    return command


def parse_last_evaluation(log_path: Path) -> dict[str, Any] | None:
    evaluations = EVALUATION_RE.findall(log_path.read_text(encoding="utf-8", errors="replace"))
    return json.loads(evaluations[-1]) if evaluations else None


def run_trial(
    command: list[str],
    log_path: Path,
    timeout_seconds: float,
    total_timesteps: int,
    label: str,
    progress_enabled: bool,
) -> dict[str, Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    progress = tqdm(
        total=total_timesteps,
        desc=label,
        unit="step",
        unit_scale=True,
        position=1,
        leave=False,
        disable=not progress_enabled,
    )
    progress.set_postfix_str("starting/compiling")
    read_offset = 0

    def refresh_progress() -> None:
        nonlocal read_offset
        with log_path.open("r", encoding="utf-8", errors="replace") as stream:
            stream.seek(read_offset)
            output = stream.read()
            read_offset = stream.tell()
        training_steps = TRAINING_PROGRESS_RE.findall(output)
        if training_steps:
            frames = min(int(training_steps[-1]), total_timesteps)
            progress.update(max(0, frames - progress.n))
            progress.set_postfix_str("training")
        evaluation_starts = EVALUATION_START_RE.findall(output)
        if evaluation_starts:
            frames = min(int(evaluation_starts[-1]), total_timesteps)
            progress.update(max(0, frames - progress.n))
            progress.set_postfix_str(f"evaluating at {frames / 1_000_000:.1f}M")

    timed_out = False
    with log_path.open("w", encoding="utf-8") as log_stream:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=log_stream,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        while process.poll() is None:
            refresh_progress()
            if time.perf_counter() - started >= timeout_seconds:
                timed_out = True
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                break
            time.sleep(0.25)
        returncode = process.returncode
        refresh_progress()

    if returncode == 0:
        progress.update(max(0, total_timesteps - progress.n))
    progress.close()
    if timed_out:
        return {
            "status": "timeout",
            "returncode": returncode,
            "wall_seconds": time.perf_counter() - started,
        }

    evaluation = parse_last_evaluation(log_path)
    if returncode == 0 and evaluation is not None:
        return {
            "status": "ok",
            "returncode": returncode,
            "wall_seconds": time.perf_counter() - started,
            "evaluation": evaluation,
        }
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    return {
        "status": "error",
        "returncode": returncode,
        "wall_seconds": time.perf_counter() - started,
        "output_tail": log_text[-8_000:],
    }


def write_summary(path: Path, records: list[dict[str, Any]]) -> None:
    latest_by_key: dict[str, dict[str, Any]] = {}
    for record in records:
        latest_by_key[trial_key(record)] = record
    rows = []
    for record in latest_by_key.values():
        evaluation = record.get("evaluation", {})
        frames = evaluation.get("frames")
        training_seconds = evaluation.get("training_seconds")
        train_sps = None
        if frames is not None and training_seconds:
            train_sps = frames / training_seconds
        rows.append(
            {
                "algorithm": record["algorithm"],
                "backend": record["backend"],
                "config": record["config"]["name"],
                "seed": record["seed"],
                "status": record["status"],
                "reward_mean": evaluation.get("reward_mean"),
                "reward_median": evaluation.get("reward_median"),
                "reward_std": evaluation.get("reward_std"),
                "frames": frames,
                "training_seconds": training_seconds,
                "train_sps": train_sps,
                "wall_seconds": record.get("wall_seconds"),
                "num_envs": record["config"].get("num_envs"),
                "num_steps": record["config"].get("num_steps"),
                "batch_size": record["config"].get("batch_size"),
                "learning_rate": record["config"].get("learning_rate"),
                "replay_ratio": record["config"].get("replay_ratio"),
                "target_network_frequency": record["config"].get("target_network_frequency"),
                "log_path": record.get("log_path"),
                "curve_path": record.get("curve_path"),
                "command": shlex.join(record.get("command", [])),
            }
        )
    rows.sort(
        key=lambda row: (
            row["algorithm"],
            row["backend"],
            row["status"] != "ok",
            -(row["reward_mean"] if row["reward_mean"] is not None else float("-inf")),
            row["training_seconds"] if row["training_seconds"] is not None else float("inf"),
        )
    )
    pair_rank: dict[tuple[str, str], int] = {}
    for row in rows:
        pair = (row["algorithm"], row["backend"])
        if row["status"] == "ok":
            pair_rank[pair] = pair_rank.get(pair, 0) + 1
            row["rank_within_pair"] = pair_rank[pair]
        else:
            row["rank_within_pair"] = ""

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "rank_within_pair",
        "algorithm",
        "backend",
        "config",
        "seed",
        "status",
        "reward_mean",
        "reward_median",
        "reward_std",
        "frames",
        "training_seconds",
        "train_sps",
        "wall_seconds",
        "num_envs",
        "num_steps",
        "batch_size",
        "learning_rate",
        "replay_ratio",
        "target_network_frequency",
        "log_path",
        "curve_path",
        "command",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithms", nargs="+", choices=sorted(TRAINERS), default=sorted(TRAINERS))
    parser.add_argument("--backends", nargs="+", choices=("cule", "envpool"), default=["cule", "envpool"])
    parser.add_argument("--configs", nargs="+", help="only run exact configuration names")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1])
    parser.add_argument("--total-timesteps", type=int)
    parser.add_argument("--evaluation-interval", type=int)
    parser.add_argument("--env-id", default="Breakout-v5")
    parser.add_argument("--evaluation-episodes", type=int, default=5)
    parser.add_argument("--evaluation-seed", type=int, default=10_000)
    parser.add_argument("--evaluation-max-episode-steps", type=int, default=18_000)
    parser.add_argument("--timeout-seconds", type=float, default=3_600)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--log-dir", type=Path)
    parser.add_argument("--curve-dir", type=Path)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--quick", action="store_true", help="run two configurations per pair (12 trials per seed)")
    parser.add_argument(
        "--selected-10m",
        action="store_true",
        help="run the six selected 2M winners for 10M transitions with 1M evaluations",
    )
    parser.add_argument("--limit", type=int, help="limit trials after filtering; useful for smoke tests")
    parser.add_argument("--dry-run", action="store_true", help="print commands without starting training")
    args = parser.parse_args()
    if args.selected_10m and (args.quick or args.configs):
        parser.error("--selected-10m cannot be combined with --quick or --configs")
    if args.total_timesteps is None:
        args.total_timesteps = 10_000_000 if args.selected_10m else 2_000_000
    if args.evaluation_interval is None:
        args.evaluation_interval = 1_000_000 if args.selected_10m else args.total_timesteps
    artifact_dir = TEN_M_ARTIFACT_DIR if args.selected_10m else ARTIFACT_DIR
    args.output = args.output or artifact_dir / "trials.jsonl"
    args.summary = args.summary or artifact_dir / "summary.csv"
    args.log_dir = args.log_dir or artifact_dir / "logs"
    args.curve_dir = args.curve_dir or artifact_dir / "curves"
    if args.total_timesteps < 1:
        parser.error("--total-timesteps must be positive")
    if args.evaluation_interval < 1:
        parser.error("--evaluation-interval must be positive")
    if args.evaluation_episodes < 1:
        parser.error("--evaluation-episodes must be positive")
    if args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be positive")
    return args


def main() -> None:
    args = parse_args()
    trials = build_trials(args)
    if not trials:
        raise SystemExit("no trials matched the requested filters")

    if args.dry_run:
        print(f"{len(trials)} trials")
        for index, trial in enumerate(trials, 1):
            log_path, curve_path = trial_paths(trial, args)
            command = build_command(trial, curve_path, args)
            print(f"[{index:02d}] {trial['algorithm']}/{trial['backend']}/{trial['config']['name']}")
            print("     " + shlex.join(command))
        return

    records = read_records(args.output)
    completed = {trial_key(record) for record in records if record.get("status") == "ok"} if args.resume else set()
    overall = tqdm(
        total=len(trials),
        desc="Breakout search",
        unit="trial",
        position=0,
        disable=not args.progress,
    )
    for index, trial in enumerate(trials, 1):
        key = trial_key(trial)
        label = f"{trial['algorithm']}/{trial['backend']}/{trial['config']['name']}/seed{trial['seed']}"
        overall.set_postfix_str(label)
        if key in completed:
            tqdm.write(f"[{index}/{len(trials)}] skip {label}")
            overall.update(1)
            continue
        log_path, curve_path = trial_paths(trial, args)
        command = build_command(trial, curve_path, args)
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
        append_jsonl(args.output, record)
        records.append(record)
        write_summary(args.summary, records)
        if record["status"] == "ok":
            evaluation = record["evaluation"]
            tqdm.write(
                f"  reward={evaluation['reward_mean']:.1f}, "
                f"training={evaluation['training_seconds']:.1f}s, "
                f"frames={evaluation['frames']:,}"
            )
        else:
            tqdm.write(f"  {record['status']}; see {log_path}")
        overall.update(1)
    overall.close()

    print(f"raw results: {args.output}")
    print(f"ranked summary: {args.summary}")


if __name__ == "__main__":
    main()
