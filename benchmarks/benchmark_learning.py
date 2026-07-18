#!/usr/bin/env python3
"""Run learning-oriented Breakout trials for A2C, DQN, PPO, and V-trace."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "benchmark_results" / "artifacts" / "learning"


CONFIGS = {
    "a2c": {
        "implementation": "examples/a2c",
        "backend": "CuLE GPU",
        "num_envs": 128,
        "num_steps": 5,
        "learning_rate": 0.00065,
        "notes": "640-transition updates with correctly ordered gradient clipping",
    },
    "vtrace": {
        "implementation": "examples/vtrace",
        "backend": "CuLE GPU",
        "num_envs": 1200,
        "num_steps": 20,
        "num_steps_per_update": 1,
        "num_minibatches": 20,
        "learning_rate": 0.001,
        "notes": "learning-oriented setting previously stronger than the throughput-only setup",
    },
    "ppo": {
        "implementation": "cleanrl/ppo_atari_envpool_torchcompile.py",
        "backend": "EnvPool CPU",
        "num_envs": 128,
        "num_steps": 32,
        "num_minibatches": 4,
        "update_epochs": 4,
        "learning_rate": 0.00025,
        "compile": True,
        "notes": "compiled implementation winner with a 4096-transition learning batch",
    },
    "dqn": {
        "implementation": "cleanrl/dqn_torchcompile.py",
        "backend": "CuLE GPU",
        "num_envs": 256,
        "batch_size": 32,
        "replay_ratio": 1.0,
        "buffer_size": 100000,
        "learning_rate": 0.0001,
        "compile": True,
        "notes": "small minibatches provide enough optimizer updates within a 10M-frame budget",
    },
}


def curve_path(prefix: str, algorithm: str, seed: int, native: bool = False) -> Path:
    suffix = "_test" if native else ""
    return RESULTS / f"{prefix}_{algorithm}_seed{seed}{suffix}.csv"


def command_for(
    algorithm: str,
    seed: int,
    total_timesteps: int,
    evaluation_interval: int,
    evaluation_episodes: int,
) -> tuple[list[str], Path, Path]:
    if algorithm == "a2c":
        output = RESULTS / f"__CURVE_PREFIX___a2c_seed{seed}.csv"
        # Native trainers evaluate at the start of an update.  One extra batch
        # ensures the final requested evaluation boundary is reached.
        t_max = total_timesteps + CONFIGS[algorithm]["num_envs"] * CONFIGS[algorithm]["num_steps"]
        command = [
            sys.executable,
            "a2c_main.py",
            "--env-name",
            "BreakoutNoFrameskip-v4",
            "--use-cuda-env",
            "--num-ales",
            "128",
            "--num-steps",
            "5",
            "--clip-rewards",
            "--episodic-life",
            "--t-max",
            str(t_max),
            "--evaluation-interval",
            str(evaluation_interval),
            "--evaluation-episodes",
            str(evaluation_episodes),
            "--evaluation-deterministic",
            "--output-filename",
            str(output),
            "--no-progress",
            "--seed",
            str(seed),
        ]
        return command, ROOT / "examples" / "a2c", output

    if algorithm == "vtrace":
        output = RESULTS / f"__CURVE_PREFIX___vtrace_seed{seed}.csv"
        t_max = total_timesteps + CONFIGS[algorithm]["num_envs"] * CONFIGS[algorithm]["num_steps_per_update"]
        command = [
            sys.executable,
            "vtrace_main.py",
            "--env-name",
            "BreakoutNoFrameskip-v4",
            "--use-cuda-env",
            "--num-ales",
            "1200",
            "--num-steps",
            "20",
            "--num-steps-per-update",
            "1",
            "--num-minibatches",
            "20",
            "--normalize",
            "--clip-rewards",
            "--episodic-life",
            "--lr",
            "0.001",
            "--t-max",
            str(t_max),
            "--evaluation-interval",
            str(evaluation_interval),
            "--evaluation-episodes",
            str(evaluation_episodes),
            "--evaluation-deterministic",
            "--output-filename",
            str(output),
            "--no-progress",
            "--seed",
            str(seed),
        ]
        return command, ROOT / "examples" / "vtrace", output

    output = RESULTS / f"__CURVE_PREFIX___{algorithm}_seed{seed}.csv"
    if algorithm == "ppo":
        command = [
            sys.executable,
            str(ROOT / "cleanrl" / "ppo_atari_envpool_torchcompile.py"),
            "--env-backend",
            "envpool",
            "--env-id",
            "Breakout-v5",
            "--num-envs",
            "128",
            "--num-steps",
            "32",
            "--num-minibatches",
            "4",
            "--update-epochs",
            "4",
            "--total-timesteps",
            str(total_timesteps),
            "--compile",
            "--evaluation-interval",
            str(evaluation_interval),
            "--evaluation-episodes",
            str(evaluation_episodes),
            "--learning-curve-path",
            str(output),
            "--seed",
            str(seed),
        ]
    else:
        command = [
            sys.executable,
            str(ROOT / "cleanrl" / "dqn_torchcompile.py"),
            "--env-backend",
            "cule",
            "--cule-device",
            "cuda",
            "--env-id",
            "BreakoutNoFrameskip-v4",
            "--num-envs",
            "256",
            "--batch-size",
            "32",
            "--buffer-size",
            "100000",
            "--learning-starts",
            "80000",
            "--replay-ratio",
            "1",
            "--total-timesteps",
            str(total_timesteps),
            "--compile",
            "--evaluation-interval",
            str(evaluation_interval),
            "--evaluation-episodes",
            str(evaluation_episodes),
            "--learning-curve-path",
            str(output),
            "--seed",
            str(seed),
        ]
    return command, ROOT, output


def run_trial(command: list[str], cwd: Path, log_path: Path, timeout: float) -> dict:
    started = time.perf_counter()
    process = subprocess.Popen(
        command,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        start_new_session=True,
    )
    output_tail: list[str] = []
    with log_path.open("w", encoding="utf-8") as log_file:
        try:
            assert process.stdout is not None
            while True:
                line = process.stdout.readline()
                if line:
                    log_file.write(line)
                    log_file.flush()
                    output_tail.append(line)
                    output_tail = output_tail[-200:]
                    if "EVALUATION_RESULT" in line or "training time:" in line:
                        print(line.rstrip(), flush=True)
                elif process.poll() is not None:
                    break
                if time.perf_counter() - started > timeout:
                    raise subprocess.TimeoutExpired(command, timeout)
            return {
                "status": "ok" if process.returncode == 0 else "error",
                "returncode": process.returncode,
                "wall_seconds": time.perf_counter() - started,
                "output_tail": "".join(output_tail)[-12000:],
            }
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            return {
                "status": "timeout",
                "returncode": process.returncode,
                "wall_seconds": time.perf_counter() - started,
                "output_tail": "".join(output_tail)[-12000:],
            }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithms", nargs="+", choices=tuple(CONFIGS), default=tuple(CONFIGS))
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--total-timesteps", type=int, default=10_000_000)
    parser.add_argument("--evaluation-interval", type=int, default=1_000_000)
    parser.add_argument("--evaluation-episodes", type=int, default=10)
    parser.add_argument("--timeout", type=float, default=3600.0)
    parser.add_argument("--prefix", default="learning_breakout")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    metadata_path = RESULTS / f"{args.prefix}_runs.jsonl"
    for algorithm in args.algorithms:
        command, cwd, placeholder_output = command_for(
            algorithm,
            args.seed,
            args.total_timesteps,
            args.evaluation_interval,
            args.evaluation_episodes,
        )
        actual_output = Path(str(placeholder_output).replace("__CURVE_PREFIX__", args.prefix))
        command = [str(actual_output) if token == str(placeholder_output) else token for token in command]
        native_curve = curve_path(args.prefix, algorithm, args.seed, native=algorithm in ("a2c", "vtrace"))
        if native_curve.exists() and not args.overwrite:
            print(f"skip {algorithm}: {native_curve} already exists", flush=True)
            continue

        log_path = RESULTS / f"{args.prefix}_{algorithm}_seed{args.seed}.log"
        print(f"run {algorithm}: {' '.join(command)}", flush=True)
        record = {
            "algorithm": algorithm,
            "command": command,
            "config": CONFIGS[algorithm],
            "curve_path": str(native_curve),
            "cwd": str(cwd),
            "evaluation_episodes": args.evaluation_episodes,
            "evaluation_interval": args.evaluation_interval,
            "log_path": str(log_path),
            "seed": args.seed,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "total_timesteps": args.total_timesteps,
        }
        record.update(run_trial(command, cwd, log_path, args.timeout))
        with metadata_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
        print(f"{algorithm}: {record['status']} in {record['wall_seconds']:.1f}s", flush=True)
        if record["status"] != "ok":
            print(record["output_tail"][-4000:], flush=True)


if __name__ == "__main__":
    main()
