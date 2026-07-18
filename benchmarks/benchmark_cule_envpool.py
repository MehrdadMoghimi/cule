#!/usr/bin/env python3
"""Reproducible Breakout CuLE-versus-EnvPool benchmark driver.

Training trials run in isolated processes and consume the machine-readable
``BENCHMARK_RESULT`` emitted by the PPO/PQN scripts.  Probe trials isolate
environment construction and stepping from model and rollout-buffer memory.
Every trial is appended to JSONL immediately so interrupted sweeps are usable.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import resource
import signal
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "benchmark_results" / "artifacts" / "cule_envpool" / "cule_envpool_breakout_raw.jsonl"
TRAINING_RESULT_RE = re.compile(r"BENCHMARK_RESULT (\{.*\})")
PROBE_RESULT_RE = re.compile(r"PROBE_RESULT (\{.*\})")
RESOURCE_RESULT_RE = re.compile(r"RESOURCE_MAX_RSS_KB ([0-9]+)")


def append_record(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")


def completed_keys(path: Path) -> set[str]:
    if not path.exists():
        return set()
    keys = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if record.get("status") == "ok":
            keys.add(json.dumps({"kind": record["kind"], "params": record["params"]}, sort_keys=True))
    return keys


def trial_key(kind: str, params: dict) -> str:
    return json.dumps({"kind": kind, "params": params}, sort_keys=True)


def timed_command(command: list[str]) -> list[str]:
    return ["/usr/bin/time", "-f", "RESOURCE_MAX_RSS_KB %M", *command]


def run_isolated(command: list[str], timeout: float, result_re: re.Pattern[str]) -> dict:
    started = time.perf_counter()
    process = subprocess.Popen(
        timed_command(command),
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    try:
        output, _ = process.communicate(timeout=timeout)
        result_match = result_re.search(output)
        rss_match = RESOURCE_RESULT_RE.search(output)
        record = {
            "returncode": process.returncode,
            "wall_seconds": time.perf_counter() - started,
        }
        if rss_match:
            record["max_rss_mb"] = int(rss_match.group(1)) / 1024
        if process.returncode == 0 and result_match:
            record.update(status="ok", result=json.loads(result_match.group(1)))
        else:
            record.update(status="error", output_tail=output[-8000:])
        return record
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            output, _ = process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            output, _ = process.communicate()
        return {
            "status": "timeout",
            "wall_seconds": time.perf_counter() - started,
            "output_tail": output[-8000:],
        }


def training_command(params: dict, warmup: int, measure: int) -> list[str]:
    script = {
        "ppo": ROOT / "cleanrl" / "ppo_atari_envpool_torchcompile.py",
        "pqn": ROOT / "cleanrl" / "pqn_atari_envpool.py",
    }[params["algorithm"]]
    command = [
        sys.executable,
        str(script),
        "--benchmark",
        "--env-backend",
        params["backend"],
        "--env-id",
        params["env_id"],
        "--num-envs",
        str(params["num_envs"]),
        "--num-steps",
        str(params["num_steps"]),
        "--num-minibatches",
        "4",
        "--update-epochs",
        "4",
        "--benchmark-warmup-iterations",
        str(warmup),
        "--benchmark-measure-iterations",
        str(measure),
        "--no-anneal-lr",
    ]
    if params["algorithm"] == "ppo" and params["compile"]:
        command.append("--compile")
    if params["backend"] == "cule" and params["algorithm"] == "ppo":
        command += ["--cule-device", "cuda"]
    return command


def run_training(args: argparse.Namespace) -> None:
    done = completed_keys(args.output) if args.resume else set()
    trials = []
    for algorithm in args.algorithms:
        num_steps = args.ppo_num_steps if algorithm == "ppo" else args.pqn_num_steps
        for backend in args.backends:
            for num_envs in args.env_counts:
                for repeat in range(1, args.repeats + 1):
                    trials.append(
                        {
                            "algorithm": algorithm,
                            "backend": backend,
                            "compile": algorithm == "ppo" and args.compile_ppo,
                            "env_id": args.env_id,
                            "num_envs": num_envs,
                            "num_steps": num_steps,
                            "repeat": repeat,
                        }
                    )

    for index, params in enumerate(trials, 1):
        key = trial_key("training", params)
        if key in done:
            print(f"[{index}/{len(trials)}] skip {params}", flush=True)
            continue
        print(f"[{index}/{len(trials)}] training {params}", flush=True)
        command = training_command(params, args.warmup_iterations, args.measure_iterations)
        record = {
            "command": command,
            "kind": "training",
            "params": params,
            "started_at": datetime.now(timezone.utc).isoformat(),
        }
        record.update(run_isolated(command, args.timeout, TRAINING_RESULT_RE))
        append_record(args.output, record)
        if record["status"] == "ok":
            print(
                f"  {record['result']['sps']:,.0f} SPS, "
                f"{record['result']['peak_cuda_memory_mb']:,.0f} MiB CUDA peak",
                flush=True,
            )
        else:
            print(f"  {record['status']}", flush=True)


def run_probe_worker(args: argparse.Namespace) -> None:
    started = time.perf_counter()
    if args.backend == "cule":
        import torch

        sys.path.insert(0, str(ROOT / "cleanrl"))
        from cule_env import make_cule_env

        device = torch.device("cuda")
        torch.cuda.reset_peak_memory_stats(device)
        create_start = time.perf_counter()
        envs = make_cule_env(args.env_id, args.num_envs, device, seed=1)
        torch.cuda.synchronize(device)
        create_seconds = time.perf_counter() - create_start
        reset_start = time.perf_counter()
        envs.reset(seed=1)
        torch.cuda.synchronize(device)
        reset_seconds = time.perf_counter() - reset_start
        actions = torch.zeros(args.num_envs, dtype=torch.int64, device=device)
        for _ in range(args.warmup_steps):
            envs.step(actions)
        torch.cuda.synchronize(device)
        measure_start = time.perf_counter()
        for _ in range(args.measure_steps):
            envs.step(actions)
        torch.cuda.synchronize(device)
        measure_seconds = time.perf_counter() - measure_start
        cuda_memory_mb = torch.cuda.max_memory_allocated(device) / (1024**2)
        env_device = "cuda"
    else:
        import envpool
        import numpy as np

        create_start = time.perf_counter()
        envs = envpool.make(
            args.env_id,
            env_type="gym",
            num_envs=args.num_envs,
            episodic_life=True,
            reward_clip=True,
            seed=1,
        )
        create_seconds = time.perf_counter() - create_start
        reset_start = time.perf_counter()
        envs.reset()
        reset_seconds = time.perf_counter() - reset_start
        actions = np.zeros(args.num_envs, dtype=np.int64)
        for _ in range(args.warmup_steps):
            envs.step(actions)
        measure_start = time.perf_counter()
        for _ in range(args.measure_steps):
            envs.step(actions)
        measure_seconds = time.perf_counter() - measure_start
        cuda_memory_mb = 0.0
        env_device = "cpu"

    result = {
        "backend": args.backend,
        "benchmark": "environment_only",
        "create_seconds": create_seconds,
        "env_device": env_device,
        "env_id": args.env_id,
        "measure_seconds": measure_seconds,
        "measure_steps": args.measure_steps,
        "num_envs": args.num_envs,
        "peak_cuda_memory_mb": cuda_memory_mb,
        "process_seconds": time.perf_counter() - started,
        "reset_seconds": reset_seconds,
        "rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "schema_version": 1,
        "sps": args.num_envs * args.measure_steps / measure_seconds,
        "warmup_steps": args.warmup_steps,
    }
    print(f"PROBE_RESULT {json.dumps(result, sort_keys=True)}", flush=True)
    envs.close()


def run_probes(args: argparse.Namespace) -> None:
    done = completed_keys(args.output) if args.resume else set()
    trials = [
        {"backend": backend, "env_id": args.env_id, "num_envs": num_envs}
        for backend in args.backends
        for num_envs in args.env_counts
    ]
    for index, params in enumerate(trials, 1):
        key = trial_key("probe", params)
        if key in done:
            print(f"[{index}/{len(trials)}] skip {params}", flush=True)
            continue
        print(f"[{index}/{len(trials)}] probe {params}", flush=True)
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "probe-worker",
            "--backend",
            params["backend"],
            "--env-id",
            params["env_id"],
            "--num-envs",
            str(params["num_envs"]),
            "--warmup-steps",
            str(args.warmup_steps),
            "--measure-steps",
            str(args.measure_steps),
        ]
        record = {
            "command": command,
            "kind": "probe",
            "params": params,
            "started_at": datetime.now(timezone.utc).isoformat(),
        }
        record.update(run_isolated(command, args.timeout, PROBE_RESULT_RE))
        append_record(args.output, record)
        if record["status"] == "ok":
            print(
                f"  {record['result']['sps']:,.0f} env SPS, "
                f"{record['result']['rss_mb']:,.0f} MiB RSS",
                flush=True,
            )
        else:
            print(f"  {record['status']}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)

    training = subparsers.add_parser("training", help="run full trainer benchmarks")
    training.add_argument("--algorithms", nargs="+", choices=["ppo", "pqn"], default=["ppo", "pqn"])
    training.add_argument("--backends", nargs="+", choices=["cule", "envpool"], default=["cule", "envpool"])
    training.add_argument("--env-counts", nargs="+", type=int, required=True)
    training.add_argument("--env-id", default="Breakout-v5")
    training.add_argument("--ppo-num-steps", type=int, default=32)
    training.add_argument("--pqn-num-steps", type=int, default=128)
    training.add_argument("--compile-ppo", action=argparse.BooleanOptionalAction, default=True)
    training.add_argument("--warmup-iterations", type=int, default=3)
    training.add_argument("--measure-iterations", type=int, default=10)
    training.add_argument("--repeats", type=int, default=1)
    training.add_argument("--timeout", type=float, default=300)
    training.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    training.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    training.set_defaults(handler=run_training)

    probe = subparsers.add_parser("probe", help="run isolated environment-only probes")
    probe.add_argument("--backends", nargs="+", choices=["cule", "envpool"], default=["cule", "envpool"])
    probe.add_argument("--env-counts", nargs="+", type=int, required=True)
    probe.add_argument("--env-id", default="Breakout-v5")
    probe.add_argument("--warmup-steps", type=int, default=5)
    probe.add_argument("--measure-steps", type=int, default=20)
    probe.add_argument("--timeout", type=float, default=120)
    probe.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    probe.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    probe.set_defaults(handler=run_probes)

    worker = subparsers.add_parser("probe-worker", help=argparse.SUPPRESS)
    worker.add_argument("--backend", choices=["cule", "envpool"], required=True)
    worker.add_argument("--env-id", default="Breakout-v5")
    worker.add_argument("--num-envs", type=int, required=True)
    worker.add_argument("--warmup-steps", type=int, default=5)
    worker.add_argument("--measure-steps", type=int, default=20)
    worker.set_defaults(handler=run_probe_worker)
    return parser


if __name__ == "__main__":
    parsed_args = build_parser().parse_args()
    parsed_args.handler(parsed_args)
