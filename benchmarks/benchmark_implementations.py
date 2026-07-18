#!/usr/bin/env python3
"""Benchmark native CuLE examples, CleanRL, and torchcompile-oriented trainers.

All trials use Breakout and run in isolated child processes.  The driver stores
one JSON object per trial immediately, so an interrupted run can be resumed.
The timed region is emitted by each trainer and covers environment stepping,
policy inference, return/loss computation, backward passes, and optimizer work.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "benchmark_results" / "artifacts" / "implementation" / "implementation_breakout_raw.jsonl"
RESULT_RE = re.compile(r"(?:THROUGHPUT_RESULT|BENCHMARK_RESULT) (\{.*\})")
RESOURCE_RE = re.compile(r"RESOURCE_MAX_RSS_KB ([0-9]+)")


def append_record(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")


def trial_key(params: dict) -> str:
    return json.dumps(params, sort_keys=True)


def completed_keys(path: Path) -> set[str]:
    if not path.exists():
        return set()
    keys = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if record.get("status") == "ok":
            keys.add(trial_key(record["params"]))
    return keys


def run_isolated(command: list[str], cwd: Path, timeout: float) -> dict:
    timed_command = ["/usr/bin/time", "-f", "RESOURCE_MAX_RSS_KB %M", *command]
    started = time.perf_counter()
    process = subprocess.Popen(
        timed_command,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    try:
        output, _ = process.communicate(timeout=timeout)
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

    result_match = RESULT_RE.search(output)
    rss_match = RESOURCE_RE.search(output)
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


def native_on_policy_command(algorithm: str, num_envs: int, compiled: bool = False) -> tuple[list[str], Path]:
    command = [
        sys.executable,
        f"{algorithm}_main.py",
        "--env-name",
        "BreakoutNoFrameskip-v4",
        "--use-cuda-env",
        "--num-ales",
        str(num_envs),
        "--num-steps",
        "4" if algorithm == "ppo" else "5",
        "--clip-rewards",
        "--episodic-life",
        "--throughput-benchmark",
        "--benchmark-warmup-iterations",
        "3" if algorithm == "ppo" else "10",
        "--benchmark-measure-iterations",
        "10" if algorithm == "ppo" else "30",
        "--no-progress",
        "--seed",
        "1",
    ]
    if algorithm == "ppo":
        command += ["--ppo-epoch", "4", "--use-adam"]
    if compiled:
        command.append("--torch-compile")
    return command, ROOT / "examples" / algorithm


def native_vtrace_command(num_envs: int) -> tuple[list[str], Path]:
    command = [
        sys.executable,
        "vtrace_main.py",
        "--env-name",
        "BreakoutNoFrameskip-v4",
        "--use-cuda-env",
        "--num-ales",
        str(num_envs),
        "--num-steps",
        "5",
        "--num-minibatches",
        str(max(1, num_envs // 64)),
        "--num-steps-per-update",
        "1",
        "--clip-rewards",
        "--episodic-life",
        "--benchmark",
        "--throughput-benchmark",
        "--benchmark-warmup-iterations",
        "10",
        "--benchmark-measure-iterations",
        "30",
        "--no-progress",
        "--seed",
        "1",
    ]
    return command, ROOT / "examples" / "vtrace"


def native_dqn_command(num_envs: int = 256) -> tuple[list[str], Path]:
    command = [
        sys.executable,
        "dqn_main.py",
        "--env-name",
        "BreakoutNoFrameskip-v4",
        "--use-cuda-env",
        "--num-ales",
        str(num_envs),
        "--batch-size",
        "32",
        "--replay-frequency",
        "4",
        "--multi-step",
        "1",
        "--memory-capacity",
        "100000",
        "--learn-start",
        str(num_envs),
        "--reward-clip",
        "--throughput-benchmark",
        "--benchmark-warmup-iterations",
        "10",
        "--benchmark-measure-iterations",
        "30",
        "--no-progress",
        "--seed",
        "1",
    ]
    return command, ROOT / "examples" / "dqn"


def cleanrl_ppo_command(script: str, compiled: bool, backend: str = "cule") -> tuple[list[str], Path]:
    env_id = "Breakout-v5" if backend == "envpool" else "BreakoutNoFrameskip-v4"
    command = [
        sys.executable,
        str(ROOT / "cleanrl" / script),
        "--benchmark",
        "--env-backend",
        backend,
        "--env-id",
        env_id,
        "--num-envs",
        "256",
        "--num-steps",
        "4",
        "--num-minibatches",
        "4",
        "--update-epochs",
        "4",
        "--benchmark-warmup-iterations",
        "3",
        "--benchmark-measure-iterations",
        "10",
        "--no-anneal-lr",
        "--seed",
        "1",
    ]
    if script == "ppo_atari_envpool_torchcompile.py" and backend == "cule":
        command += ["--cule-device", "cuda"]
    if compiled:
        command.append("--compile")
    return command, ROOT


def cleanrl_dqn_command(script: str, compiled: bool) -> tuple[list[str], Path]:
    command = [
        sys.executable,
        str(ROOT / "cleanrl" / script),
        "--benchmark",
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
        "256",
        "--replay-ratio",
        "1",
        "--benchmark-warmup-iterations",
        "10",
        "--benchmark-measure-iterations",
        "30",
        "--seed",
        "1",
    ]
    if compiled:
        command.append("--compile")
    return command, ROOT


def build_trials(profiles: list[str], repeats: int, env_counts: list[int]) -> list[dict]:
    trials = []
    if "native-sweep" in profiles:
        for algorithm in ("a2c", "ppo", "vtrace"):
            for num_envs in env_counts:
                for repeat in range(1, repeats + 1):
                    trials.append(
                        {
                            "profile": "native-sweep",
                            "family": "cule_examples",
                            "algorithm": algorithm,
                            "variant": "eager",
                            "num_envs": num_envs,
                            "repeat": repeat,
                        }
                    )
        for repeat in range(1, repeats + 1):
            trials.append(
                {
                    "profile": "native-sweep",
                    "family": "cule_examples",
                    "algorithm": "dqn",
                    "variant": "eager",
                    "num_envs": 256,
                    "repeat": repeat,
                }
            )

    if "matched-ppo" in profiles:
        variants = (
            ("cule_examples", "native_eager"),
            ("cule_examples", "native_compiled"),
            ("cleanrl", "cleanrl_eager"),
            ("torchcompile", "tc_eager"),
            ("torchcompile", "tc_compiled"),
            ("torchcompile", "tc_envpool_compiled"),
        )
        for family, variant in variants:
            for repeat in range(1, repeats + 1):
                trials.append(
                    {
                        "profile": "matched-ppo",
                        "family": family,
                        "algorithm": "ppo",
                        "variant": variant,
                        "num_envs": 256,
                        "num_steps": 4,
                        "update_epochs": 4,
                        "num_minibatches": 4,
                        "repeat": repeat,
                    }
                )

    if "matched-dqn" in profiles:
        variants = (
            ("cule_examples", "native_eager"),
            ("cleanrl", "cleanrl_eager"),
            ("torchcompile", "tc_eager"),
            ("torchcompile", "tc_compiled"),
        )
        for family, variant in variants:
            for repeat in range(1, repeats + 1):
                trials.append(
                    {
                        "profile": "matched-dqn",
                        "family": family,
                        "algorithm": "dqn",
                        "variant": variant,
                        "num_envs": 256,
                        "batch_size": 32,
                        "replay_ratio": 1.0,
                        "multi_step": 1,
                        "repeat": repeat,
                    }
                )
    return trials


def command_for_trial(params: dict) -> tuple[list[str], Path]:
    profile = params["profile"]
    algorithm = params["algorithm"]
    variant = params["variant"]
    if profile == "native-sweep":
        if algorithm in ("a2c", "ppo"):
            return native_on_policy_command(algorithm, params["num_envs"])
        if algorithm == "vtrace":
            return native_vtrace_command(params["num_envs"])
        return native_dqn_command(params["num_envs"])
    if profile == "matched-ppo":
        if variant.startswith("native_"):
            return native_on_policy_command("ppo", 256, compiled=variant.endswith("compiled"))
        if variant == "cleanrl_eager":
            return cleanrl_ppo_command("ppo_atari.py", False)
        backend = "envpool" if variant == "tc_envpool_compiled" else "cule"
        return cleanrl_ppo_command(
            "ppo_atari_envpool_torchcompile.py",
            compiled=variant.endswith("compiled"),
            backend=backend,
        )
    if variant == "native_eager":
        return native_dqn_command()
    if variant == "cleanrl_eager":
        return cleanrl_dqn_command("dqn_atari.py", False)
    return cleanrl_dqn_command("dqn_torchcompile.py", compiled=variant.endswith("compiled"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profiles",
        nargs="+",
        choices=("native-sweep", "matched-ppo", "matched-dqn"),
        default=("native-sweep", "matched-ppo", "matched-dqn"),
    )
    parser.add_argument("--env-counts", type=int, nargs="+", default=(256, 512, 1024, 2048, 4096))
    parser.add_argument(
        "--algorithms",
        nargs="+",
        choices=("a2c", "dqn", "ppo", "vtrace"),
        default=("a2c", "dqn", "ppo", "vtrace"),
        help="filter trials after profile expansion",
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    done = set() if args.no_resume else completed_keys(args.output)
    trials = build_trials(args.profiles, args.repeats, args.env_counts)
    trials = [trial for trial in trials if trial["algorithm"] in args.algorithms]
    for index, params in enumerate(trials, 1):
        if trial_key(params) in done:
            print(f"[{index}/{len(trials)}] skip {params}", flush=True)
            continue
        command, cwd = command_for_trial(params)
        print(f"[{index}/{len(trials)}] run {params}", flush=True)
        record = {
            "command": command,
            "cwd": str(cwd),
            "params": params,
            "started_at": datetime.now(timezone.utc).isoformat(),
        }
        record.update(run_isolated(command, cwd, args.timeout))
        append_record(args.output, record)
        if record["status"] == "ok":
            result = record["result"]
            print(
                f"  {result['fps' if 'fps' in result else 'sps']:,.0f} SPS; "
                f"{result.get('peak_cuda_memory_mb', 0):,.0f} MiB CUDA peak",
                flush=True,
            )
        else:
            print(f"  {record['status']}: {record.get('output_tail', '')[-1000:]}", flush=True)


if __name__ == "__main__":
    main()
