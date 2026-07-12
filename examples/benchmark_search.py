#!/usr/bin/env python3
"""Search CuLE policy-training throughput on the local GPU.

Every completed trial is appended to JSONL, so an interrupted search can be
resumed without losing results.  The benchmark measures complete rollout and
optimizer iterations; evaluation and checkpoint I/O are excluded.
"""

import argparse
import itertools
import json
from pathlib import Path
import re
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parent
RESULT_RE = re.compile(r"THROUGHPUT_RESULT (\{.*\})")
VTRACE_RE = re.compile(r"Benchmark - training: ([0-9]+) PFS")


def trial_matrix(algorithms, profile, requested_env_counts=None,
                 requested_rollout_steps=None, normalization="on", repeats=1):
    if profile == "quick":
        env_counts = [512, 1200, 2000]
        rollout_steps = [5, 20]
    else:
        env_counts = [256, 512, 800, 1200, 1600, 2000, 2400]
        rollout_steps = [5, 10, 20, 40]
    env_counts = requested_env_counts or env_counts
    rollout_steps = requested_rollout_steps or rollout_steps
    normalizations = {"on": [True], "off": [False], "both": [True, False]}[normalization]

    for algorithm in algorithms:
        for num_ales, num_steps, normalize, repeat in itertools.product(
                env_counts, rollout_steps, normalizations, range(repeats)):
            trial = {"algorithm": algorithm, "num_ales": num_ales,
                     "num_steps": num_steps, "normalize": normalize}
            if repeats > 1:
                trial["repeat"] = repeat + 1
            if algorithm == "vtrace":
                # Keep the optimizer minibatch near 64 environments while
                # scaling the simulator batch.
                divisors = [d for d in range(1, num_ales + 1)
                            if num_ales % d == 0]
                trial["num_minibatches"] = min(
                    divisors, key=lambda d: abs(num_ales / d - 64))
                trial["num_steps_per_update"] = 1
            yield trial


def command_for(trial, env_name, warmup, measure):
    algorithm = trial["algorithm"]
    command = [sys.executable, f"{algorithm}_main.py", "--env-name", env_name,
               "--use-cuda-env", "--num-ales", str(trial["num_ales"]),
               "--num-steps", str(trial["num_steps"]),
               "--benchmark-warmup-iterations", str(warmup),
               "--benchmark-measure-iterations", str(measure)]
    if trial["normalize"]:
        command.append("--normalize")
    if algorithm == "vtrace":
        command += ["--benchmark", "--num-minibatches",
                    str(trial["num_minibatches"]), "--num-steps-per-update",
                    str(trial["num_steps_per_update"])]
    else:
        command += ["--throughput-benchmark"]
    return command


def key(trial):
    return json.dumps(trial, sort_keys=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithms", nargs="+", choices=["a2c", "ppo", "vtrace"],
                        default=["a2c", "ppo", "vtrace"])
    parser.add_argument("--profile", choices=["quick", "full"], default="quick")
    parser.add_argument("--env-counts", nargs="+", type=int)
    parser.add_argument("--rollout-steps", nargs="+", type=int)
    parser.add_argument("--normalization", choices=["on", "off", "both"], default="on")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--env-name", default="PongNoFrameskip-v4")
    parser.add_argument("--warmup-iterations", type=int, default=10)
    parser.add_argument("--measure-iterations", type=int, default=30)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "benchmark_results.jsonl")
    parser.add_argument("--timeout", type=float, default=180.0)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    completed = set()
    if args.output.exists():
        for line in args.output.read_text().splitlines():
            record = json.loads(line)
            if record.get("status") == "ok":
                completed.add(key(record["params"]))

    trials = list(trial_matrix(args.algorithms, args.profile, args.env_counts,
                               args.rollout_steps, args.normalization, args.repeats))
    for index, trial in enumerate(trials, 1):
        if key(trial) in completed:
            print(f"[{index}/{len(trials)}] skip completed {trial}", flush=True)
            continue
        command = command_for(trial, args.env_name, args.warmup_iterations,
                              args.measure_iterations)
        cwd = ROOT / trial["algorithm"]
        print(f"[{index}/{len(trials)}] {trial}", flush=True)
        started = time.time()
        record = {"params": trial, "command": command, "status": "error"}
        try:
            proc = subprocess.run(command, cwd=cwd, text=True,
                                  stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT,
                                  timeout=args.timeout)
            match = RESULT_RE.search(proc.stdout)
            if trial["algorithm"] == "vtrace":
                match = VTRACE_RE.search(proc.stdout)
            if proc.returncode == 0 and match:
                result = (json.loads(match.group(1)) if trial["algorithm"] != "vtrace"
                          else {"algorithm": "vtrace", "fps": float(match.group(1))})
                record.update(status="ok", result=result)
            else:
                record["returncode"] = proc.returncode
                record["output_tail"] = proc.stdout[-4000:]
        except subprocess.TimeoutExpired as exc:
            record["status"] = "timeout"
            record["output_tail"] = (exc.stdout or "")[-4000:]
        record["wall_seconds"] = time.time() - started
        with args.output.open("a") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
        if record["status"] == "ok":
            print(f"  {record['result']['fps']:,.0f} FPS", flush=True)
        else:
            print(f"  {record['status']}", flush=True)

    successful = []
    for line in args.output.read_text().splitlines():
        record = json.loads(line)
        if record.get("status") == "ok" and record["params"]["algorithm"] in args.algorithms:
            successful.append(record)
    successful.sort(key=lambda item: item["result"]["fps"], reverse=True)
    print("\nRanking")
    for rank, record in enumerate(successful[:10], 1):
        print(f"{rank:2}. {record['result']['fps']:10,.0f} FPS  {record['params']}")


if __name__ == "__main__":
    main()
