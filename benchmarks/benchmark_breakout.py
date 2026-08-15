#!/usr/bin/env python3
"""Fixed-budget Breakout learning benchmark across every `cleanrl/` trainer.

Protocol
--------
One seed, one game (`BreakoutNoFrameskip-v4`), one backend (CuLE on the GPU), one
budget in frame-skipped agent transitions.  Environment count is the only knob
the benchmark tunes; every other hyperparameter is pinned to the value the
algorithm publishes.

Holding "everything else at the published value" takes two pieces of care.

*Replay trainers.*  The shipped file defaults to `num_envs=1` with one minibatch
per transition.  Raising `--num-envs` from there would silently divide the
gradient cadence by the environment count -- a hundredfold hyperparameter change
wearing the costume of a parallelism change.  The repo already has the receipt:
its 2026-07 sweep ran Rainbow at 512 environments with `--replay-ratio 1` and the
policy never left random play.  So the benchmark holds the *replay ratio* fixed
at the published cadence (`--replay-ratio`, in sampled replay items per collected
transition) while environment count moves.

*On-policy trainers.*  Raising `--num-envs` multiplies the rollout batch, so
`--num-minibatches` rises with it to keep the minibatch size, the epoch count and
therefore the gradient work per collected transition exactly as shipped.

Every hyperparameter is passed explicitly rather than inherited, because the
`*_torchcompile.py` twins ship different defaults from their eager counterparts
(256 environments and batch 512 instead of 1 and 32).  "Eager" and "compiled" are
then the same configuration at different speeds, and nothing else.

Metric
------
The training episodic return.  Every trainer logs it through the same path, and
`cule_env.CuLEVectorEnv` accumulates *unclipped* rewards and flushes only when
`terminated & lives == 0`, so the number is the full-game Breakout score even
though the learner sees sign-clipped rewards under `EpisodicLifeEnv`.  For
epsilon-greedy trainers it is a behaviour-policy score and therefore a slight
underestimate of the greedy policy.

Outputs, under `benchmark_results/artifacts/breakout/<tag>/`:
  runs.jsonl        one record per run: command, config, status, timings
  curves/<key>.csv  global_step, mean_return, batch_mean_return, wall_seconds
  logs/<key>.log    complete trainer output
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from breakout_algorithms import BY_KEY, ENV_ID, OFF_POLICY, ON_POLICY, ROOT
from calibrate_breakout import cule_device_for, on_policy_minibatch_flags, variant_flags

RESULTS = ROOT / "benchmark_results" / "artifacts" / "breakout"
PYTHON = os.environ.get("CULE_PYTHON", "/home/mehrdad96/anaconda3/envs/cule312/bin/python")

# `EpisodeStats` trainers print the windowed mean; ppo_atari and
# pqn_atari_envpool keep CleanRL's original one-episode line.  Same quantity.
STATS_RE = re.compile(
    r"global_step=(\d+), completed_episodes=(\d+), "
    r"batch_mean_return=([-\d.eE+]+), mean_return_(\d+)=([-\d.eE+]+)"
)
SIMPLE_RE = re.compile(r"global_step=(\d+), episodic_return=([-\d.eE+]+)")
SPS_RE = re.compile(r"^SPS:\s*(\d+)")


def build_command(algo, config: dict, total_timesteps: int, seed: int) -> list[str]:
    variant = config["variant"]
    path = algo.eager_path if variant == "eager" else algo.torchcompile_path
    num_envs = config["num_envs"]
    cmd = [
        PYTHON, str(path),
        "--env-backend", "cule",
        "--env-id", ENV_ID,
        "--num-envs", str(num_envs),
        "--total-timesteps", str(total_timesteps),
        "--seed", str(seed),
    ]
    if algo.has_cule_device:
        cmd += ["--cule-device", cule_device_for(num_envs)]
    if algo.has_batch_size and algo.batch_size:
        cmd += ["--batch-size", str(algo.batch_size)]
    cmd += variant_flags(algo, variant)

    if algo.family == OFF_POLICY:
        # BTR and R2D2 ship the published vectorized configuration, so their
        # cadence is left alone; R2D2's flag counts sequences rather than
        # transitions and overriding it would be wrong by a factor of seq_len.
        if (algo.has_replay_ratio and not algo.use_shipped_cadence
                and config.get("replay_ratio") is not None):
            cmd += ["--replay-ratio", str(config["replay_ratio"])]
        if algo.has_buffer_size and config.get("buffer_size"):
            cmd += ["--buffer-size", str(config["buffer_size"])]
    elif algo.family == ON_POLICY:
        cmd += ["--num-steps", str(algo.num_steps)]
        cmd += on_policy_minibatch_flags(algo, num_envs)
    for flag in algo.budget_flags:
        cmd += [flag, str(total_timesteps)]
    cmd += algo.extra
    return cmd


def _terminate(proc: subprocess.Popen) -> None:
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(proc.pid, sig)
            proc.wait(timeout=20)
            return
        except Exception:  # noqa: BLE001 - best effort teardown
            continue


def run_training(cmd: list[str], log_path: Path, curve_path: Path, timeout: float,
                 progress_every: float = 180.0) -> dict:
    """Stream a trainer, tee its output, and distil the learning curve as it goes."""
    started = time.perf_counter()
    rows: list[dict] = []
    sps_samples: list[int] = []
    tail: list[str] = []
    last_progress = started

    proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1,
                            start_new_session=True)
    status = "ok"
    try:
        with log_path.open("w", encoding="utf-8") as log_file:
            assert proc.stdout is not None
            for line in proc.stdout:
                log_file.write(line)
                tail.append(line)
                if len(tail) > 400:
                    del tail[:200]

                match = STATS_RE.search(line)
                simple = None if match else SIMPLE_RE.search(line)
                if match or simple:
                    if match:
                        step, episodes = int(match.group(1)), int(match.group(2))
                        batch_mean, window = float(match.group(3)), int(match.group(4))
                        mean = float(match.group(5))
                    else:
                        step, episodes = int(simple.group(1)), 1
                        batch_mean = mean = float(simple.group(2))
                        window = 1
                    rows.append({
                        "global_step": step, "completed_episodes": episodes,
                        "batch_mean_return": batch_mean, "window": window,
                        "mean_return": mean,
                        "wall_seconds": round(time.perf_counter() - started, 3),
                    })
                sps = SPS_RE.match(line)
                if sps:
                    sps_samples.append(int(sps.group(1)))

                now = time.perf_counter()
                if now - last_progress > progress_every and rows:
                    last_progress = now
                    print(f"      step={rows[-1]['global_step']:>9,} "
                          f"return={rows[-1]['mean_return']:.1f} "
                          f"t={now - started:.0f}s", flush=True)
                if now - started > timeout:
                    raise subprocess.TimeoutExpired(cmd, timeout)
        proc.wait(timeout=60)
        if proc.returncode != 0:
            status = "error"
    except subprocess.TimeoutExpired:
        status = "timeout"
        _terminate(proc)
    finally:
        if rows:
            with curve_path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)

    elapsed = time.perf_counter() - started
    final_step = rows[-1]["global_step"] if rows else 0
    return {
        "status": status,
        "returncode": proc.returncode,
        "wall_seconds": elapsed,
        "curve_points": len(rows),
        "final_step": final_step,
        "final_mean_return": rows[-1]["mean_return"] if rows else None,
        "peak_mean_return": max((r["mean_return"] for r in rows), default=None),
        "reported_sps": sps_samples[-1] if sps_samples else None,
        "wall_sps": final_step / elapsed if elapsed > 0 else None,
        "tail": "".join(tail)[-6000:],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--plan", type=Path, default=RESULTS / "plan.json")
    parser.add_argument("--algorithms", nargs="+", default=None)
    parser.add_argument("--total-timesteps", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--tag", default="1m")
    parser.add_argument("--timeout", type=float, default=3 * 3600.0,
                        help="per-run wall-clock cap in seconds")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    plan = json.loads(args.plan.read_text())
    out_dir = RESULTS / args.tag
    (out_dir / "curves").mkdir(parents=True, exist_ok=True)
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)
    runs_path = out_dir / "runs.jsonl"

    done: set[str] = set()
    if runs_path.exists() and not args.overwrite:
        for line in runs_path.read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                if rec.get("status") == "ok":
                    done.add(rec["algorithm"])

    keys = args.algorithms or plan["order"]
    pending = [k for k in keys if k not in done]
    predicted = sum(plan["configs"][k].get("predicted_seconds") or 0 for k in pending)
    print(f"{len(keys)} algorithms, {len(done)} complete, {len(pending)} to run")
    print(f"budget {args.total_timesteps:,} transitions each; "
          f"predicted total {predicted / 3600:.1f} h\n", flush=True)

    for index, key in enumerate(pending, start=1):
        algo = BY_KEY[key]
        config = plan["configs"][key]
        cmd = build_command(algo, config, args.total_timesteps, args.seed)
        if args.dry_run:
            print(f"[{index}/{len(pending)}] {key}\n    {' '.join(cmd)}\n", flush=True)
            continue

        seconds = config.get("predicted_seconds")
        print(f"[{index}/{len(pending)}] {key} ({algo.label}) "
              f"envs={config['num_envs']} variant={config['variant']} "
              f"ratio={config.get('replay_ratio')} "
              f"predicted={(seconds or 0) / 60:.0f}min", flush=True)

        record = {
            "algorithm": key, "label": algo.label, "family": algo.family,
            "paper": algo.paper, "config": config, "command": cmd,
            "total_timesteps": args.total_timesteps, "seed": args.seed,
            "batch_size": algo.batch_size,
            "paper_replay_ratio": algo.paper_replay_ratio,
            "started_at": datetime.now(timezone.utc).isoformat(),
        }
        record.update(run_training(cmd, out_dir / "logs" / f"{key}.log",
                                   out_dir / "curves" / f"{key}.csv", args.timeout))
        with runs_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
        print(f"    {record['status']} in {record['wall_seconds'] / 60:.1f} min | "
              f"final={record['final_mean_return']} peak={record['peak_mean_return']} "
              f"step={record['final_step']:,} sps={(record['wall_sps'] or 0):.0f}",
              flush=True)
        if record["status"] != "ok":
            print(record["tail"][-1500:], flush=True)


if __name__ == "__main__":
    main()
