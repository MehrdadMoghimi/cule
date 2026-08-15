# Benchmarks

Standalone scripts for measuring CuLE throughput and learning performance.
They launch the trainers in [../cleanrl/](../cleanrl/) and the native
examples in [../examples/](../examples/) as subprocesses, and write raw
JSONL/CSV artifacts and reports to `benchmark_results/` at the repository
root (git-ignored; results are machine-specific). Headline results from an
RTX 4090 are summarized in the [project README](../README.md).

## Throughput

- `benchmark_step_throughput.py` — raw CuLE environment stepping (no
  learner), one process, sweepable over environment counts. Also useful for
  tuning `CULE_RENDER_LANES`.
- `benchmark_cule_envpool.py` — CuLE vs EnvPool for the compiled PPO and PQN
  trainers: matched environment counts, per-backend scaling curves, and
  standalone environment-probe capacity checks.
- `benchmark_implementations.py` — cross-implementation comparison: native
  CuLE examples (A2C, DQN, PPO, V-trace) versus eager CleanRL and
  torchcompile trainers at matched schedules.

## Learning

- `breakout_algorithms.py` / `calibrate_breakout.py` / `plan_breakout.py` /
  `benchmark_breakout.py` / `analyze_breakout.py` — the all-algorithm Breakout
  benchmark. One entry per algorithm in the registry, a throughput calibration
  that separates collection cost from gradient cost, a planner that turns those
  measurements into an operating point, the runner, and the report generator.
  Environment count is the only knob tuned; every other hyperparameter is pinned
  to the value the algorithm publishes. See "Breakout, all algorithms" below.
- `benchmark_learning.py` — fixed-budget (10M transitions) Breakout learning
  comparison across A2C, DQN, PPO and V-trace with a shared evaluation protocol.
- `search_torchcompile_breakout.py` — short-budget hyperparameter search for
  the torchcompile PPO/PQN/Rainbow trainers, plus selected long reruns
  (`--selected-10m`).
- `run_ppo_breakout_paired_10m.py` — paired CuLE-vs-EnvPool compiled-PPO runs
  at 10M transitions: identical hyperparameters per pair, common CPU CuLE
  evaluation for both backends.

## Analysis

Each `analyze_*.py` script aggregates the artifacts of its matching runner
into summary CSVs and plots:

- `analyze_breakout.py` ← `benchmark_breakout.py`
- `analyze_cule_envpool.py` ← `benchmark_cule_envpool.py`
- `analyze_implementation_benchmark.py` ← `benchmark_implementations.py`
- `analyze_learning_benchmark.py` ← `benchmark_learning.py`
- `analyze_torchcompile_breakout_10m.py` ← `search_torchcompile_breakout.py --selected-10m`
- `analyze_ppo_breakout_paired_10m.py` ← `run_ppo_breakout_paired_10m.py`

A related throughput sweep for the native examples lives at
[../examples/benchmark_search.py](../examples/benchmark_search.py).

## Example usage

```bash
# Raw environment stepping at 1,024 envs
python benchmarks/benchmark_step_throughput.py --envs 1024

# CuLE vs EnvPool PPO/PQN training sweep
python benchmarks/benchmark_cule_envpool.py training \
  --output benchmark_results/cule_envpool.jsonl

# Paired 10M-step PPO learning runs, then aggregate
python benchmarks/run_ppo_breakout_paired_10m.py
python benchmarks/analyze_ppo_breakout_paired_10m.py
```

## Breakout, all algorithms

A fixed-budget learning comparison across every trainer in `cleanrl/`, on
`BreakoutNoFrameskip-v4` with the CuLE GPU backend. Environment count is the
only knob tuned; everything else is pinned to the algorithm's published value.

Run it in four steps — each stage writes to
`benchmark_results/artifacts/breakout/` and every stage is resumable:

```bash
# 1. Measure collection cost and gradient cost separately, per algorithm
python benchmarks/calibrate_breakout.py --variants eager

# 2. Turn those measurements into an operating point (envs, replay ratio)
python benchmarks/plan_breakout.py --variant eager --total-timesteps 1000000

# 3. Run every algorithm at its planned operating point
python benchmarks/benchmark_breakout.py --total-timesteps 1000000 --tag 1m

# 4. Rank, plot and write the report
python benchmarks/analyze_breakout.py --tag 1m
```

Why the calibration step exists: a replay trainer's shipped default is one
environment with one minibatch per transition, so raising `--num-envs` without
compensating would divide its gradient cadence by the environment count. The
repo has already seen where that leads — the 2026-07 sweep ran Rainbow at 512
environments with `--replay-ratio 1` and it never left random play. The
calibration separates the two cost terms so environment count can be chosen for
throughput while the *replay ratio* stays at the published value.

`plan_breakout.py --budget-hours` caps how long one run may take; a trainer
whose published cadence would exceed it has its replay ratio stepped down a
power-of-two ladder, and the report flags every row where that happened.
