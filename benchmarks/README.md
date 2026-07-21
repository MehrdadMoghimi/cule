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

- `benchmark_learning.py` — fixed-budget (10M transitions) Breakout learning
  comparison across algorithms with a shared evaluation protocol.
- `search_torchcompile_breakout.py` — short-budget hyperparameter search for
  the torchcompile PPO/PQN/Rainbow trainers, plus selected long reruns
  (`--selected-10m`).
- `run_ppo_breakout_paired_10m.py` — paired CuLE-vs-EnvPool compiled-PPO runs
  at 10M transitions: identical hyperparameters per pair, common CPU CuLE
  evaluation for both backends.

## Analysis

Each `analyze_*.py` script aggregates the artifacts of its matching runner
into summary CSVs and plots:

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
