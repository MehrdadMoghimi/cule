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

## Correctness

- `verify_chain.py` — train a trainer on an MDP whose optimal action-value
  function is known exactly, then compare the learned Q against it entry by
  entry. `../cleanrl/diagnostic_env.py` defines the MDP (a chain: advance five
  states, collect the reward, terminate) and renders it into the same 84x84
  uint8 four-frame observations Atari produces, so every trainer and every
  network runs on it unmodified. `make_cule_env` routes to it whenever the env
  id names it, which is why there is no new flag on any trainer.

  This covers what the other two layers cannot. The equivalence tests in
  `../tests/` check that each trainer computes the right formulas; the
  cross-checks in `../tests/crosscheck/` check its components against the
  authors' official code; a Breakout run only detects breakage severe enough to
  clear seed noise. Wiring bugs — an unmasked terminal bootstrap, an off-by-one
  in the discount, a stale target — survive all three and cost a few points of
  final score. Against a closed form they are unmistakable:

```bash
python benchmarks/verify_chain.py --all             # every trainer, coverage table
python benchmarks/verify_chain.py dqn_torchcompile --quiet
python benchmarks/verify_chain.py --list            # trainers with a Q adapter

# Confirm the check has teeth: mutate the trainer's source and watch it fail.
python benchmarks/verify_chain.py dqn_torchcompile --inject-bug no-terminal-mask
python benchmarks/verify_chain.py dqn_torchcompile --inject-bug discount-the-reward
```

  The report also fits the best scalar `c` in `Q ~ c * Q*`, because the common
  bugs are scale errors with a known constant: dropping the terminal mask
  converges to `Q*` times `1 / (1 - gamma**n)`, and discounting the reward as
  well as the bootstrap converges to `Q*` times `gamma`. Both leave the greedy
  policy intact, so no score curve can see them and this table names them.
  `../tests/test_diagnostic_env.py` verifies the closed form against tabular
  value iteration and the environment's dynamics against the model that value
  iteration was given.

  `chain_adapters.py` holds the per-trainer knowledge of *where* the value
  function lives: a plain `Q(s, .)` head, a distribution over atoms or
  quantiles, a critic's `V(s)`, a recurrent state that has to be carried down
  the trajectory, or a latent only the trainer's own encoder defines. It also
  records what kind of check each trainer supports — a full value comparison,
  a greedy-policy check where no value is readable at a raw observation, or
  unsupported with the reason. `--all` runs each trainer in its own subprocess
  (sixty-two of them share module names, CUDA state and compile caches, and one
  that hangs must not take the sweep with it) and writes per-trainer JSON plus
  a summary to `benchmark_results/chain/`.

  Budgets are per family, in `verify_chain.py`. This matters more than it
  looks: A2C does one update per rollout, so 200k transitions buy it about 195
  gradient steps and it fails the comparison for lack of training rather than
  for being wrong. A false FAIL is the one outcome that would make the harness
  untrustworthy, so the on-policy family gets a budget that buys a comparable
  number of updates to the off-policy default.

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

## Hyperparameter search

Two passes, deliberately split by trial budget.

`sweep_highenv.py` is the **breadth** pass: it pins the environment count as
high as each algorithm can go (1,024, falling back to 512 then 256) and walks a
hand-written ladder of candidates — feasibility, screen, refine, confirm — over
all thirty-five trainers on a wall-clock budget. Nine to eleven trials per
algorithm is below the number at which a model-based optimizer beats a ladder,
which is exactly why the ladder is what runs here.

```bash
python benchmarks/sweep_highenv.py                    # all stages, all algorithms
python benchmarks/analyze_highenv.py                  # rank and write the report
```

`sweep_protein.py` is the **depth** pass over the ladder's top finishers, using
Protein — PufferAI's reworked CARBS, vendored as `protein.py` — to spend a
hundred or more trials on one algorithm. It needs `gpytorch`, `scikit-learn`
and `scipy`.

```bash
# top 5 of the ladder, 100 trials each, resumable
python benchmarks/sweep_protein.py --top 5 --trials 100

# re-run each winner at the full window on three seeds, then rewrite the report
python benchmarks/sweep_protein.py --top 5 --stages confirm
python benchmarks/sweep_protein.py --stages report
```

The training window is itself a searched dimension (30 s to `--max-window`) and
is what Protein models as *cost*, so the search opens with cheap ~30 s trials
and buys longer ones only where the score model says the neighbourhood is worth
paying for. Cost is measured as wall clock *including startup*, which charges a
`_torchcompile.py` twin for its compile and graph-capture time instead of
letting the protocol hide it.

Environment count, the eager/compiled variant and `--total-timesteps` are taken
from the ladder's `state.json` and held fixed — the last of those because every
annealing schedule in these trainers is a fraction of it, so searching it would
make a faster configuration anneal more slowly. The variant is fixed rather than
searched because it is not a continuum; `--variant both` runs it as two studies
instead.

## Cross-game exploration

The sweeps above optimise a configuration. The `exp_h0NN_*.py` scripts do the
opposite: they pin everything, move one thing, and record what changed. Results
land in `benchmark_results/exploration/<experiment>/runs.jsonl`, keyed by a name
that is unique within the experiment, so an interrupted run resumes by being
launched again and a completed cell is never repeated.

```bash
# Training-free game profiles: random-policy baselines, reward structure,
# state-revisit rate.  Seconds per game, and the baselines every cross-game
# score is measured against.
python benchmarks/game_descriptors.py

# One experiment; add --seeds 2,3 to replicate, --dry-run to print commands
python benchmarks/exp_h021_currency_by_game.py --seeds 1,2,3

# Per-(game, trainer) slopes with error bars, for any slope-shaped experiment
python benchmarks/analyze_slopes.py h021_currency_by_game h023_budget_tier

# Hypotheses, findings and the rendered journal
python benchmarks/journal.py hypothesis "..." --why "..."
python benchmarks/journal.py finding "..." --claim "..." --confidence high
python benchmarks/journal.py render
```

Three conventions are worth keeping, all of them learned the hard way and all
recorded with the evidence in `JOURNAL.md`:

**Score above the random baseline, never raw.** Krull pays a random agent 1,634
and Breakout 1.3, so a log ratio of raw returns is mostly a ratio of baselines,
compressed by a different factor in every game. `game_descriptors.py` measures
those baselines; nothing cross-game should be read without them.

**Form contrasts within a seed, then average.** Averaging scores across seeds
first and taking one log ratio of the means collapses when any seed finishes
below the baseline — on Krull one of three does, which gave that cell an error
of ±17 under the wrong order.

**Three seeds, always.** Across sixteen contrasts that now have them, five
change sign between the seed-1 value and the multi-seed mean: a 31% sign-error
rate, on runs that cost 45–200 seconds each. Four conclusions in this journal
were retracted by replication, so a one-seed number is a hypothesis, not a
result.

Experiments that change per-step GPU work go through `guard_spec` in
`exp_runner.py`. A configuration authored here once bugchecked the host with a
Windows display-watchdog timeout; the cap is on the replay ratio, which is the
learner-to-emulator work ratio and the only quantity that separated the cell
that crashed from the cells that had run for hours. See the comment there before
raising it.
