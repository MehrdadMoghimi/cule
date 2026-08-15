"""Reproduce the gridworld experiment of the mean-expansion layer paper.

"Accelerating Q-learning through Efficient Value-Sharing across Actions",
Nagarajan, Daley, White & Machado, ICML 2026 (arXiv:2606.29806), Section 6.1
and Figure 2.

This is the only quantitative result in the paper that is reproducible without
a GPU cluster: the deep results are 57 Atari games x 5 seeds x 50M timesteps.
The claim under test is Figure 2's caption, "For most k, IBQ(k) can complete
over 20% more episodes than Q-learning within 1k timesteps", and the sentence
"the gap between Q-learning and IBQ decreases with more experience".

Setup, quoting Section 6.1: a 5x5 stochastic gridworld; "Episodes begin in the
bottom left corner and terminate upon reaching the goal state in the top right
corner"; "the discount rate is 0.95"; "The agent receives a reward of 5 for
reaching the goal and 0 otherwise"; "The agent's actions are the four cardinal
directions"; "The agent transitions according to its chosen action with
probability 3/4 and according to a different randomly selected action with
probability 1/4"; "All agents employ an epsilon-greedy policy with epsilon =
0.1 for 5k timesteps"; "For each k, we report the best results across a sweep
over 61 step sizes chosen from a logarithmic search over (0, 1], running each
combination for 128 seeds."

Two details the paper leaves open, both flagged in the output:
  * the lower end of the step-size sweep -- (0, 1] with 61 log-spaced points is
    taken here as 1e-3 to 1, and `--alpha-min` changes it;
  * what happens at a wall -- the agent is held in place, the usual convention.

IBQ is tabular Q-learning over residuals Z with Q(s, .) = M_k Z(s, .); the
semi-gradient update (Equations 8 and 9) spreads each TD error over every
action in the state:

    Z(s_t, a_t) += alpha * delta * (1 + k/n)
    Z(s_t, a)   += alpha * delta * (k/n)          for a != a_t

k = 0 is exactly Q-learning, which is the baseline the percentages are against.

Usage:
    python benchmarks/me_layer_gridworld.py
    python benchmarks/me_layer_gridworld.py --seeds 128 --timesteps 5000
"""

import argparse
import json
import os
import sys
import time

import numpy as np

GRID = 5
N_ACTIONS = 4
# up, down, left, right as (row, column) deltas
MOVES = np.array([[1, 0], [-1, 0], [0, -1], [0, 1]])
START = (0, 0)
GOAL = (GRID - 1, GRID - 1)


def run_configuration(k, alpha, seeds, timesteps, gamma, epsilon, checkpoints, rng):
    """Run `seeds` independent IBQ(k) agents in lockstep.

    Returns the episode counts at each checkpoint and the largest residual seen,
    which the caller uses to drop step sizes that diverged.

    Everything is vectorized over the seed axis: each array below has a leading
    `seeds` dimension and one environment step advances all of them at once.
    """
    residuals = np.zeros((seeds, GRID * GRID, N_ACTIONS), dtype=np.float64)
    rows = np.full(seeds, START[0], dtype=np.int64)
    columns = np.full(seeds, START[1], dtype=np.int64)
    completed = np.zeros(seeds, dtype=np.int64)
    counts = {}

    shared = k / N_ACTIONS
    for step in range(timesteps):
        states = rows * GRID + columns
        # Q = M_k Z, i.e. the mean component of Z scaled by k + 1.
        q_values = residuals[np.arange(seeds), states]
        q_values = q_values + shared * q_values.sum(axis=1, keepdims=True)

        # Ties are broken uniformly at random. This matters more than it looks:
        # Z starts at zero, so every action ties at the first visit to a state,
        # and np.argmax's first-index rule would send every agent in the same
        # direction. Worse, the ME layer lifts all of a state's actions
        # together, so deterministic tie-breaking penalises exactly the large-k
        # runs the experiment is about.
        maxima = q_values.max(axis=1, keepdims=True)
        tie_breaker = rng.random(q_values.shape) * (q_values == maxima)
        greedy = np.argmax(tie_breaker, axis=1)
        explore = rng.random(seeds) < epsilon
        random_actions = rng.integers(0, N_ACTIONS, size=seeds)
        actions = np.where(explore, random_actions, greedy)

        # 3/4 of the time the chosen action happens; otherwise one of the other
        # three, drawn uniformly.
        slipped = rng.random(seeds) >= 0.75
        alternatives = (actions + 1 + rng.integers(0, N_ACTIONS - 1, size=seeds)) % N_ACTIONS
        executed = np.where(slipped, alternatives, actions)

        next_rows = np.clip(rows + MOVES[executed, 0], 0, GRID - 1)
        next_columns = np.clip(columns + MOVES[executed, 1], 0, GRID - 1)
        terminal = (next_rows == GOAL[0]) & (next_columns == GOAL[1])
        rewards = np.where(terminal, 5.0, 0.0)

        next_states = next_rows * GRID + next_columns
        next_q = residuals[np.arange(seeds), next_states]
        next_q = next_q + shared * next_q.sum(axis=1, keepdims=True)
        bootstrap = np.where(terminal, 0.0, next_q.max(axis=1))

        chosen_q = q_values[np.arange(seeds), actions]
        td_error = rewards + gamma * bootstrap - chosen_q

        # Equation 9: every action in the state absorbs alpha * delta * k/n ...
        residuals[np.arange(seeds), states] += (alpha * td_error * shared)[:, None]
        # ... and Equation 8 gives the taken action its extra alpha * delta.
        residuals[np.arange(seeds), states, actions] += alpha * td_error

        completed += terminal
        rows = np.where(terminal, START[0], next_rows)
        columns = np.where(terminal, START[1], next_columns)

        if (step + 1) in checkpoints:
            counts[step + 1] = completed.copy()
    return counts, np.abs(residuals).max()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=128)
    parser.add_argument("--timesteps", type=int, default=5000)
    parser.add_argument("--step-sizes", type=int, default=61)
    parser.add_argument("--alpha-min", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default=None, help="write the raw results as JSON")
    arguments = parser.parse_args()

    checkpoints = [1000, 2000, 3000, arguments.timesteps]
    checkpoints = sorted(set(c for c in checkpoints if c <= arguments.timesteps))
    alphas = np.logspace(np.log10(arguments.alpha_min), 0.0, arguments.step_sizes)
    # k = 0 is Q-learning; n = |A| = 4 is the paper's default for the deep runs.
    ks = [0.0, 0.1, 0.5, 1.0, 3.0, float(N_ACTIONS)]

    print(f"5x5 stochastic gridworld, {arguments.seeds} seeds x {arguments.step_sizes} step sizes "
          f"x {len(ks)} values of k, {arguments.timesteps} timesteps")
    print(f"step sizes: {alphas[0]:.4g} .. {alphas[-1]:.4g} (log-spaced)")

    start = time.perf_counter()
    best = {}
    raw = {}
    matched = {k: {c: {} for c in checkpoints} for k in ks}
    for k in ks:
        per_checkpoint = {c: (-1.0, None) for c in checkpoints}
        for alpha in alphas:
            rng = np.random.default_rng(arguments.seed)
            counts, residual_health = run_configuration(
                k, alpha, arguments.seeds, arguments.timesteps,
                arguments.gamma, arguments.epsilon, set(checkpoints), rng,
            )
            if not np.isfinite(residual_health):
                continue  # this step size diverged; it is not a candidate
            for checkpoint, values in counts.items():
                matched[k][checkpoint][float(alpha)] = float(values.mean())
                mean = float(values.mean())
                if mean > per_checkpoint[checkpoint][0]:
                    per_checkpoint[checkpoint] = (mean, float(alpha), values.std(ddof=1) / np.sqrt(len(values)))
        best[k] = per_checkpoint
        raw[k] = {str(c): per_checkpoint[c] for c in checkpoints}
        print(f"  k={k:<4g} done ({time.perf_counter() - start:.0f}s)")

    baseline = best[0.0]
    print(f"\n{'k':>6} | " + " | ".join(f"{c:>6} steps" for c in checkpoints))
    print("-" * (8 + 15 * len(checkpoints)))
    for k in ks:
        cells = []
        for checkpoint in checkpoints:
            mean = best[k][checkpoint][0]
            reference = baseline[checkpoint][0]
            gain = 100.0 * (mean - reference) / reference if reference else float("nan")
            cells.append(f"{mean:6.1f} ({gain:+5.1f}%)")
        label = "0 (Q)" if k == 0 else f"{k:g}"
        print(f"{label:>6} | " + " | ".join(cells))

    # The two protocols answer different questions, and they disagree, so both
    # are printed. Tuning each k separately compares the best IBQ(k) agent to
    # the best Q-learning agent; holding alpha fixed isolates the value-sharing
    # mechanism from the fact that large k wants a smaller step size -- which
    # the paper itself notes ("when k is large, smaller step sizes are more
    # suitable").
    first = checkpoints[0]
    shared_alphas = sorted(set.intersection(*[set(matched[k][first]) for k in ks]))
    print(f"\nSame step size for every k, at {first} timesteps:")
    print(f"{'alpha':>8} | " + " | ".join(f"k={k:<5g}" for k in ks))
    print("-" * (10 + 9 * len(ks)))
    for alpha in shared_alphas:
        if alpha > 0.2:
            continue
        reference = matched[0.0][first][alpha]
        cells = []
        for k in ks:
            value = matched[k][first][alpha]
            cells.append(f"{value:5.1f}" if k == 0 else f"{100 * (value - reference) / reference:+6.1f}%")
        print(f"{alpha:>8.4f} | " + " | ".join(f"{cell:>7}" for cell in cells))

    print("\nPaper (Figure 2): \"For most k, IBQ(k) can complete over 20% more episodes")
    print("than Q-learning within 1k timesteps\", and the gap shrinks with more experience.")
    first = checkpoints[0]
    gains = {
        k: 100.0 * (best[k][first][0] - baseline[first][0]) / baseline[first][0]
        for k in ks if k != 0.0
    }
    over_twenty = [k for k, gain in gains.items() if gain > 20.0]
    print(f"\nAt {first} timesteps this run gives: "
          + ", ".join(f"k={k:g} {gain:+.1f}%" for k, gain in gains.items()))
    print(f"  -> {len(over_twenty)}/{len(gains)} values of k exceed +20%"
          f" ({'matches' if len(over_twenty) >= len(gains) / 2 else 'does NOT match'} \"most k\")")

    last = checkpoints[-1]
    late = {
        k: 100.0 * (best[k][last][0] - baseline[last][0]) / baseline[last][0]
        for k in ks if k != 0.0
    }
    shrunk = sum(1 for k in gains if late[k] < gains[k])
    print(f"At {last} timesteps: " + ", ".join(f"k={k:g} {gain:+.1f}%" for k, gain in late.items()))
    print(f"  -> the gap shrank for {shrunk}/{len(gains)} values of k"
          f" ({'matches' if shrunk >= len(gains) / 2 else 'does NOT match'} the paper)")

    if arguments.output:
        os.makedirs(os.path.dirname(os.path.abspath(arguments.output)), exist_ok=True)
        with open(arguments.output, "w") as handle:
            json.dump(
                {"config": vars(arguments), "checkpoints": checkpoints, "best": raw},
                handle, indent=2, default=float,
            )
        print(f"\nwrote {arguments.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
