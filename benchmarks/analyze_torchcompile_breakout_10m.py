#!/usr/bin/env python3
"""Generate plots and derived metrics for the selected 10M Breakout runs."""

from __future__ import annotations

import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cule")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "benchmark_results" / "artifacts" / "torchcompile_breakout_10m"
CURVE_DIR = ARTIFACT_DIR / "curves"
PLOT_DIR = ARTIFACT_DIR / "plots"
DERIVED_CSV = ARTIFACT_DIR / "curve_summary.csv"

ALGORITHMS = ("ppo", "pqn", "rainbow")
BACKENDS = ("cule", "envpool")
COLORS = {"cule": "#0072B2", "envpool": "#D55E00"}
DISPLAY = {"ppo": "PPO", "pqn": "PQN", "rainbow": "Rainbow", "cule": "CuLE", "envpool": "EnvPool"}


def read_curve(path: Path) -> list[dict[str, float]]:
    with path.open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    return [{key: float(value) for key, value in row.items() if key not in {"algorithm"}} for row in rows]


def load_curves() -> dict[tuple[str, str], list[dict[str, float]]]:
    curves = {}
    for path in sorted(CURVE_DIR.glob("*.csv")):
        algorithm, backend, *_ = path.stem.split("__")
        curves[(algorithm, backend)] = read_curve(path)
    missing = [
        (algorithm, backend)
        for algorithm in ALGORITHMS
        for backend in BACKENDS
        if (algorithm, backend) not in curves
    ]
    if missing:
        raise FileNotFoundError(f"missing curves: {missing}")
    return curves


def plot_learning(curves: dict, x_key: str, divisor: float, xlabel: str, output_name: str) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)
    for axis, algorithm in zip(axes, ALGORITHMS):
        for backend in BACKENDS:
            rows = curves[(algorithm, backend)]
            x = np.array([row[x_key] / divisor for row in rows])
            mean = np.array([row["reward_mean"] for row in rows])
            std = np.array([row["reward_std"] for row in rows])
            axis.plot(
                x,
                mean,
                color=COLORS[backend],
                marker="o",
                linewidth=2.2,
                markersize=4.5,
                label=DISPLAY[backend],
            )
            axis.fill_between(x, np.maximum(0, mean - std), mean + std, color=COLORS[backend], alpha=0.14)
        axis.set_title(DISPLAY[algorithm], fontweight="bold")
        axis.set_xlabel(xlabel)
        axis.set_ylabel("Full-game reward (mean of 5)")
        axis.grid(alpha=0.25)
        axis.legend(frameon=False)
        axis.set_ylim(bottom=0)
    figure.suptitle("Compiled Breakout learning curves: CuLE vs EnvPool", fontsize=14, fontweight="bold")
    figure.savefig(PLOT_DIR / output_name, dpi=200)
    plt.close(figure)


def plot_final_comparison(curves: dict) -> None:
    labels = []
    final_rewards = []
    throughput = []
    colors = []
    for algorithm in ALGORITHMS:
        for backend in BACKENDS:
            row = curves[(algorithm, backend)][-1]
            labels.append(f"{DISPLAY[algorithm]}\n{DISPLAY[backend]}")
            final_rewards.append(row["reward_mean"])
            throughput.append(row["frames"] / row["training_seconds"])
            colors.append(COLORS[backend])

    x = np.arange(len(labels))
    figure, (reward_axis, speed_axis) = plt.subplots(1, 2, figsize=(14, 4.8), constrained_layout=True)
    reward_bars = reward_axis.bar(x, final_rewards, color=colors)
    reward_axis.set_xticks(x, labels)
    reward_axis.set_ylabel("Final full-game reward")
    reward_axis.set_title("Final reward at 10M transitions", fontweight="bold")
    reward_axis.grid(axis="y", alpha=0.25)
    reward_axis.bar_label(reward_bars, fmt="%.1f", padding=3)

    speed_bars = speed_axis.bar(x, np.array(throughput) / 1000, color=colors)
    speed_axis.set_xticks(x, labels)
    speed_axis.set_ylabel("Training throughput (thousand transitions/s)")
    speed_axis.set_title("End-to-end training throughput", fontweight="bold")
    speed_axis.grid(axis="y", alpha=0.25)
    speed_axis.bar_label(speed_bars, fmt="%.1f", padding=3)
    figure.savefig(PLOT_DIR / "final_reward_and_throughput.png", dpi=200)
    plt.close(figure)


def plot_peak_vs_final(curves: dict) -> None:
    labels = []
    peaks = []
    finals = []
    for algorithm in ALGORITHMS:
        for backend in BACKENDS:
            rewards = [row["reward_mean"] for row in curves[(algorithm, backend)]]
            labels.append(f"{DISPLAY[algorithm]}\n{DISPLAY[backend]}")
            peaks.append(max(rewards))
            finals.append(rewards[-1])

    x = np.arange(len(labels))
    width = 0.36
    figure, axis = plt.subplots(figsize=(11.5, 4.8), constrained_layout=True)
    peak_bars = axis.bar(x - width / 2, peaks, width, label="Peak", color="#009E73")
    final_bars = axis.bar(x + width / 2, finals, width, label="Final", color="#CC79A7")
    axis.set_xticks(x, labels)
    axis.set_ylabel("Full-game reward")
    axis.set_title("Peak versus final evaluation reward", fontweight="bold")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False)
    axis.bar_label(peak_bars, fmt="%.1f", padding=3)
    axis.bar_label(final_bars, fmt="%.1f", padding=3)
    figure.savefig(PLOT_DIR / "peak_vs_final_reward.png", dpi=200)
    plt.close(figure)


def write_derived_summary(curves: dict) -> None:
    rows = []
    for algorithm in ALGORITHMS:
        for backend in BACKENDS:
            curve = curves[(algorithm, backend)]
            rewards = np.array([row["reward_mean"] for row in curve])
            frames = np.array([row["frames"] for row in curve])
            seconds = np.array([row["training_seconds"] for row in curve])
            peak_index = int(np.argmax(rewards))
            rows.append(
                {
                    "algorithm": algorithm,
                    "backend": backend,
                    "peak_reward": rewards[peak_index],
                    "peak_frames": int(frames[peak_index]),
                    "peak_training_seconds": seconds[peak_index],
                    "final_reward": rewards[-1],
                    "final_frames": int(frames[-1]),
                    "final_training_seconds": seconds[-1],
                    "train_sps": frames[-1] / seconds[-1],
                    "curve_reward_mean": rewards.mean(),
                    "peak_to_final_change": rewards[-1] - rewards[peak_index],
                }
            )
    with DERIVED_CSV.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    curves = load_curves()
    plot_learning(curves, "frames", 1_000_000, "Agent transitions (millions)", "learning_vs_transitions.png")
    plot_learning(curves, "training_seconds", 60, "Training time (minutes)", "learning_vs_training_time.png")
    plot_final_comparison(curves)
    plot_peak_vs_final(curves)
    write_derived_summary(curves)
    print(f"plots: {PLOT_DIR}")
    print(f"derived summary: {DERIVED_CSV}")


if __name__ == "__main__":
    main()
