#!/usr/bin/env python3
"""Aggregate and plot the selected Breakout learning runs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "benchmark_results" / "artifacts" / "learning"

CURVES = {
    "A2C": RESULTS / "learning_breakout_selected_a2c_seed1_test.csv",
    "DQN": RESULTS / "learning_breakout_selected_dqn_seed1.csv",
    "PPO": RESULTS / "learning_breakout_ppo_seed1.csv",
    "V-trace": RESULTS / "learning_breakout_vtrace_seed1_test.csv",
}

NATIVE_COLUMNS = {
    "total_time": "training_seconds",
    "rmean": "reward_mean",
    "rmedian": "reward_median",
    "rmin": "reward_min",
    "rmax": "reward_max",
    "rstd": "reward_std",
    "lmean": "length_mean",
    "lmedian": "length_median",
    "lmin": "length_min",
    "lmax": "length_max",
    "lstd": "length_std",
}

OUTPUT_COLUMNS = [
    "algorithm",
    "seed",
    "frames",
    "training_seconds",
    "reward_mean",
    "reward_median",
    "reward_min",
    "reward_max",
    "reward_std",
    "length_mean",
    "length_median",
    "length_min",
    "length_max",
    "length_std",
]


def read_curve(algorithm: str, path: Path) -> list[dict[str, float | int | str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    lines = path.read_text(encoding="utf-8").splitlines()
    header_index = next(index for index, line in enumerate(lines) if line.startswith("frames,") or line.startswith("algorithm,"))
    rows: list[dict[str, float | int | str]] = []
    for source in csv.DictReader(lines[header_index:]):
        row: dict[str, float | int | str] = {"algorithm": algorithm, "seed": int(source.get("seed", 1))}
        for source_name, value in source.items():
            name = NATIVE_COLUMNS.get(source_name, source_name)
            if name in ("algorithm", "seed", "worker_wall_seconds") or value in (None, ""):
                continue
            row[name] = int(float(value)) if name in ("frames", "length_min", "length_max") else float(value)
        rows.append(row)
    if len(rows) < 2:
        raise ValueError(f"incomplete curve in {path}: found {len(rows)} evaluation point(s)")
    return rows


def first_crossing(rows: list[dict[str, float | int | str]], threshold: float, key: str) -> str:
    for row in rows:
        if float(row["reward_mean"]) >= threshold:
            value = float(row[key])
            return str(int(value)) if key == "frames" else f"{value:.3f}"
    return ""


def summarize(algorithm: str, rows: list[dict[str, float | int | str]]) -> dict[str, str]:
    best = max(rows, key=lambda row: float(row["reward_mean"]))
    final = rows[-1]
    frames = np.asarray([float(row["frames"]) for row in rows])
    rewards = np.asarray([float(row["reward_mean"]) for row in rows])
    auc_mean = float(np.trapezoid(rewards, frames) / (frames[-1] - frames[0]))
    summary = {
        "algorithm": algorithm,
        "final_frames": str(int(float(final["frames"]))),
        "final_training_seconds": f"{float(final['training_seconds']):.3f}",
        "training_fps": f"{float(final['frames']) / float(final['training_seconds']):.1f}",
        "final_reward_mean": f"{float(final['reward_mean']):.3f}",
        "final_reward_std": f"{float(final['reward_std']):.3f}",
        "best_reward_mean": f"{float(best['reward_mean']):.3f}",
        "best_frames": str(int(float(best["frames"]))),
        "best_training_seconds": f"{float(best['training_seconds']):.3f}",
        "curve_mean_reward": f"{auc_mean:.3f}",
    }
    for threshold in (10, 25, 50, 100, 200):
        summary[f"frames_to_{threshold}"] = first_crossing(rows, threshold, "frames")
        summary[f"training_seconds_to_{threshold}"] = first_crossing(rows, threshold, "training_seconds")
    return summary


def write_csv(path: Path, rows: list[dict[str, object]], columns: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def plot_curves(curves: dict[str, list[dict[str, float | int | str]]], output: Path) -> None:
    colors = {"A2C": "#4C78A8", "DQN": "#F58518", "PPO": "#54A24B", "V-trace": "#B279A2"}
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.1), constrained_layout=True)
    for algorithm, rows in curves.items():
        frames = np.asarray([float(row["frames"]) / 1_000_000 for row in rows])
        seconds = np.asarray([float(row["training_seconds"]) / 60 for row in rows])
        mean = np.asarray([float(row["reward_mean"]) for row in rows])
        for axis, x in zip(axes, (frames, seconds)):
            axis.plot(x, mean, marker="o", markersize=4, linewidth=2.2, label=algorithm, color=colors[algorithm])
    axes[0].set_xlabel("Environment transitions (millions)")
    axes[0].set_title("Sample efficiency")
    axes[1].set_xlabel("Training time, excluding evaluation (minutes)")
    axes[1].set_title("Compute efficiency")
    for axis in axes:
        axis.set_ylabel("Mean full-game Breakout return")
        axis.grid(alpha=0.25)
        axis.set_ylim(bottom=0)
    axes[0].legend(frameon=False, ncol=2)
    fig.suptitle("Breakout learning curves — seed 1, 10 deterministic evaluation games", fontsize=14)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-prefix", default="learning_breakout")
    args = parser.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    curves = {algorithm: read_curve(algorithm, path) for algorithm, path in CURVES.items()}
    combined = [row for algorithm in CURVES for row in curves[algorithm]]
    combined.sort(key=lambda row: (str(row["algorithm"]), int(row["frames"])))
    summaries = [summarize(algorithm, curves[algorithm]) for algorithm in CURVES]
    summaries.sort(key=lambda row: float(row["best_reward_mean"]), reverse=True)

    combined_path = RESULTS / f"{args.output_prefix}_evaluations.csv"
    summary_path = RESULTS / f"{args.output_prefix}_summary.csv"
    plot_path = RESULTS / f"{args.output_prefix}_curves.png"
    write_csv(combined_path, combined, OUTPUT_COLUMNS)
    write_csv(summary_path, summaries, list(summaries[0]))
    plot_curves(curves, plot_path)
    print(combined_path)
    print(summary_path)
    print(plot_path)


if __name__ == "__main__":
    main()
