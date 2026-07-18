#!/usr/bin/env python3
"""Generate the report figure and derived metrics for paired PPO Breakout runs."""

from __future__ import annotations

import csv
import os
from pathlib import Path
import re

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cule")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "benchmark_results" / "artifacts" / "ppo_breakout_paired_10m"
CURVE_DIR = ARTIFACT_DIR / "curves"
PLOT_DIR = ARTIFACT_DIR / "plots"
CURVE_SUMMARY = ARTIFACT_DIR / "curve_summary.csv"
PAIR_SUMMARY = ARTIFACT_DIR / "pair_summary.csv"

ENV_COUNTS = (256, 512, 1024, 2048)
BACKENDS = ("cule", "envpool")
COLORS = {"cule": "#0072B2", "envpool": "#D55E00"}
DISPLAY = {"cule": "CuLE", "envpool": "EnvPool"}
CONFIG_RE = re.compile(r"n(?P<num_envs>\d+)_s(?P<num_steps>\d+)_b(?P<batch>\d+).*_lr(?P<lr>\d+)")
THRESHOLDS = (100, 200, 400, 600)


def read_curve(path: Path) -> list[dict[str, float]]:
    with path.open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    return [
        {key: float(value) for key, value in row.items() if key != "algorithm"}
        for row in rows
    ]


def load_curves() -> tuple[dict, dict]:
    curves: dict[tuple[int, str], list[dict[str, float]]] = {}
    configs: dict[int, dict[str, float | int | str]] = {}
    for path in sorted(CURVE_DIR.glob("*.csv")):
        _, backend, config_name, *_ = path.stem.split("__")
        match = CONFIG_RE.fullmatch(config_name)
        if not match:
            raise ValueError(f"cannot parse configuration from {path.name}")
        num_envs = int(match.group("num_envs"))
        curves[(num_envs, backend)] = read_curve(path)
        configs[num_envs] = {
            "config": config_name,
            "num_envs": num_envs,
            "num_steps": int(match.group("num_steps")),
            "batch_size": int(match.group("batch")),
            "learning_rate": int(match.group("lr")) / 100_000,
        }

    missing = [
        (num_envs, backend)
        for num_envs in ENV_COUNTS
        for backend in BACKENDS
        if (num_envs, backend) not in curves
    ]
    if missing:
        raise FileNotFoundError(f"missing curves: {missing}")
    for key, curve in curves.items():
        if len(curve) != 10:
            raise ValueError(f"expected 10 checkpoints for {key}, found {len(curve)}")
    return curves, configs


def first_threshold(curve: list[dict[str, float]], threshold: float) -> dict[str, float] | None:
    return next((row for row in curve if row["reward_mean"] >= threshold), None)


def summarize(curves: dict, configs: dict) -> list[dict[str, float | int | str]]:
    rows = []
    for num_envs in ENV_COUNTS:
        for backend in BACKENDS:
            curve = curves[(num_envs, backend)]
            rewards = np.array([row["reward_mean"] for row in curve])
            peak_index = int(np.argmax(rewards))
            final = curve[-1]
            row: dict[str, float | int | str] = {
                **configs[num_envs],
                "backend": backend,
                "peak_reward": rewards[peak_index],
                "peak_frames": int(curve[peak_index]["frames"]),
                "peak_training_seconds": curve[peak_index]["training_seconds"],
                "final_reward": final["reward_mean"],
                "final_median": final["reward_median"],
                "final_std": final["reward_std"],
                "final_length_mean": final["length_mean"],
                "final_length_median": final["length_median"],
                "final_frames": int(final["frames"]),
                "final_training_seconds": final["training_seconds"],
                "worker_wall_seconds": final["worker_wall_seconds"],
                "train_sps": final["frames"] / final["training_seconds"],
                "curve_reward_mean": rewards.mean(),
                "peak_to_final_change": final["reward_mean"] - rewards[peak_index],
            }
            for threshold in THRESHOLDS:
                reached = first_threshold(curve, threshold)
                row[f"frames_to_{threshold}"] = int(reached["frames"]) if reached else ""
                row[f"seconds_to_{threshold}"] = reached["training_seconds"] if reached else ""
            rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_pair_summary(rows: list[dict]) -> list[dict]:
    by_key = {(int(row["num_envs"]), row["backend"]): row for row in rows}
    pairs = []
    for num_envs in ENV_COUNTS:
        cule = by_key[(num_envs, "cule")]
        envpool = by_key[(num_envs, "envpool")]
        pairs.append(
            {
                "num_envs": num_envs,
                "num_steps": cule["num_steps"],
                "batch_size": cule["batch_size"],
                "learning_rate": cule["learning_rate"],
                "cule_final_reward": cule["final_reward"],
                "envpool_final_reward": envpool["final_reward"],
                "cule_minus_envpool_final": float(cule["final_reward"])
                - float(envpool["final_reward"]),
                "cule_peak_reward": cule["peak_reward"],
                "envpool_peak_reward": envpool["peak_reward"],
                "cule_curve_mean": cule["curve_reward_mean"],
                "envpool_curve_mean": envpool["curve_reward_mean"],
                "cule_train_sps": cule["train_sps"],
                "envpool_train_sps": envpool["train_sps"],
                "cule_over_envpool_sps": float(cule["train_sps"])
                / float(envpool["train_sps"]),
            }
        )
    return pairs


def plot_learning(
    curves: dict,
    configs: dict,
    *,
    x_key: str,
    divisor: float,
    xlabel: str,
    output_name: str,
    title_suffix: str,
) -> Path:
    upper_reward = max(
        row["reward_mean"] + row["reward_std"]
        for curve in curves.values()
        for row in curve
    )
    upper_reward = max(100, int(np.ceil(upper_reward / 100)) * 100)
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(13.5, 9.5),
        sharex=True,
        sharey=True,
    )
    for axis, num_envs in zip(axes.flat, ENV_COUNTS):
        config = configs[num_envs]
        for backend in BACKENDS:
            curve = curves[(num_envs, backend)]
            x = np.array([row[x_key] / divisor for row in curve])
            mean = np.array([row["reward_mean"] for row in curve])
            std = np.array([row["reward_std"] for row in curve])
            axis.plot(
                x,
                mean,
                color=COLORS[backend],
                marker="o",
                linewidth=2.2,
                markersize=4.5,
                label=DISPLAY[backend],
            )
            axis.fill_between(
                x,
                np.maximum(0, mean - std),
                mean + std,
                color=COLORS[backend],
                alpha=0.12,
            )
        axis.set_title(
            f"{num_envs:,} training envs — batch {int(config['batch_size']):,}, "
            f"lr {float(config['learning_rate']):.1e}",
            fontweight="bold",
        )
        axis.grid(alpha=0.25)

    maximum_x = max(
        row[x_key] / divisor for curve in curves.values() for row in curve
    )
    if x_key == "frames":
        axes[0, 0].set_xlim(0.8, maximum_x * 1.02)
    else:
        axes[0, 0].set_xlim(0, maximum_x * 1.03)

    axes[0, 0].set_ylim(0, upper_reward)

    for axis in axes[-1]:
        axis.set_xlabel(xlabel)
    for axis in axes[:, 0]:
        axis.set_ylabel("Full-game reward (mean of 64)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.subplots_adjust(top=0.86, bottom=0.10, left=0.08, right=0.98, hspace=0.24, wspace=0.08)
    figure.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.925))
    figure.suptitle(
        f"Compiled PPO on Breakout: matched CuLE vs EnvPool — {title_suffix}",
        fontsize=15,
        fontweight="bold",
        y=0.975,
    )
    figure.text(
        0.5,
        0.025,
        "Shading: ±1 SD across 64 parallel evaluation games; one training seed per curve.",
        ha="center",
        fontsize=10,
    )
    output = PLOT_DIR / output_name
    figure.savefig(output, dpi=200)
    plt.close(figure)
    return output


def main() -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    curves, configs = load_curves()
    rows = summarize(curves, configs)
    pairs = make_pair_summary(rows)
    write_csv(CURVE_SUMMARY, rows)
    write_csv(PAIR_SUMMARY, pairs)
    transition_plot = plot_learning(
        curves,
        configs,
        x_key="frames",
        divisor=1_000_000,
        xlabel="Agent transitions (millions)",
        output_name="paired_learning_curves.png",
        title_suffix="by transitions",
    )
    time_plot = plot_learning(
        curves,
        configs,
        x_key="training_seconds",
        divisor=60,
        xlabel="Measured training time (minutes)",
        output_name="paired_learning_vs_training_time.png",
        title_suffix="by training time",
    )
    print(f"validated runs: {len(rows)}")
    print(f"evaluation points: {sum(len(curve) for curve in curves.values())}")
    print(f"transition figure: {transition_plot}")
    print(f"training-time figure: {time_plot}")
    print(f"curve summary: {CURVE_SUMMARY}")
    print(f"pair summary: {PAIR_SUMMARY}")


if __name__ == "__main__":
    main()
