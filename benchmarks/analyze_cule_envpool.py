#!/usr/bin/env python3
"""Aggregate CuLE-vs-EnvPool scaling benchmarks into CSVs, a plot, and a report."""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cule")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "benchmark_results" / "artifacts" / "cule_envpool" / "cule_envpool_breakout_raw.jsonl"
ARTIFACT_DIR = ROOT / "benchmark_results" / "artifacts" / "cule_envpool"

BACKENDS = ("cule", "envpool")
COLORS = {"cule": "#0072B2", "envpool": "#D55E00"}
DISPLAY = {"cule": "CuLE", "envpool": "EnvPool"}


def load_records(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def failure_reason(record: dict) -> str:
    if record["status"] == "timeout":
        return "timeout"
    tail = record.get("output_tail", "")
    if "DefaultCPUAllocator: can't allocate memory" in tail:
        return "host OOM (rollout buffer)"
    if "CUDA out of memory" in tail:
        return "CUDA OOM"
    if "terminated by signal 9" in tail:
        return "OOM-killed (signal 9)"
    return f"error (rc={record.get('returncode')})"


def summarize_training(records: list[dict]) -> tuple[list[dict], list[dict]]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for record in records:
        if record["kind"] != "training":
            continue
        params = record["params"]
        groups[(params["algorithm"], params["backend"], params["num_envs"])].append(record)

    rows, failures = [], []
    for (algorithm, backend, num_envs), group in sorted(groups.items()):
        ok = [r for r in group if r["status"] == "ok"]
        for record in group:
            if record["status"] != "ok":
                failures.append(
                    {
                        "algorithm": algorithm,
                        "backend": backend,
                        "num_envs": num_envs,
                        "status": record["status"],
                        "reason": failure_reason(record),
                    }
                )
        if not ok:
            continue
        sps = [r["result"]["sps"] for r in ok]
        rows.append(
            {
                "algorithm": algorithm,
                "backend": backend,
                "num_envs": num_envs,
                "repeats": len(ok),
                "sps_mean": round(mean(sps), 1),
                "sps_std": round(stdev(sps), 1) if len(sps) > 1 else 0.0,
                "peak_cuda_memory_mb": round(max(r["result"]["peak_cuda_memory_mb"] for r in ok), 1),
                "max_rss_mb": round(max(r["max_rss_mb"] for r in ok), 1),
            }
        )
    return rows, failures


def summarize_probe(records: list[dict]) -> list[dict]:
    rows = []
    for record in records:
        if record["kind"] != "probe" or record["status"] != "ok":
            continue
        result = record["result"]
        rows.append(
            {
                "backend": result["backend"],
                "num_envs": result["num_envs"],
                "sps": round(result["sps"], 1),
                "rss_mb": round(result["rss_mb"], 1),
                "peak_cuda_memory_mb": round(result["peak_cuda_memory_mb"], 1),
            }
        )
    return sorted(rows, key=lambda row: (row["backend"], row["num_envs"]))


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def series(rows: list[dict], backend: str, **filters) -> tuple[list[int], list[float]]:
    key = "sps_mean" if "sps_mean" in (rows[0] if rows else {}) else "sps"
    points = sorted(
        (row["num_envs"], row[key])
        for row in rows
        if row["backend"] == backend and all(row.get(k) == v for k, v in filters.items())
    )
    return [p[0] for p in points], [p[1] for p in points]


def draw_panel(axis, title: str, rows: list[dict]) -> None:
    for backend in BACKENDS:
        envs, sps = series(rows, backend)
        if not envs:
            continue
        axis.plot(
            envs,
            sps,
            color=COLORS[backend],
            linewidth=2,
            marker="o",
            markersize=5,
            label=DISPLAY[backend],
        )
        axis.annotate(
            DISPLAY[backend],
            (envs[-1], sps[-1]),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=9,
            color="#333333",
        )
    ticks = sorted({row["num_envs"] for row in rows})
    axis.set_xscale("log", base=2)
    axis.set_xticks(ticks)
    axis.set_xticklabels([str(n) for n in ticks], rotation=45, fontsize=8)
    axis.set_title(title, fontsize=11)
    axis.set_xlabel("environments")
    axis.grid(True, axis="y", color="#dddddd", linewidth=0.6)
    axis.spines[["top", "right"]].set_visible(False)
    axis.margins(x=0.12)


def panels(training: list[dict], probe: list[dict]) -> list[tuple[str, str, list[dict]]]:
    return [
        ("ppo", "PPO training (compiled)", [r for r in training if r["algorithm"] == "ppo"]),
        ("pqn", "PQN training", [r for r in training if r["algorithm"] == "pqn"]),
        ("probe", "Environment stepping only", probe),
    ]


def plot_scaling(training: list[dict], probe: list[dict], output_dir: Path) -> dict[str, Path]:
    """Write the combined 3-panel figure plus one standalone PNG per panel."""
    specs = panels(training, probe)

    combined, axes = plt.subplots(1, 3, figsize=(13.5, 4.6), sharex=False)
    for axis, (_, title, rows) in zip(axes, specs):
        draw_panel(axis, title, rows)
    axes[0].set_ylabel("steps / second")
    axes[0].legend(frameon=False, fontsize=9, loc="upper left")
    combined.suptitle("CuLE vs EnvPool throughput scaling (Breakout-v5)", fontsize=13)
    combined.tight_layout(rect=(0, 0, 1, 0.95))
    combined.savefig(output_dir / "scaling.png", dpi=150)
    plt.close(combined)

    paths = {"combined": output_dir / "scaling.png"}
    for slug, title, rows in specs:
        if not rows:
            continue
        figure, axis = plt.subplots(figsize=(5.2, 4.2))
        draw_panel(axis, title, rows)
        axis.set_ylabel("steps / second")
        axis.legend(frameon=False, fontsize=9, loc="upper left")
        axis.set_title(f"CuLE vs EnvPool — {title}\nBreakout-v5", fontsize=11)
        figure.tight_layout()
        path = output_dir / f"scaling_{slug}.png"
        figure.savefig(path, dpi=150)
        plt.close(figure)
        paths[slug] = path
    return paths


def ratio_line(rows: list[dict], num_envs: int, **filters) -> str | None:
    values = {}
    for backend in BACKENDS:
        envs, sps = series(rows, backend, **filters)
        if num_envs in envs:
            values[backend] = sps[envs.index(num_envs)]
    if len(values) != 2:
        return None
    return f"{values['cule'] / values['envpool']:.2f}x"


def markdown_table(rows: list[dict], columns: list[str]) -> list[str]:
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    for row in rows:
        lines.append("| " + " | ".join(f"{row[c]:,}" if isinstance(row[c], (int, float)) else str(row[c]) for c in columns) + " |")
    return lines


def write_report(training: list[dict], probe: list[dict], failures: list[dict], inputs: list[Path], path: Path) -> None:
    ppo = [r for r in training if r["algorithm"] == "ppo"]
    pqn = [r for r in training if r["algorithm"] == "pqn"]
    sources = ", ".join(f"`{p.name}`" for p in inputs)
    lines = [
        "# CuLE vs EnvPool throughput scaling",
        "",
        f"Source: {sources} — Breakout-v5, compiled PPO (32 steps) and PQN"
        " (128 steps) full training loops, plus environment-only stepping probes.",
        "Mean steps/second over repeats; single measurement for probes.",
        "",
        "![scaling](scaling.png)",
        "",
        "Standalone panels: [PPO](scaling_ppo.png) · [PQN](scaling_pqn.png)"
        " · [env stepping](scaling_probe.png)",
        "",
        "## Headline ratios (CuLE / EnvPool)",
        "",
        "| envs | PPO training | PQN training | env stepping |",
        "|---|---|---|---|",
    ]
    for num_envs in sorted({r["num_envs"] for r in training} | {r["num_envs"] for r in probe}):
        cells = [
            ratio_line(ppo, num_envs) or "—",
            ratio_line(pqn, num_envs) or "—",
            ratio_line(probe, num_envs) or "—",
        ]
        lines.append(f"| {num_envs} | " + " | ".join(cells) + " |")

    lines += ["", "## PPO training", ""]
    lines += markdown_table(ppo, ["backend", "num_envs", "sps_mean", "sps_std", "peak_cuda_memory_mb", "max_rss_mb"])
    lines += ["", "## PQN training", ""]
    lines += markdown_table(pqn, ["backend", "num_envs", "sps_mean", "sps_std", "peak_cuda_memory_mb", "max_rss_mb"])
    lines += ["", "## Environment stepping (no learner)", ""]
    lines += markdown_table(probe, ["backend", "num_envs", "sps", "rss_mb", "peak_cuda_memory_mb"])

    if failures:
        deduped: dict[tuple, dict] = {}
        for failure in failures:
            key = (failure["algorithm"], failure["backend"], failure["num_envs"], failure["reason"])
            deduped.setdefault(key, {**failure, "count": 0})["count"] += 1
        lines += ["", "## Failed points", ""]
        lines += markdown_table(
            sorted(deduped.values(), key=lambda f: (f["algorithm"], f["backend"], f["num_envs"])),
            ["algorithm", "backend", "num_envs", "reason", "count"],
        )
        lines += [
            "",
            "PQN's host-side rollout buffer (`num_steps=128`) exceeds host RAM at",
            "2,048+ environments on both backends, so those points are a trainer",
            "limitation rather than a backend difference. PPO CuLE runs out of GPU",
            "memory beyond 4,096 environments (CuLE's frame buffers live on-device).",
        ]

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, nargs="+", default=[DEFAULT_INPUT])
    parser.add_argument("--output-dir", type=Path, default=ARTIFACT_DIR)
    parser.add_argument("--exclude-envs", type=int, nargs="*", default=[3072],
                        help="environment counts to drop from the report/figures")
    args = parser.parse_args()

    excluded = set(args.exclude_envs)
    records = [
        record
        for path in args.input
        for record in load_records(path)
        if record["params"].get("num_envs") not in excluded
    ]
    training, failures = summarize_training(records)
    probe = summarize_probe(records)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if training:
        write_csv(args.output_dir / "training_summary.csv", training)
    if probe:
        write_csv(args.output_dir / "probe_summary.csv", probe)
    plot_scaling(training, probe, args.output_dir)
    write_report(training, probe, failures, args.input, args.output_dir / "REPORT.md")
    print(f"wrote {args.output_dir / 'REPORT.md'}")


if __name__ == "__main__":
    main()
