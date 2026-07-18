#!/usr/bin/env python3
"""Aggregate the Breakout implementation benchmark JSONL into report CSVs."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import median


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "benchmark_results" / "artifacts"
RESULTS = ARTIFACTS / "implementation"
RAW = RESULTS / "implementation_breakout_raw.jsonl"
PRIOR_TRAINING = ARTIFACTS / "cule_envpool" / "cule_envpool_breakout_training.csv"


def result_sps(record: dict) -> float:
    return float(record["result"].get("sps", record["result"].get("fps")))


def latest_trials() -> list[dict]:
    latest = {}
    for line in RAW.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        key = json.dumps(record["params"], sort_keys=True)
        latest[key] = record
    return list(latest.values())


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def aggregate(records: list[dict], keys: tuple[str, ...]) -> list[dict]:
    groups = {}
    for record in records:
        group_key = tuple(record["params"].get(key) for key in keys)
        groups.setdefault(group_key, []).append(record)

    rows = []
    for group_key, trials in groups.items():
        successes = [trial for trial in trials if trial["status"] == "ok"]
        if not successes:
            continue
        sps_values = [result_sps(trial) for trial in successes]
        cuda_values = [float(trial["result"].get("peak_cuda_memory_mb", 0.0)) for trial in successes]
        rss_values = [float(trial["max_rss_mb"]) for trial in successes if "max_rss_mb" in trial]
        ups_values = [float(trial["result"]["ups"]) for trial in successes if "ups" in trial["result"]]
        row = dict(zip(keys, group_key))
        row.update(
            attempted_runs=len(trials),
            successful_runs=len(successes),
            success_rate=len(successes) / len(trials),
            median_sps=median(sps_values),
            min_sps=min(sps_values),
            max_sps=max(sps_values),
            peak_cuda_memory_mb=max(cuda_values),
            median_rss_mb=median(rss_values) if rss_values else 0.0,
            median_ups=median(ups_values) if ups_values else "",
        )
        rows.append(row)
    return rows


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    records = latest_trials()
    matched = aggregate(
        [record for record in records if record["params"]["profile"].startswith("matched-")],
        ("profile", "algorithm", "family", "variant", "num_envs"),
    )
    matched.sort(key=lambda row: (row["profile"], -row["median_sps"]))
    for row in matched:
        for key in ("success_rate", "median_sps", "min_sps", "max_sps", "peak_cuda_memory_mb", "median_rss_mb"):
            row[key] = f"{row[key]:.3f}"
        if row["median_ups"] != "":
            row["median_ups"] = f"{row['median_ups']:.3f}"
    matched_fields = [
        "profile",
        "algorithm",
        "family",
        "variant",
        "num_envs",
        "attempted_runs",
        "successful_runs",
        "success_rate",
        "median_sps",
        "min_sps",
        "max_sps",
        "median_ups",
        "peak_cuda_memory_mb",
        "median_rss_mb",
    ]
    write_csv(RESULTS / "implementation_breakout_matched.csv", matched_fields, matched)

    scaling = aggregate(
        [record for record in records if record["params"]["profile"] == "native-sweep"],
        ("algorithm", "variant", "num_envs"),
    )
    scaling.sort(key=lambda row: (row["algorithm"], row["num_envs"]))
    for row in scaling:
        for key in ("success_rate", "median_sps", "min_sps", "max_sps", "peak_cuda_memory_mb", "median_rss_mb"):
            row[key] = f"{row[key]:.3f}"
        if row["median_ups"] != "":
            row["median_ups"] = f"{row['median_ups']:.3f}"
    scaling_fields = [
        "algorithm",
        "variant",
        "num_envs",
        "attempted_runs",
        "successful_runs",
        "success_rate",
        "median_sps",
        "min_sps",
        "max_sps",
        "median_ups",
        "peak_cuda_memory_mb",
        "median_rss_mb",
    ]
    write_csv(RESULTS / "implementation_breakout_native_scaling.csv", scaling_fields, scaling)

    # Combine each native example's observed peak with the previous CuLE/EnvPool
    # PPO/PQN sweep.  These rows compare useful end-to-end configurations, not
    # identical learning workloads.
    numeric_scaling = aggregate(
        [record for record in records if record["params"]["profile"] == "native-sweep"],
        ("algorithm", "variant", "num_envs"),
    )
    peak_rows = []
    for algorithm in ("a2c", "ppo", "vtrace"):
        candidates = [row for row in numeric_scaling if row["algorithm"] == algorithm]
        best = max(candidates, key=lambda row: row["median_sps"])
        peak_rows.append(
            {
                "implementation": "CuLE examples",
                "algorithm": algorithm,
                "backend": "CuLE GPU",
                "compiled": "no",
                "num_envs": best["num_envs"],
                "median_sps": f"{best['median_sps']:.3f}",
                "successful_runs": best["successful_runs"],
                "peak_cuda_memory_mb": f"{best['peak_cuda_memory_mb']:.3f}",
            }
        )

    native_dqn = next(
        row
        for row in aggregate(
            [record for record in records if record["params"]["profile"] == "matched-dqn"],
            ("family", "variant", "num_envs"),
        )
        if row["family"] == "cule_examples" and row["variant"] == "native_eager"
    )
    peak_rows.append(
        {
            "implementation": "CuLE examples",
            "algorithm": "dqn",
            "backend": "CuLE GPU",
            "compiled": "no",
            "num_envs": native_dqn["num_envs"],
            "median_sps": f"{native_dqn['median_sps']:.3f}",
            "successful_runs": native_dqn["successful_runs"],
            "peak_cuda_memory_mb": f"{native_dqn['peak_cuda_memory_mb']:.3f}",
        }
    )

    with PRIOR_TRAINING.open(encoding="utf-8", newline="") as stream:
        prior = list(csv.DictReader(stream))
    for algorithm in ("ppo", "pqn"):
        for backend in ("cule", "envpool"):
            candidates = [row for row in prior if row["algorithm"] == algorithm and row["backend"] == backend]
            best = max(candidates, key=lambda row: float(row["median_sps"]))
            peak_rows.append(
                {
                    "implementation": "torchcompile trainer" if algorithm == "ppo" else "CleanRL PQN",
                    "algorithm": algorithm,
                    "backend": "CuLE GPU" if backend == "cule" else "EnvPool CPU",
                    "compiled": "yes" if algorithm == "ppo" else "no",
                    "num_envs": best["num_envs"],
                    "median_sps": best["median_sps"],
                    "successful_runs": best["repeats"],
                    "peak_cuda_memory_mb": best["peak_cuda_memory_mb"],
                }
            )
    peak_rows.sort(key=lambda row: -float(row["median_sps"]))
    write_csv(
        RESULTS / "implementation_breakout_peak_comparison.csv",
        [
            "implementation",
            "algorithm",
            "backend",
            "compiled",
            "num_envs",
            "median_sps",
            "successful_runs",
            "peak_cuda_memory_mb",
        ],
        peak_rows,
    )


if __name__ == "__main__":
    main()
