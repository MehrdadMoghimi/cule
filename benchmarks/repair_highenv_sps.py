#!/usr/bin/env python3
"""Re-derive every highenv trial's throughput from its stored log.

The sweep that produced `benchmark_results/artifacts/breakout/highenv/` ran on
2026-08-15, a day before `sweep_highenv.py` learned that an on-policy compiled
twin's progress bar ticks once per *rollout* rather than once per vector step.
Trials from that run therefore divided the transition count of every such
trainer by its `num_steps`, and the SPS column of the report -- along with the
per-gradient-step cost table regressed from it -- inherited the error.

Nothing here re-runs anything: the complete stdout of all 385 trials is on disk,
so the parse can simply be redone with the correct scale and the two answers
compared.  Writes `highenv_sps_corrections.csv` under
`benchmark_results/exploration/` and leaves the original artifacts untouched.
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from breakout_algorithms import BY_KEY, ON_POLICY
from sweep_highenv import (
    FRAMES_RE, PROGRESS_RE, SIMPLE_RE, STATS_RE, TQDM_BAR_RE, TQDM_RET_RE,
    ANCHOR_RE, Knobs, args_fields, trainer_path,
)

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "benchmark_results" / "artifacts" / "breakout" / "highenv"
OUT = ROOT / "benchmark_results" / "exploration" / "highenv_sps_corrections.csv"


def bar_scale_for(record: dict) -> int:
    """The transitions one progress-bar tick represents, computed as the fixed code does."""
    algo = BY_KEY[record["algorithm"]]
    scale = record["num_envs"]
    if algo.family == ON_POLICY:
        path = trainer_path(algo, record["variant"])
        knobs = record.get("knobs", {})
        scale *= knobs.get("num_steps") or args_fields(path).get("num_steps") or 1
    return scale


def reparse(log: Path, bar_scale: int) -> dict:
    """Replay run_trial's line parsing over a stored log.

    Only the step counter is recovered.  The wall clock is not in the log, so
    the corrected SPS is formed against the trial's own recorded
    `trained_seconds`, which the bug did not touch.
    """
    last_step = 0
    reported: list[float] = []
    for raw in re.split(rb"[\r\n]", log.read_bytes()):
        line = raw.decode("utf-8", "replace")
        if not line.strip():
            continue
        anchor = ANCHOR_RE.search(line)
        if anchor:
            digits = re.search(r"([\d.]+)", anchor.group(0))
            if digits:
                reported.append(float(digits.group(1)))
        step = None
        match = STATS_RE.search(line) or SIMPLE_RE.search(line)
        if match:
            step = int(match.group(1))
        else:
            match = TQDM_RET_RE.search(line)
            if match and "sps" in line:
                progress, bar = PROGRESS_RE.search(line), TQDM_BAR_RE.search(line)
                if progress:
                    step = int(progress.group(1))
                elif bar:
                    step = int(bar.group(1)) * bar_scale
        if step is None:
            progress = PROGRESS_RE.search(line)
            frames = FRAMES_RE.search(line)
            bar = TQDM_BAR_RE.search(line)
            if progress:
                step = int(progress.group(1))
            elif frames:
                step = int(float(frames.group(1)))
            elif bar and "sps" in line.lower():
                step = int(bar.group(1)) * bar_scale
        if step is not None:
            last_step = max(last_step, step)
    return {"final_step": last_step, "reported_sps": reported[-1] if reported else None}


def main() -> None:
    records = [json.loads(line) for line in (ART / "trials.jsonl").read_text().splitlines()
               if line.strip()]
    rows, changed = [], 0
    for record in records:
        log = ART / "logs" / f"{record['trial_id']}.log"
        if not log.exists() or not record.get("trained_seconds"):
            continue
        scale = bar_scale_for(record)
        fresh = reparse(log, scale)
        trained = record["trained_seconds"]
        corrected = round(fresh["final_step"] / trained) if trained > 1 else None
        old = record.get("train_sps")
        if not corrected or not old:
            continue
        ratio = corrected / old
        if abs(ratio - 1.0) > 0.05:
            changed += 1
        rows.append({
            "trial_id": record["trial_id"],
            "algorithm": record["algorithm"],
            "variant": record["variant"],
            "label": record["label"],
            "stage": record["stage"],
            "score": record.get("score"),
            "bar_scale": scale,
            "recorded_sps": old,
            "corrected_sps": corrected,
            "ratio": round(ratio, 3),
            "trainer_reported_sps": fresh["reported_sps"],
        })

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    big = sorted((r for r in rows if abs(r["ratio"] - 1.0) > 0.2),
                 key=lambda r: -abs(r["ratio"] - 1.0))
    print(f"{len(rows)} trials reparsed, {changed} differ by >5%, {len(big)} by >20%\n")
    print(f"{'trial':56s} {'recorded':>9s} {'corrected':>10s} {'x':>6s} {'trainer says':>12s}")
    for r in big[:25]:
        says = f"{r['trainer_reported_sps']:.0f}" if r["trainer_reported_sps"] else "-"
        print(f"{r['trial_id'][:56]:56s} {r['recorded_sps']:9d} {r['corrected_sps']:10d} "
              f"{r['ratio']:6.2f} {says:>12s}")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
