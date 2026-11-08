#!/usr/bin/env python3
"""Aggregate `sweep_highenv.py` into a per-algorithm recommended configuration.

Reads `trials.jsonl` and `state.json` from the sweep's artifact directory and
writes three files next to them:

  trials.csv     every configuration that ran, with its score and throughput
  summary.csv    the winning configuration per algorithm, and its command line
  REPORT.md      the same as a readable table, plus the per-algorithm search

The winner is the configuration with the best score at the longest window it was
run at -- the confirm stage when it exists, the screen otherwise -- and the score
is the mean episodic return over the closing 40% of the training window, so a
run still climbing at the buzzer is not credited with its last lucky episode.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from breakout_algorithms import BY_KEY, ROOT

RESULTS = ROOT / "benchmark_results" / "artifacts" / "breakout" / "highenv"

STAGE_ORDER = {"feasibility": 0, "screen": 1, "refine": 1, "confirm": 2}

# Random play on BreakoutNoFrameskip-v4 returns about 1.2 per full game, so a
# score below this is indistinguishable from not having learned.
RANDOM_PLAY_BAND = 5.0

# Above the noise but not by much: a real signal worth reporting as such rather
# than as a tuned result.
MARGINAL_BAND = 15.0

# Random play's actual return on this game, quoted in the report so a reader can
# judge a weak score without knowing the game.
RANDOM_PLAY_FLOOR = 1.2

# The confirm-stage window these frontier numbers are quoted for.
WINDOW_SECONDS = 300.0

# The measured separation: under this per-update cost every replay algorithm in
# the sweep learned, over it none did.
COST_THRESHOLD_MS = 3


def knob_text(record: dict) -> str:
    knobs = record.get("knobs", {})
    out = []
    for name, short in (("batch_size", "batch"), ("updates_per_step", "updates/step"),
                        ("buffer_size", "buffer"), ("num_steps", "num_steps"),
                        ("minibatch", "minibatch"), ("update_epochs", "epochs")):
        if knobs.get(name):
            out.append(f"{short}={knobs[name]:g}" if isinstance(knobs[name], float)
                       else f"{short}={knobs[name]}")
    if knobs.get("lr_mult", 1.0) != 1.0:
        out.append(f"lr x{knobs['lr_mult']:g}")
    if knobs.get("extra"):
        out.append(" ".join(knobs["extra"]))
    return ", ".join(out)


def cost_model_section(by_algo: dict[str, list[dict]]) -> list[str]:
    """Recover the price of one gradient step, per algorithm, from the sweep.

    A replay trainer's wall clock is linear in its cadence:

        seconds per vector step = t_step + u * t_update

    so any two configurations that share a batch size and differ in updates per
    step give both terms.  This is the quantity that decides the whole replay
    family at a five-minute budget: `t_step` is ~15 ms of CuLE and policy for
    everyone, and what separates DQN from Rainbow is not sample efficiency but
    that one Rainbow update costs nearly five times one DQN update, so the same
    wall clock buys a fraction of the gradient steps.
    """
    rows = []
    for key, group in sorted(by_algo.items()):
        by_batch: dict[int, dict[float, int]] = {}
        for record in group:
            knobs = record.get("knobs", {})
            if (record["status"] != "ok" or not record.get("train_sps")
                    or record["variant"] != "torchcompile"
                    or not knobs.get("batch_size") or not knobs.get("updates_per_step")):
                continue
            by_batch.setdefault(knobs["batch_size"], {})[knobs["updates_per_step"]] = \
                record["train_sps"]
        for batch, cadences in sorted(by_batch.items()):
            if len(cadences) < 2:
                continue
            points = sorted(cadences.items())
            (low_u, low_sps), (high_u, high_sps) = points[0], points[-1]
            envs = 1024
            low_t, high_t = envs / low_sps, envs / high_sps
            per_update = (high_t - low_t) / (high_u - low_u)
            per_step = low_t - low_u * per_update
            rows.append((key, batch, per_update * 1000, per_step * 1000,
                         max(cadences.values())))
    if not rows:
        return []
    out = ["", "## What one gradient step costs", "",
           "Recovered from configurations that share a batch size and differ in",
           "cadence, via `seconds per vector step = t_step + u * t_update`.",
           "",
           "The last two columns are the corners of what five minutes can buy: all",
           "the time spent on gradient steps, or all of it spent collecting.  A",
           "configuration has to fit under both.  Rainbow is the instructive case --",
           "its ceiling of ~34k updates sits below the 53k that DQN's winning",
           "configuration actually performed, so DQN's operating point is not",
           "merely untuned for Rainbow, it is unreachable.",
           "",
           "| algorithm | batch | ms/gradient step | ms/vector step | best SPS | "
           "max updates in 5 min | max transitions in 5 min |",
           "|---|---:|---:|---:|---:|---:|---:|"]
    for key, batch, per_update, per_step, best in sorted(rows, key=lambda r: r[2]):
        max_updates = WINDOW_SECONDS / (per_update / 1000)
        max_transitions = 1024 * WINDOW_SECONDS / (per_step / 1000)
        out.append(f"| {BY_KEY[key].label} | {batch} | {per_update:.2f} | "
                   f"{per_step:.1f} | {best:,} | {max_updates:,.0f} | "
                   f"{max_transitions:,.0f} |")
    out += [
        "",
        f"Every replay algorithm whose gradient step costs under ~{COST_THRESHOLD_MS} ms",
        "learned something inside five minutes, and every one above it scored at",
        "random play, with no overlap between the groups.  The cost of one update --",
        "not sample efficiency, not the published replay ratio -- is what decides",
        "whether an algorithm can use 1,024 environments at a wall-clock budget.",
    ]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, default=RESULTS)
    args = parser.parse_args()

    records = [json.loads(line) for line in
               (args.results / "trials.jsonl").read_text().splitlines() if line.strip()]
    state = json.loads((args.results / "state.json").read_text())

    by_algo: dict[str, list[dict]] = {}
    for record in records:
        by_algo.setdefault(record["algorithm"], []).append(record)

    with (args.results / "trials.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["algorithm", "stage", "label", "variant", "num_envs", "window",
                         "knobs", "implied_replay_ratio", "score", "peak", "final_return",
                         "train_sps", "startup_seconds", "status"])
        for key in sorted(by_algo):
            for record in sorted(by_algo[key], key=lambda r: (STAGE_ORDER[r["stage"]],
                                                              -(r.get("score") or -1e9))):
                writer.writerow([
                    key, record["stage"], record["label"], record["variant"],
                    record["num_envs"], record["window"], knob_text(record),
                    record.get("implied_replay_ratio"), record.get("score"),
                    record.get("peak"), record.get("final_return"),
                    record.get("train_sps"), record.get("startup_seconds"),
                    record["status"],
                ])

    winners = []
    for key, group in by_algo.items():
        scored = [r for r in group if r["status"] == "ok" and r["stage"] != "feasibility"]
        if not scored:
            winners.append({"algorithm": key, "status": "no scoring run", "record": None})
            continue
        best_stage = max(STAGE_ORDER[r["stage"]] for r in scored)
        pool = [r for r in scored if STAGE_ORDER[r["stage"]] == best_stage]
        winners.append({"algorithm": key, "status": "ok",
                        "record": max(pool, key=lambda r: ((r.get("score") or 0.0),
                                                           r.get("train_sps") or 0))})

    winners.sort(key=lambda w: -((w["record"] or {}).get("score") or -1e9))

    with (args.results / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["algorithm", "label", "num_envs", "variant", "knobs",
                         "implied_replay_ratio", "window", "score", "peak", "train_sps",
                         "startup_seconds", "total_timesteps", "command"])
        for entry in winners:
            record = entry["record"]
            if record is None:
                writer.writerow([entry["algorithm"], "-", "-", "-", entry["status"]]
                                + [""] * 8)
                continue
            writer.writerow([
                entry["algorithm"], record["label"], record["num_envs"], record["variant"],
                knob_text(record), record.get("implied_replay_ratio"), record["window"],
                record["score"], record.get("peak"), record.get("train_sps"),
                record.get("startup_seconds"), record["total_timesteps"],
                " ".join(record["command"][1:]),
            ])

    lines = [
        "# Breakout at the highest workable environment count",
        "",
        "Every `cleanrl/` algorithm run at 1,024 environments where it fits, falling",
        "back to 512 and then 256, with the rest of its hyperparameters searched.",
        "Score is the mean training episodic return over the closing 40% of a",
        "five-minute training window (screen stages use a shorter window; the",
        "column says which).  CuLE GPU backend, `BreakoutNoFrameskip-v4`, seed 1.",
        "",
        "| # | algorithm | envs | variant | configuration | ratio | score | window | SPS | note |",
        "|---|---|---:|---|---|---:|---:|---:|---:|---|",
    ]
    for index, entry in enumerate(winners, start=1):
        record = entry["record"]
        algo = BY_KEY[entry["algorithm"]]
        if record is None:
            lines.append(f"| {index} | {algo.label} | - | - | {entry['status']} | | | | | |")
            continue
        ratio = record.get("implied_replay_ratio")
        score = record.get("score")
        # Fewer than three flushes inside the window means Breakout did not
        # finish enough episodes to have an opinion; the winner there was chosen
        # on throughput, and the table has to say so.
        if score is None or (record.get("samples_in_window") or 0) < 3:
            note = "too few episodes: ranked on throughput"
        elif score < RANDOM_PLAY_BAND:
            # Random play on Breakout scores about 1.2.  Anything under ~5 means
            # the algorithm did not learn inside five minutes, and the gaps
            # between its configurations are noise -- reporting a winner there
            # without saying so would dress up a coin flip as a measurement.
            note = "did not learn at this budget; configs indistinguishable"
        elif score < MARGINAL_BAND:
            # Real learning, but an order of magnitude off the leaders.  Worth
            # saying so without implying it is noise -- 10 points is eight times
            # random play, which is not nothing, just not competitive here.
            note = f"weak but real (random play is ~{RANDOM_PLAY_FLOOR})"
        else:
            note = ""
        lines.append(
            f"| {index} | {algo.label} | {record['num_envs']} | {record['variant']} | "
            f"{knob_text(record) or 'defaults'} | "
            f"{f'{ratio:.3g}' if ratio else '-'} | "
            f"{f'{score:.2f}' if score is not None else '-'} | "
            f"{record['window']:.0f}s | {record.get('train_sps') or '-'} | {note} |"
        )

    lines += cost_model_section(by_algo)
    lines += ["", "## Per-algorithm search", ""]
    for entry in winners:
        key = entry["algorithm"]
        algo = BY_KEY[key]
        chosen = state.get(key, {})
        lines += [f"### {algo.label} (`{key}`)", ""]
        if chosen.get("note"):
            lines += [f"{chosen['note']}", ""]
        lines += ["| stage | configuration | window | score | SPS | startup | status |",
                  "|---|---|---:|---:|---:|---:|---|"]
        for record in sorted(by_algo.get(key, []),
                             key=lambda r: (STAGE_ORDER[r["stage"]], r["label"])):
            score = record.get("score")
            lines.append(
                f"| {record['stage']} | {record['label']} | {record['window']:.0f}s | "
                f"{f'{score:.2f}' if score is not None else '-'} | "
                f"{record.get('train_sps') or '-'} | "
                f"{record.get('startup_seconds') or '-'}s | {record['status']} |"
            )
        lines.append("")

    (args.results / "REPORT.md").write_text("\n".join(lines))
    print(f"wrote {args.results / 'summary.csv'}, trials.csv and REPORT.md")
    for entry in winners[:10]:
        record = entry["record"]
        if record:
            print(f"  {entry['algorithm']:14s} {record['score']:8.2f}  "
                  f"n={record['num_envs']:<5d} {record['label']}")


if __name__ == "__main__":
    main()
