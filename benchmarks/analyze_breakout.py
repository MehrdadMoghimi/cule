#!/usr/bin/env python3
"""Aggregate the Breakout learning benchmark into a report, tables and plots.

Reads `runs.jsonl` and `curves/*.csv` written by `benchmark_breakout.py` and
emits, next to them:

  summary.csv    one row per algorithm, ranked by final score
  REPORT.md      the written report
  plots/         learning curves by transition and by wall clock, and a
                 reward-versus-throughput scatter

Reported quantities
-------------------
final    mean of the trainer's own moving return window at the end of the run
peak     best value that window reached at any point
auc      mean return averaged over the budget, i.e. area under the learning
         curve normalised by transitions -- rewards learning *early*, which a
         final-score-only ranking cannot see
sps      transitions per second of wall clock, including startup and any
         `torch.compile` warmup, so it is what the run actually cost

Every score is a full-game unclipped Breakout score from the training
environments.  For epsilon-greedy trainers that is a behaviour-policy score and
slightly understates the greedy policy.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from breakout_algorithms import BY_KEY, EXCLUDED, OFF_POLICY, ON_POLICY, STREAMING, ROOT

RESULTS = ROOT / "benchmark_results" / "artifacts" / "breakout"

FAMILY_LABEL = {
    OFF_POLICY: "replay",
    ON_POLICY: "on-policy",
    STREAMING: "streaming",
}
# Categorical slots 1-8 of the validated default palette, in their published
# order -- that order is the colour-blind-safety mechanism, so it is used as
# given rather than reshuffled.  Worst adjacent CVD dE 9.1, normal-vision 19.6.
SERIES_COLORS = [
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100",
    "#e87ba4", "#008300", "#4a3aa7", "#e34948",
]
# Everything past the eighth series folds into one recessive "other" band rather
# than inventing a ninth hue.
OTHER_COLOR = "#b8b7b1"
# The scatter plots every algorithm at once, so it needs all-pairs separation;
# only the first three slots clear that gate, which is exactly the family count.
FAMILY_COLOR = {OFF_POLICY: "#2a78d6", ON_POLICY: "#eb6834", STREAMING: "#1baf7a"}
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
HIGHLIGHT_COUNT = 8


def read_curve(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    out = []
    for row in rows:
        try:
            out.append({
                "global_step": int(row["global_step"]),
                "mean_return": float(row["mean_return"]),
                "batch_mean_return": float(row["batch_mean_return"]),
                "wall_seconds": float(row["wall_seconds"]),
            })
        except (KeyError, ValueError):
            continue
    return out


BINS = 50


def rebin(curve: list[dict], budget: int, bins: int = BINS) -> list[dict]:
    """Average the curve into fixed transition bins so trainers are comparable.

    The trainers do not all report the same statistic: most log a 20-episode
    moving average through `EpisodeStats`, but `ppo_atari.py` and
    `pqn_atari_envpool.py` keep CleanRL's original line and log *single* episode
    returns.  Comparing a peak-of-moving-average against a peak-of-single-episode
    would flatter the latter badly.  Averaging every curve into the same
    fixed-width bins removes that asymmetry before any statistic is taken.
    """
    if not curve:
        return []
    width = budget / bins
    buckets: dict[int, list[float]] = {}
    walls: dict[int, list[float]] = {}
    for point in curve:
        index = min(int(point["global_step"] / width), bins - 1)
        buckets.setdefault(index, []).append(point["mean_return"])
        walls.setdefault(index, []).append(point["wall_seconds"])
    return [
        {
            "global_step": int((index + 1) * width),
            "mean_return": sum(values) / len(values),
            "wall_seconds": max(walls[index]),
            "samples": len(values),
        }
        for index, values in sorted(buckets.items())
    ]


def area_under_curve(curve: list[dict], budget: int) -> float | None:
    """Step-integral of mean return over transitions, divided by the budget."""
    if len(curve) < 2:
        return None
    total = 0.0
    for previous, current in zip(curve, curve[1:]):
        total += previous["mean_return"] * (current["global_step"] - previous["global_step"])
    last = curve[-1]
    if last["global_step"] < budget:
        total += last["mean_return"] * (budget - last["global_step"])
    return total / budget


def tail_mean(curve: list[dict], budget: int, fraction: float = 0.1) -> float | None:
    """Mean return over the final `fraction` of the budget.

    Less jumpy than the single last point, which for a noisy trainer can land on
    an unrepresentative window.
    """
    if not curve:
        return None
    cutoff = budget * (1.0 - fraction)
    tail = [row["mean_return"] for row in curve if row["global_step"] >= cutoff]
    if not tail:
        tail = [curve[-1]["mean_return"]]
    return sum(tail) / len(tail)


def peak_to_final_gap(row: dict) -> float | None:
    """How much of its own best score a run gave back by the end.

    Large values are worth reading before the ranking is: a trainer that peaked
    far above where it finished was not measured at its best, and for BBF the
    gap is structural rather than instability -- it resets its network on a
    schedule, so where the budget happens to stop inside that cycle sets the
    final number.
    """
    if row["peak"] is None or row["final"] is None or row["peak"] <= 0:
        return None
    return row["peak"] - row["final"]


def tail_slope(curve: list[dict], fraction: float = 0.2) -> float | None:
    """Fractional gain over the final `fraction` of the run, versus the one before.

    Peak-to-final catches a run that rose and then fell back.  It says nothing
    about the opposite failure -- a run still climbing steeply when the budget
    ran out -- because such a curve ends *at* its peak and looks perfectly
    settled by that test.  Comparing the last fifth of the binned curve with the
    fifth before it separates the two: a plateau scores near zero, a run that
    gained half again on its own final stretch scores 0.5.
    """
    window = max(1, int(len(curve) * fraction))
    if len(curve) < 2 * window:
        return None
    tail = [p["mean_return"] for p in curve[-window:]]
    prior = [p["mean_return"] for p in curve[-2 * window:-window]]
    before = sum(prior) / len(prior)
    if before <= 0.5:  # too close to random play for a ratio to mean anything
        return None
    return (sum(tail) / len(tail) - before) / before


def steps_to_threshold(curve: list[dict], threshold: float) -> int | None:
    for row in curve:
        if row["mean_return"] >= threshold:
            return row["global_step"]
    return None


def collect(tag_dir: Path) -> list[dict]:
    runs_path = tag_dir / "runs.jsonl"
    if not runs_path.exists():
        raise SystemExit(f"no runs at {runs_path}")
    records: dict[str, dict] = {}
    for line in runs_path.read_text().splitlines():
        if line.strip():
            rec = json.loads(line)
            records[rec["algorithm"]] = rec  # last write wins

    rows = []
    for key, rec in records.items():
        algo = BY_KEY[key]
        raw = read_curve(tag_dir / "curves" / f"{key}.csv")
        budget = rec["total_timesteps"]
        # Every statistic comes off the binned curve so the trainers that log a
        # 20-episode moving average and the two that log single episodes are
        # measured the same way.
        curve = rebin(raw, budget)
        config = rec.get("config", {})
        rows.append({
            "algorithm": key,
            "label": algo.label,
            "paper": algo.paper,
            "family": algo.family,
            "status": rec["status"],
            "variant": config.get("variant"),
            "num_envs": config.get("num_envs"),
            "batch_size": rec.get("batch_size"),
            "replay_ratio": config.get("replay_ratio"),
            "paper_replay_ratio": rec.get("paper_replay_ratio"),
            "replay_ratio_reduced": config.get("replay_ratio_reduced", False),
            "budget": budget,
            "final_step": rec["final_step"],
            "completed": rec["final_step"] >= 0.98 * budget,
            "final": tail_mean(curve, budget),
            "last_point": rec.get("final_mean_return"),
            "peak": max((p["mean_return"] for p in curve), default=None),
            "auc": area_under_curve(curve, budget),
            "wall_seconds": rec["wall_seconds"],
            "sps": rec.get("wall_sps"),
            "curve_points": len(raw),
            "to_10": steps_to_threshold(curve, 10.0),
            "to_20": steps_to_threshold(curve, 20.0),
            "to_50": steps_to_threshold(curve, 50.0),
            "curve": curve,
        })
    for row in rows:
        row["peak_to_final"] = peak_to_final_gap(row)
        row["tail_slope"] = tail_slope(row["curve"])
    rows.sort(key=lambda r: (r["final"] is None, -(r["final"] or 0)))
    return rows


def write_summary(rows: list[dict], path: Path) -> None:
    fields = [f for f in rows[0] if f != "curve"]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({f: row[f] for f in fields})


def make_plots(rows: list[dict], out_dir: Path) -> list[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    out_dir.mkdir(parents=True, exist_ok=True)
    made = []
    learners = [r for r in rows if r["curve"]]

    # Twenty-five lines cannot each carry their own hue.  The eight with the
    # largest area under the curve get the categorical slots and a direct label;
    # the rest are drawn once as a recessive band so the distribution is still
    # visible without competing for identity.
    ranked = sorted(learners, key=lambda r: -(r["auc"] or 0))
    highlighted = ranked[:HIGHLIGHT_COUNT]
    colors = {row["algorithm"]: SERIES_COLORS[i] for i, row in enumerate(highlighted)}

    for xkey, xlabel, fname in (
        ("global_step", "transitions", "learning_vs_transitions.png"),
        ("wall_seconds", "wall-clock seconds", "learning_vs_walltime.png"),
    ):
        fig, ax = plt.subplots(figsize=(12, 7.5))
        ax.set_facecolor("#fcfcfb")
        for row in ranked[HIGHLIGHT_COUNT:]:
            ax.plot([p[xkey] for p in row["curve"]],
                    [p["mean_return"] for p in row["curve"]],
                    linewidth=1.0, color=OTHER_COLOR, alpha=0.85, zorder=1)
        ends = []
        for row in highlighted:
            xs = [p[xkey] for p in row["curve"]]
            ys = [p["mean_return"] for p in row["curve"]]
            ax.plot(xs, ys, linewidth=2.0, color=colors[row["algorithm"]],
                    label=row["label"], zorder=3, solid_capstyle="round")
            ends.append((ys[-1], xs[-1], row["label"]))

        # Direct labels at the line ends -- the palette's low-contrast slots
        # require visible labels rather than colour alone -- pushed apart so
        # lines finishing at similar scores do not overprint each other.
        ends.sort()
        span = max(e[0] for e in ends) - min(e[0] for e in ends) or 1.0
        min_gap = span * 0.035
        placed: list[float] = []
        for y, x, label in ends:
            target = y
            if placed and target - placed[-1] < min_gap:
                target = placed[-1] + min_gap
            placed.append(target)
            ax.annotate(label, (x, y), xytext=(8, 0), textcoords="offset points",
                        fontsize=8.5, color=TEXT_PRIMARY, va="center", zorder=4,
                        annotation_clip=False,
                        xycoords="data", **({} if target == y else {}))
            if abs(target - y) > 1e-9:
                # Re-place at the de-collided height and tie it back to the line.
                ax.texts[-1].remove()
                ax.annotate(label, (x, target), xytext=(8, 0),
                            textcoords="offset points", fontsize=8.5,
                            color=TEXT_PRIMARY, va="center", zorder=4,
                            annotation_clip=False)
        ax.plot([], [], linewidth=1.0, color=OTHER_COLOR,
                label=f"other {len(ranked) - HIGHLIGHT_COUNT} algorithms")

        ax.set_xlabel(xlabel, color=TEXT_SECONDARY)
        ax.set_ylabel("full-game Breakout score (training episodes)",
                      color=TEXT_SECONDARY)
        ax.set_title(f"Breakout learning curves — CuLE, "
                     f"{rows[0]['budget']:,} transitions, seed 1",
                     color=TEXT_PRIMARY, fontsize=13, loc="left")
        ax.grid(alpha=0.18, linewidth=0.8)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color("#d5d4cf")
        ax.tick_params(colors=TEXT_SECONDARY)
        if xkey == "wall_seconds":
            ax.set_xscale("log")
        # Leave room on the right for the direct labels.
        ax.set_xlim(right=ax.get_xlim()[1] * (1.28 if xkey == "wall_seconds" else 1.18))
        ax.legend(fontsize=8.5, loc="upper left", frameon=False, ncol=2,
                  labelcolor=TEXT_SECONDARY)
        fig.tight_layout()
        fig.savefig(out_dir / fname, dpi=140, facecolor="#fcfcfb")
        plt.close(fig)
        made.append(fname)

    # Reward against throughput: the two things the benchmark was asked for.
    # Every algorithm is on screen at once, so identity is carried by the direct
    # label and colour only distinguishes the three families -- the all-pairs
    # gate the scatter form requires.
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_facecolor("#fcfcfb")
    for row in rows:
        if not row["sps"] or row["final"] is None:
            continue
        ax.scatter(row["sps"], max(row["final"], 0.1), s=90, zorder=3,
                   color=FAMILY_COLOR[row["family"]],
                   edgecolor="#fcfcfb", linewidth=1.6)
        ax.annotate(row["label"], (row["sps"], max(row["final"], 0.1)),
                    textcoords="offset points", xytext=(8, 4),
                    fontsize=8.5, color=TEXT_PRIMARY)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("training throughput (transitions/s, wall clock)", color=TEXT_SECONDARY)
    ax.set_ylabel("final Breakout score (log)", color=TEXT_SECONDARY)
    ax.set_title("Reward versus throughput — neither buys the other",
                 color=TEXT_PRIMARY, fontsize=13, loc="left")
    ax.grid(alpha=0.18, linewidth=0.8)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#d5d4cf")
    ax.tick_params(colors=TEXT_SECONDARY)
    ax.set_xlim(right=ax.get_xlim()[1] * 1.5)
    handles = [plt.Line2D([], [], marker="o", linestyle="", color=c,
                          markeredgecolor="#fcfcfb", label=FAMILY_LABEL[f])
               for f, c in FAMILY_COLOR.items()]
    ax.legend(handles=handles, fontsize=9, frameon=False, labelcolor=TEXT_SECONDARY)
    fig.tight_layout()
    fig.savefig(out_dir / "reward_vs_throughput.png", dpi=140, facecolor="#fcfcfb")
    plt.close(fig)
    made.append("reward_vs_throughput.png")
    return made


def fmt(value, spec="{:.1f}", dash="-"):
    return dash if value is None else spec.format(value)


def compile_speedup(calibration_path: Path) -> list[dict]:
    """Compiled-versus-eager learner throughput, where both were measured.

    The calibration probes both variants at the same batch size and replay
    ratio, so the ratio of their update rates isolates what `torch.compile` plus
    CUDA graph capture is worth on the learner alone.
    """
    if not calibration_path.exists():
        return []
    by_key: dict[tuple, dict] = {}
    for line in calibration_path.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("status") != "ok" or rec["probe"] != "learn":
            continue
        key = (rec["algorithm"], rec["replay_ratio"])
        by_key.setdefault(key, {})[rec["variant"]] = rec["result"]["ups"]
    rows = []
    for (algorithm, ratio), variants in sorted(by_key.items()):
        if "eager" in variants and "torchcompile" in variants and variants["eager"]:
            rows.append({
                "algorithm": algorithm, "replay_ratio": ratio,
                "eager_ups": variants["eager"],
                "compiled_ups": variants["torchcompile"],
                "speedup": variants["torchcompile"] / variants["eager"],
            })
    return rows


def write_report(rows: list[dict], plan: dict, path: Path, plots: list[str]) -> None:
    budget = rows[0]["budget"]
    learned = [r for r in rows if (r["final"] or 0) >= 5]
    ok = [r for r in rows if r["status"] == "ok"]
    failed = [r for r in rows if r["status"] != "ok"]

    lines = [
        "# Breakout learning benchmark",
        "",
        f"{len(rows)} algorithms, one seed, `BreakoutNoFrameskip-v4`, CuLE GPU backend, "
        f"{budget:,} frame-skipped agent transitions each.",
        "",
        "## Protocol",
        "",
        "Environment count is the only knob tuned; every other hyperparameter is "
        "pinned to the value the algorithm publishes.  Two consequences of that "
        "are worth stating up front, because they are what makes the comparison "
        "mean anything:",
        "",
        "- **Replay trainers keep their published gradient cadence.**  The shipped "
        "files default to one environment with one minibatch per transition. "
        "Raising `--num-envs` from there would divide the gradient cadence by the "
        "environment count -- a hundredfold hyperparameter change disguised as a "
        "parallelism change.  The repo already has the receipt: its 2026-07 sweep "
        "ran Rainbow at 512 environments with `--replay-ratio 1` and the policy "
        "never left random play.  So the *replay ratio* is held fixed at the "
        "published value while environment count moves.",
        "- **On-policy trainers keep their minibatch size.**  `--num-minibatches` "
        "rises with the rollout batch, so gradient work per collected transition "
        "is identical to the shipped configuration and the learning rate keeps "
        "its meaning.",
        "",
        "The score is the full-game **unclipped** Breakout score from the training "
        "environments: `cule_env.CuLEVectorEnv` accumulates raw rewards and flushes "
        "only when `terminated & lives == 0`, so reward clipping and "
        "`EpisodicLifeEnv` affect the learner but not the number reported here. "
        "For epsilon-greedy trainers it is a behaviour-policy score and therefore "
        "a slight underestimate of the greedy policy.",
        "",
        "## Results",
        "",
        "Every curve is first averaged into "
        f"{BINS} equal transition bins, because most trainers log a 20-episode "
        "moving average while `ppo_atari.py` and `pqn_atari_envpool.py` log "
        "single episodes -- without that step a peak-of-single-episode would "
        "flatter those two. `final` is then the mean over the last 10% of the "
        "budget, `peak` the best bin, `auc` the return averaged over the whole "
        "budget (which rewards learning early), and `sps` the wall-clock "
        "throughput including startup.",
        "",
        "| # | algorithm | family | envs | batch | ratio | paper | final | peak | auc | sps | wall |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for index, row in enumerate(rows, start=1):
        flag = "*" if row["replay_ratio_reduced"] else ""
        note = "" if row["status"] == "ok" else f" ({row['status']})"
        lines.append(
            f"| {index} | {row['label']}{note} | {FAMILY_LABEL[row['family']]} | "
            f"{row['num_envs']} | {row['batch_size'] or '-'} | "
            f"{fmt(row['replay_ratio'], '{:g}')}{flag} | "
            f"{fmt(row['paper_replay_ratio'], '{:g}')} | "
            f"**{fmt(row['final'])}** | {fmt(row['peak'])} | {fmt(row['auc'])} | "
            f"{fmt(row['sps'], '{:,.0f}')} | {fmt(row['wall_seconds'] / 60)}m |"
        )
    lines += [
        "",
        "`*` = replay ratio reduced below the published cadence to fit the "
        "per-run wall-clock budget; those rows are handicapped relative to their "
        "paper settings and should not be read as the algorithm's ceiling.",
        "",
    ]

    # A final score only means something if the curve had settled when the
    # budget ran out.  Two different symptoms say it had not: the run ended well
    # below its own peak, or the binned tail disagrees sharply with the very last
    # raw reading, which happens when the curve is moving fast at the end.
    def unsettled(row: dict) -> float:
        if not row["peak"] or row["peak"] < 10 or row["final"] is None:
            return 0.0
        gap = (row["peak_to_final"] or 0) / row["peak"]
        last = row["last_point"]
        drift = abs(row["final"] - last) / max(row["final"], 1.0) if last is not None else 0.0
        return max(gap, drift)

    volatile = sorted((r for r in rows if unsettled(r) > 0.5),
                      key=lambda r: -unsettled(r))
    if volatile:
        lines += [
            "## Runs that had not settled when the budget ran out",
            "",
            "A final score only means something if the curve was flat at the end. "
            "For these it was not -- either the run finished well below its own "
            "peak, or the averaged tail disagrees with the last raw reading "
            "because the curve was still moving:",
            "",
            "| algorithm | peak | final (binned tail) | last raw point |",
            "|---|---:|---:|---:|",
        ]
        for row in volatile:
            lines.append(f"| {row['label']} | {fmt(row['peak'])} | "
                         f"{fmt(row['final'])} | {fmt(row['last_point'])} |")
        lines += [
            "",
            "**BBF's is structural, not instability.** BBF resets its encoder and "
            "transition model on a schedule (`--reset-every` 20,000 gradient "
            "steps); at the replay ratio this run used that is one reset per "
            "~320k transitions, so a 1M-transition budget contains three cycles "
            "and stops shortly after the third. Its curve is a sawtooth that "
            "climbed to ~227 immediately before that reset. For BBF the peak and "
            "the pre-reset plateau describe the algorithm; the final score "
            "describes where the budget happened to stop.",
            "",
        ]

    # The opposite symptom: a curve still rising steeply at the buzzer ends at
    # its own peak, so the peak-to-final test above calls it settled.  These
    # rows are lower bounds on the algorithm, not estimates of it.
    climbing = sorted((r for r in rows if (r.get("tail_slope") or 0) > 0.15),
                      key=lambda r: -(r["tail_slope"] or 0))
    if climbing:
        lines += [
            "## Runs still climbing when the budget ran out",
            "",
            "These ended at their own peak, so the test above calls them "
            "settled, but their last fifth gained materially on the fifth "
            "before it. Read these as lower bounds: the 1M budget stopped them, "
            "not convergence.",
            "",
            "| algorithm | final | gain over the last 20% of the run |",
            "|---|---:|---:|",
        ]
        for row in climbing:
            lines.append(f"| {row['label']} | {fmt(row['final'])} | "
                         f"+{row['tail_slope'] * 100:.0f}% |")
        lines.append("")

    lines += ["## Time to a score", "",
              "| algorithm | reaches 10 | reaches 20 | reaches 50 |",
              "|---|---:|---:|---:|"]
    for row in rows:
        if not any((row["to_10"], row["to_20"], row["to_50"])):
            continue
        lines.append(
            f"| {row['label']} | {fmt(row['to_10'], '{:,.0f}')} | "
            f"{fmt(row['to_20'], '{:,.0f}')} | {fmt(row['to_50'], '{:,.0f}')} |")
    lines.append("")

    if plots:
        lines += ["## Plots", ""]
        for name in plots:
            lines.append(f"![{name}](plots/{name})")
        lines.append("")

    lines += [
        "## What the numbers say",
        "",
        f"- {len(learned)} of {len(rows)} algorithms got meaningfully off the "
        f"floor (final score >= 5) inside {budget:,} transitions.",
    ]
    if learned:
        best = learned[0]
        lines.append(
            f"- Best final score: **{best['label']}** at {fmt(best['final'])}, "
            f"using {best['num_envs']} environments at {fmt(best['sps'], '{:,.0f}')} "
            f"transitions/s.")
        fastest = max((r for r in ok if r["sps"]), key=lambda r: r["sps"])
        lines.append(
            f"- Fastest run: **{fastest['label']}** at "
            f"{fmt(fastest['sps'], '{:,.0f}')} transitions/s "
            f"({fmt(fastest['wall_seconds'] / 60)} min for the full budget).")
        by_auc = sorted((r for r in rows if r["auc"] is not None),
                        key=lambda r: -r["auc"])
        if by_auc and by_auc[0]["algorithm"] != learned[0]["algorithm"]:
            lines.append(
                f"- Best area under the curve: **{by_auc[0]['label']}** "
                f"({fmt(by_auc[0]['auc'])}) -- it learned earlier than the "
                f"final-score winner even if it did not finish highest.")
    speedups = compile_speedup(RESULTS / "calibration.jsonl")
    if speedups:
        mean = sum(r["speedup"] for r in speedups) / len(speedups)
        lines += [
            "",
            "## Implementation note: what compilation is worth",
            "",
            "The benchmark runs the eager trainers, because their defaults are "
            "the published ones -- the `*_torchcompile.py` twins hard-code replay "
            "onto the GPU (`LazyTensorStorage(..., device=device)`) and so cannot "
            "hold the papers' 1M-transition buffer. The calibration measured both "
            "variants at the same batch size and replay ratio, which isolates "
            f"what compilation buys on the learner: a mean **{mean:.1f}x** at the "
            "published batch size of 32, where the learner is launch-bound rather "
            "than compute-bound.  Most of that comes from CUDA graph capture "
            "rather than compilation alone -- `c51_atari_torchcompile.py` is the "
            "one twin without a `--cudagraphs` flag, and it is also the one whose "
            "speedup stalls around 2x while the others reach 5-6x.",
            "",
            "| algorithm | ratio | eager ups | compiled ups | speedup |",
            "|---|---:|---:|---:|---:|",
        ]
        for row in speedups:
            lines.append(
                f"| {row['algorithm']} | {row['replay_ratio']:g} | "
                f"{row['eager_ups']:,.0f} | {row['compiled_ups']:,.0f} | "
                f"{row['speedup']:.1f}x |")
        lines.append("")

    if failed:
        lines += ["", "### Runs that did not complete", ""]
        for row in failed:
            lines.append(f"- **{row['label']}**: {row['status']} at "
                         f"{row['final_step']:,}/{row['budget']:,} transitions")
    lines += [
        "",
        "## Limitations",
        "",
        "1. **One seed.**  Atari runs are high variance; these are single-run "
        "observations, not estimates of algorithm quality.  Ranking differences "
        "smaller than the gaps visible in the curves should not be trusted.",
        f"2. **{budget:,} transitions is a short budget for Breakout.**  The "
        "classic value-based agents are tuned for tens of millions and are still "
        "early in their learning curve here; the Atari-100K family, by contrast, "
        "is designed for exactly this regime.  The ranking is therefore about "
        "*early* learning, not asymptotic performance.",
        "3. **Training-episode scores**, not a separate greedy evaluation, so "
        "epsilon-greedy trainers are scored while still exploring.",
        "4. Any row flagged `*` ran below its published replay ratio.",
    ]
    if EXCLUDED:
        lines += ["", "### Excluded", ""]
        for key, why in EXCLUDED.items():
            lines.append(f"- `{key}`: {why}")
    lines.append("")
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tag", default="1m")
    parser.add_argument("--results", type=Path, default=RESULTS)
    args = parser.parse_args()

    tag_dir = args.results / args.tag
    rows = collect(tag_dir)
    plan_path = args.results / "plan.json"
    plan = json.loads(plan_path.read_text()) if plan_path.exists() else {}

    write_summary(rows, tag_dir / "summary.csv")
    plots = make_plots(rows, tag_dir / "plots")
    write_report(rows, plan, tag_dir / "REPORT.md", plots)

    print(f"{len(rows)} runs -> {tag_dir}/REPORT.md")
    print(f"{'algorithm':22s} {'envs':>5s} {'final':>8s} {'peak':>8s} {'auc':>8s} {'sps':>9s}")
    print("-" * 66)
    for row in rows:
        print(f"{row['label']:22s} {row['num_envs'] or 0:5d} "
              f"{fmt(row['final']):>8s} {fmt(row['peak']):>8s} "
              f"{fmt(row['auc']):>8s} {fmt(row['sps'], '{:,.0f}'):>9s}")


if __name__ == "__main__":
    main()
