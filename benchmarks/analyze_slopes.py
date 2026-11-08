#!/usr/bin/env python3
"""Turn H021/H023 cells into per-(game, trainer) slopes with uncertainties.

Coordinates, spelled out because they are easy to confuse with F012's:

    slope_data   d log2(score - random) per doubling of FRESH TRANSITIONS, with
                 optimizer steps and samples consumed both held fixed (the
                 budget doubles and the replay ratio halves, cancelling exactly)
    slope_steps  the same per doubling of REPLAY at fixed fresh transitions, so
                 optimizer steps and samples consumed both double together --
                 with the batch pinned those two are perfectly collinear and
                 this design cannot separate them, by construction
    R            slope_steps / slope_data

Scores are taken above the random-policy return from `game_descriptors.jsonl`,
because five of the six games pay a random agent a substantial score and a log
ratio of raw returns would be mostly a ratio of baselines.

Slopes are formed WITHIN a seed and then averaged across seeds, never the other
way round.  Averaging the scores first and taking one log ratio of the means is
what the first version of this script did, and it fails badly: on Krull one of
three DQN seeds finishes below the random baseline, so the mean above-random
score collapses toward zero and the log ratio acquires an error of +/-17.
Differencing within a seed also removes the variance the base and treatment runs
share through the same environment initialisation.

Uncertainty is the standard error over seeds where a cell is replicated, and
otherwise the spread of the closing window within the single run -- a floor on
the true seed-to-seed variance rather than an estimate of it, reported with `~`
to keep the distinction visible.  R is reported only where slope_data is
separated from zero by at least twice its error; a ratio to a slope consistent
with zero is not a number worth printing.
"""

from __future__ import annotations

import csv, json, math, statistics, sys
from pathlib import Path

EXP = Path(__file__).resolve().parents[1] / "benchmark_results" / "exploration"


def descriptors():
    out = {}
    for line in (EXP / "game_descriptors.jsonl").open():
        if line.strip():
            r = json.loads(line)
            if "error" not in r:
                out[r["game"]] = r
    return out


def cells(experiment):
    """(game, algo, arm) -> {'by_seed': {seed: score}, 'within': [cv...]}"""
    runs, out = {}, {}
    path = EXP / experiment / "runs.jsonl"
    for line in path.open():
        if line.strip():
            r = json.loads(line)
            runs[r["name"]] = r
    for r in runs.values():
        if r.get("status") != "ok" or r.get("score") is None:
            continue
        key = (r["game"], r["algorithm"], r["arm"])
        out.setdefault(key, {"by_seed": {}, "within": []})["by_seed"][r["seed"]] = r["score"]
        curve = EXP / experiment / "curves" / f"{r['name']}.csv"
        if curve.exists():
            pts = [(int(x["global_step"]), float(x["mean_return"]))
                   for x in csv.DictReader(curve.open()) if x["mean_return"] not in ("", "nan")]
            if len(pts) > 3:
                cut = max(s for s, _ in pts) * 0.75
                tail = [v for s, v in pts if s >= cut]
                m = statistics.fmean(tail)
                out[key]["within"].append(statistics.pstdev(tail) / max(abs(m), 1e-9))
    return out


def slopes(experiment, rand):
    grid = cells(experiment)
    rows = []
    for (game, algo, arm) in list(grid):
        if arm != "base":
            continue
        base = grid.get((game, algo, "base"))
        row = {"game": game, "algo": algo,
               "seeds": len(base["by_seed"]),
               "base_above": statistics.fmean(v - rand[game] for v in base["by_seed"].values())}
        for a, key in (("data", "slope_data"), ("steps", "slope_steps")):
            cell = grid.get((game, algo, a))
            per_seed = []
            if cell:
                for seed, score in cell["by_seed"].items():
                    b = base["by_seed"].get(seed)
                    if b is None:
                        continue
                    x, y = b - rand[game], score - rand[game]
                    if x > 0 and y > 0:
                        per_seed.append(math.log2(y / x))
            if per_seed:
                row[key] = statistics.fmean(per_seed)
                row[key + "_n"] = len(per_seed)
                if len(per_seed) > 1:
                    row[key + "_err"] = statistics.stdev(per_seed) / math.sqrt(len(per_seed))
                else:
                    # single seed: fall back to the within-run closing spread of
                    # both cells, which bounds rather than estimates the error
                    w = [statistics.fmean(c["within"]) for c in (base, cell) if c["within"]]
                    row[key + "_err"] = (math.hypot(*w) / math.log(2) if len(w) == 2
                                         else float("nan"))
                row[key + "_replicated"] = len(per_seed) > 1
            else:
                row[key] = row[key + "_err"] = None
                row[key + "_replicated"] = False
        sd, se = row.get("slope_data"), row.get("slope_data_err")
        row["R"] = (row["slope_steps"] / sd
                    if sd is not None and row.get("slope_steps") is not None
                    and se == se and se is not None and abs(sd) > 2 * se else None)
        if row.get("slope_data") is not None or row.get("slope_steps") is not None:
            rows.append(row)
    return rows


def report(experiment):
    rand = {g: d["random_return"] for g, d in descriptors().items()}
    rows = slopes(experiment, rand)
    print(f"\n=== {experiment} ===")
    print(f"{'game':<14}{'algo':<9}{'seeds':>6}{'base>rand':>11}"
          f"{'slope_data':>18}{'slope_steps':>18}{'R':>8}")
    for r in sorted(rows, key=lambda r: (r["algo"], r["game"])):
        def f(k):
            if r[k] is None:
                return f"{'-':>18}"
            e = r[k + "_err"]
            mark = " " if r.get(k + "_replicated") else "~"
            return f"{r[k]:>+11.2f}{mark}{e:>5.2f}" if e == e else f"{r[k]:>+11.2f}{'':>6}"
        R = f"{r['R']:>8.2f}" if r["R"] is not None else f"{'-':>8}"
        print(f"{r['game']:<14}{r['algo']:<9}{r['seeds']:>6}{r['base_above']:>11.1f}"
              f"{f('slope_data')}{f('slope_steps')}{R}")
    for algo in sorted({r["algo"] for r in rows}):
        sub = [r for r in rows if r["algo"] == algo
               and r["slope_data"] is not None and r["slope_steps"] is not None]
        if sub:
            sd = [r["slope_data"] for r in sub]
            ss = [r["slope_steps"] for r in sub]
            print(f"  {algo}: n={len(sub)}  slope_data mean {statistics.fmean(sd):+.2f} "
                  f"median {statistics.median(sd):+.2f}   "
                  f"slope_steps mean {statistics.fmean(ss):+.2f} "
                  f"median {statistics.median(ss):+.2f}")
    return rows


if __name__ == "__main__":
    out = {}
    for exp in (sys.argv[1:] or ["h021_currency_by_game"]):
        out[exp] = report(exp)
    json.dump(out, (EXP / "slopes.json").open("w"), indent=1)
