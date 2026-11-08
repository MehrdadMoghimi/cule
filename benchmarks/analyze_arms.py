#!/usr/bin/env python3
"""Compare treatment arms against a baseline arm, per game and per trainer.

Used by the schedule experiments (H024, H025, H026) and the clipping experiment
(H022).  Every contrast is a log2 ratio of score ABOVE THE RANDOM BASELINE,
formed within a seed and then averaged across seeds -- the same discipline
`analyze_slopes.py` uses and for the same reason: a seed that finishes below the
baseline has no defined ratio, and averaging the scores first hides that instead
of reporting it.

    python benchmarks/analyze_arms.py h024_ratio_schedule const8
    python benchmarks/analyze_arms.py h026_schedule_generality const8 --by algorithm
"""

from __future__ import annotations

import json, math, statistics, sys
from pathlib import Path

EXP = Path(__file__).resolve().parents[1] / "benchmark_results" / "exploration"


def random_returns():
    out = {}
    for line in (EXP / "game_descriptors.jsonl").open():
        if line.strip():
            r = json.loads(line)
            if "error" not in r:
                out[r["game"]] = r["random_return"]
    return out


def load(experiment):
    cells = {}
    path = EXP / experiment / "runs.jsonl"
    for line in path.open():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("status") != "ok" or r.get("score") is None:
            continue
        key = (r.get("algorithm", "?"), r["game"], r["arm"])
        cells.setdefault(key, {})[r["seed"]] = r["score"]
    return cells


def contrast(cells, algo, game, arm, base, rand):
    a, b = cells.get((algo, game, arm), {}), cells.get((algo, game, base), {})
    vals = []
    for s in sorted(set(a) & set(b)):
        x, y = b[s] - rand, a[s] - rand
        if x > 0 and y > 0:
            vals.append(math.log2(y / x))
    if not vals:
        return None
    err = statistics.stdev(vals) / math.sqrt(len(vals)) if len(vals) > 1 else None
    return {"mean": statistics.fmean(vals), "n": len(vals), "err": err}


def main() -> int:
    experiment = sys.argv[1]
    base = sys.argv[2] if len(sys.argv) > 2 else "const8"
    cells = load(experiment)
    rand = random_returns()
    algos = sorted({k[0] for k in cells})
    arms = [a for a in dict.fromkeys(k[2] for k in cells) if a != base]

    totals = {a: [] for a in arms}
    for algo in algos:
        games = sorted({g for (al, g, _) in cells if al == algo})
        print(f"\n--- {algo} ---")
        print(f"{'game':<15}{'n':>3}  " + "".join(f"{a + ' vs ' + base:>26}" for a in arms))
        per_algo = {a: [] for a in arms}
        for game in games:
            n = len(cells.get((algo, game, base), {}))
            row = f"{game:<15}{n:>3}  "
            for arm in arms:
                c = contrast(cells, algo, game, arm, base, rand[game])
                if c is None:
                    row += f"{'-':>26}"
                    continue
                totals[arm].append(c["mean"])
                per_algo[arm].append(c["mean"])
                e = f"+/-{c['err']:.2f}" if c["err"] is not None else "  ~   "
                star = "*" if c["err"] and abs(c["mean"]) > 2 * c["err"] else " "
                row += f"{c['mean']:>+17.2f} {e}{star}"
            print(row)
        for arm in arms:
            v = per_algo[arm]
            if v:
                print(f"  {arm:<12} mean {statistics.fmean(v):+.3f} ({2 ** statistics.fmean(v):.2f}x)"
                      f"  better in {sum(1 for x in v if x > 0)}/{len(v)}")

    print(f"\n=== pooled over {len(algos)} trainer(s) ===")
    for arm in arms:
        v = totals[arm]
        if v:
            wins = sum(1 for x in v if x > 0)
            print(f"{arm:<12} mean {statistics.fmean(v):+.3f} ({2 ** statistics.fmean(v):.2f}x)"
                  f"  better than {base} in {wins}/{len(v)} cells")
    print("\n(* = separated from zero by twice its standard error; ~ = single seed)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
