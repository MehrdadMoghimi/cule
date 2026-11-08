#!/usr/bin/env python3
"""Collect every number this campaign's report quotes, from the run records.

Nothing here is transcribed by hand: the report is generated from this, so a
figure in it and a row in `runs.jsonl` cannot drift apart.
"""

from __future__ import annotations

import json, math, statistics
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


def runs(experiment):
    out = {}
    path = EXP / experiment / "runs.jsonl"
    if not path.exists():
        return out
    for line in path.open():
        if line.strip():
            r = json.loads(line)
            if r.get("status") == "ok" and r.get("score") is not None:
                out.setdefault((r["game"], r["arm"]), {})[r["seed"]] = r["score"]
    return out


def contrast(cells, game, treat, base, rand):
    """Per-seed log2 ratio of above-random scores, treatment over base."""
    a, b = cells.get((game, treat), {}), cells.get((game, base), {})
    vals = []
    for s in sorted(set(a) & set(b)):
        x, y = b[s] - rand, a[s] - rand
        if x > 0 and y > 0:
            vals.append(math.log2(y / x))
    if not vals:
        return None
    return {"mean": statistics.fmean(vals), "n": len(vals),
            "err": statistics.stdev(vals) / math.sqrt(len(vals)) if len(vals) > 1 else None}


def main():
    d = descriptors()
    rand = {g: v["random_return"] for g, v in d.items()}
    out = {"descriptors": d}

    for exp, arms, base in (("h021_currency_by_game", ("data", "steps"), "base"),
                            ("h023_budget_tier", ("data", "steps"), "base"),
                            ("h022_reward_clipping", ("raw",), "clip"),
                            ("h024_ratio_schedule", ("ramp_up", "ramp_down"), "const8"),
                            ("h025_ramp_strength", ("ramp11", "ramp14", "ramp16"), "const8")):
        cells = runs(exp)
        games = sorted({g for g, _ in cells})
        block = {}
        for g in games:
            block[g] = {"scores": {a: cells.get((g, a), {}) for a in (base,) + arms},
                        "contrasts": {a: contrast(cells, g, a, base, rand[g]) for a in arms}}
        out[exp] = block
    json.dump(out, (EXP / "report_data.json").open("w"), indent=1)

    for exp in ("h021_currency_by_game", "h023_budget_tier",
                "h022_reward_clipping", "h024_ratio_schedule", "h025_ramp_strength"):
        block = out.get(exp) or {}
        n = sum(len(v["scores"][a]) for v in block.values() for a in v["scores"])
        print(f"{exp:<28} {len(block):>2} games, {n:>3} runs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
