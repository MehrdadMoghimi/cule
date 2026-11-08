#!/usr/bin/env python3
"""Regenerate every quantitative claim in the exploration journal.

The findings in `benchmark_results/exploration/findings.jsonl` were produced by
one-off analyses.  This file re-derives all of them from the artifacts, so a
claim can be audited, and so the numbers can be refreshed when a new experiment
lands rather than being retyped.  Each section prints the finding it supports.

    python benchmarks/analyze_exploration.py            # everything
    python benchmarks/analyze_exploration.py F006 F015  # named findings only
"""

from __future__ import annotations

import csv, json, math, sys, os
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr, kendalltau, mannwhitneyu, linregress

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from breakout_algorithms import BY_KEY, ON_POLICY, OFF_POLICY

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "benchmark_results" / "artifacts" / "breakout"
EXP = ROOT / "benchmark_results" / "exploration"


def corrected_sps() -> dict[str, float]:
    """Throughput re-derived from the stored logs (F003); falls back to raw."""
    path = EXP / "highenv_sps_corrections.csv"
    if not path.exists():
        return {}
    return {r["trial_id"]: float(r["corrected_sps"]) for r in csv.DictReader(path.open())}


def summary(name: str, col: str) -> dict[str, float]:
    out = {}
    for row in csv.DictReader((ART / name / "summary.csv").open()):
        try:
            out[row["algorithm"]] = float(row[col])
        except (ValueError, KeyError, TypeError):
            pass
    return out


def highenv_trials() -> list[dict]:
    fixed = corrected_sps()
    out = []
    for line in (ART / "highenv" / "trials.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r["status"] != "ok" or r.get("score") is None or not r.get("trained_seconds"):
            continue
        r["sps"] = fixed.get(r["trial_id"], r.get("train_sps") or 0)
        r["transitions"] = r["sps"] * r["trained_seconds"]
        out.append(r)
    return out


def my_runs(experiment: str) -> list[dict]:
    path = EXP / experiment / "runs.jsonl"
    if not path.exists():
        return []
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    return [r for r in rows if r.get("score") is not None and r["status"] == "ok"]


def ols(X: np.ndarray, y: np.ndarray):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    n, k = X.shape
    s2 = (resid ** 2).sum() / max(n - k, 1)
    se = np.sqrt(np.diag(s2 * np.linalg.pinv(X.T @ X)))
    r2 = 1 - (resid ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return beta, se, r2


# --------------------------------------------------------------------------- #

def f001_protocol_disagreement():
    print("\n== F001 · the two protocols disagree ==")
    s, w = summary("1m", "final"), summary("highenv", "score")
    shared = sorted(set(s) & set(w))
    a = np.array([s[k] for k in shared]); b = np.array([w[k] for k in shared])
    rho, p = spearmanr(a, b); tau, pt = kendalltau(a, b)
    print(f"  n={len(shared)}  Spearman {rho:+.3f} (p={p:.4f})  Kendall {tau:+.3f} (p={pt:.4f})")
    rs = {k: i + 1 for i, k in enumerate(sorted(shared, key=lambda k: -s[k]))}
    rw = {k: i + 1 for i, k in enumerate(sorted(shared, key=lambda k: -w[k]))}
    top = {k for k in shared if rs[k] <= 5} & {k for k in shared if rw[k] <= 5}
    print(f"  top-5 overlap: {len(top)}/5 {sorted(top)}")
    moves = sorted(shared, key=lambda k: rs[k] - rw[k])
    for tag, ks in (("biggest fallers", moves[:4]), ("biggest climbers", moves[-4:])):
        print(f"  {tag}: " + ", ".join(
            f"{k} #{rs[k]}->#{rw[k]} ({s[k]:.1f}->{w[k]:.1f})" for k in ks))
    # the report's row for an algorithm is its BEST confirm trial, so the
    # throughput compared against it has to come from that same trial
    hi: dict = {}
    for r in highenv_trials():
        if r["stage"] != "confirm":
            continue
        prev = hi.get(r["algorithm"])
        if prev is None or (r.get("score") or 0) > (prev.get("score") or 0):
            hi[r["algorithm"]] = r
    common = [k for k in shared if k in hi]
    sps = np.array([hi[k]["sps"] for k in common])
    rho_s, p_s = spearmanr(sps, [w[k] for k in common])
    print(f"  Spearman(corrected throughput, wall-clock score) = {rho_s:+.3f} "
          f"(p={p_s:.3f}, R^2 ~ {rho_s**2:.2f}) over n={len(common)}")


def f009_two_arms():
    print("\n== F009/F012 · the exchange rate between steps and data ==")
    for exp, axis, label in (("h005_batch_updates", "planned_updates", "optimizer steps"),
                             ("h006_fixed_updates", None, "samples consumed")):
        runs = my_runs(exp)
        if not runs:
            print(f"  {exp}: no runs yet"); continue
        if axis:
            x = np.array([r[axis] for r in runs])
        else:
            x = np.array([r["planned_updates"] * r["batch_size"] for r in runs])
        y = np.array([r["score"] for r in runs])
        fit = linregress(np.log2(x), y)
        print(f"  {label:16s} n={len(runs)}  slope {fit.slope:+.2f}/doubling  "
              f"R^2 {fit.rvalue**2:.3f}  p {fit.pvalue:.2e}  "
              f"range {y.max()/y.min():.2f}x over {x.max()/x.min():.0f}x")


def f010_envcount():
    print("\n== F010/H012 · environment count with batch and steps pinned ==")
    for exp in ("h008_envcount", "h012_envsweep"):
        runs = my_runs(exp)
        if not runs:
            print(f"  {exp}: no runs yet"); continue
        cells: dict = {}
        for r in runs:
            cells.setdefault((r["algorithm"], r["num_envs"]), []).append(r["score"])
        print(f"  {exp}:")
        for (algo, n), v in sorted(cells.items()):
            spread = f" spread {max(v)-min(v):.2f}" if len(v) > 1 else ""
            print(f"    {algo:9s} n={n:5d}  mean {np.mean(v):7.2f}  (n_seeds={len(v)}){spread}")


def f014_pooled():
    print("\n== F014 · pooled fit, and why the environment term is not identified ==")
    recs = []
    for r in highenv_trials():
        k = r.get("knobs", {}); b = k.get("batch_size"); ratio = r.get("implied_replay_ratio")
        if not b or not ratio or r["transitions"] < 1e5: continue
        recs.append(dict(algo=r["algorithm"], envs=r["num_envs"], trans=r["transitions"],
                         updates=r["transitions"] * ratio / b, score=max(r["score"], .5), src="highenv"))
    for exp in ("h005_batch_updates", "h006_fixed_updates", "h007_rescue",
                "h008_envcount", "h012_envsweep"):
        for r in my_runs(exp):
            if not r.get("batch_size"): continue
            recs.append(dict(algo=r["algorithm"], envs=r["num_envs"], trans=r["reached"],
                             updates=r["reached"] * r["replay_ratio"] / r["batch_size"],
                             score=max(r["score"], .5), src="mine"))
    for r in csv.DictReader((ART / "1m" / "summary.csv").open()):
        if r["family"] != "off_policy" or not r["replay_ratio"] or not r["batch_size"]: continue
        t = float(r["final_step"])
        recs.append(dict(algo=r["algorithm"], envs=int(r["num_envs"]), trans=t,
                         updates=t * float(r["replay_ratio"]) / float(r["batch_size"]),
                         score=max(float(r["final"]), .5), src="1m"))
    algos = sorted({r["algo"] for r in recs}); srcs = sorted({r["src"] for r in recs})
    for use_src in (False, True):
        X, y = [], []
        for r in recs:
            row = [math.log2(r["trans"]), math.log2(r["updates"]), math.log2(r["envs"])]
            row += [1.0 if r["algo"] == a else 0.0 for a in algos[1:]]
            if use_src: row += [1.0 if r["src"] == s else 0.0 for s in srcs[1:]]
            X.append(row + [1.0]); y.append(math.log2(r["score"]))
        beta, se, r2 = ols(np.array(X), np.array(y))
        print(f"  {'with' if use_src else 'without'} source effects: R^2 {r2:.3f} n={len(y)}")
        for i, nm in enumerate(("transitions", "optimizer steps", "environments")):
            print(f"    {nm:16s} {beta[i]:+.3f} +/- {se[i]:.3f}  t={beta[i]/se[i]:+5.1f}")


def f015_family_currency():
    print("\n== F015 · on-policy is paid in data, off-policy in gradient steps ==")
    recs = []
    for r in highenv_trials():
        if BY_KEY[r["algorithm"]].family != ON_POLICY: continue
        k = r.get("knobs", {}); mb = k.get("minibatch"); ep = k.get("update_epochs")
        if not mb or not ep or r["transitions"] < 1e5: continue
        recs.append(dict(algo=r["algorithm"], trans=r["transitions"],
                         updates=r["transitions"] * ep / mb, score=max(r["score"], .5)))
    algos = sorted({r["algo"] for r in recs})
    X = [[math.log2(r["trans"]), math.log2(r["updates"])]
         + [1.0 if r["algo"] == a else 0.0 for a in algos[1:]] + [1.0] for r in recs]
    y = [math.log2(r["score"]) for r in recs]
    beta, se, r2 = ols(np.array(X), np.array(y))
    print(f"  on-policy, n={len(recs)}, {len(algos)} algorithms, R^2 {r2:.3f}")
    for i, nm in enumerate(("fresh transitions", "optimizer steps")):
        print(f"    {nm:18s} {beta[i]:+.3f} +/- {se[i]:.3f}  t={beta[i]/se[i]:+5.1f}  "
              f"{2**beta[i]:.2f}x/doubling")
    s, w = summary("1m", "final"), summary("highenv", "score")
    shared = sorted(set(s) & set(w))
    rs = {k: i + 1 for i, k in enumerate(sorted(shared, key=lambda k: -s[k]))}
    rw = {k: i + 1 for i, k in enumerate(sorted(shared, key=lambda k: -w[k]))}
    groups: dict = {}
    for k in shared:
        groups.setdefault(BY_KEY[k].family, []).append(rs[k] - rw[k])
    for f, v in sorted(groups.items()):
        print(f"    rank change {f:11s} n={len(v):2d} mean {np.mean(v):+5.1f}")
    u, p = mannwhitneyu(groups["on_policy"], groups["off_policy"], alternative="greater")
    print(f"    on-policy climbs more: U={u:.0f}, p={p:.5f}")


SECTIONS = {"F001": f001_protocol_disagreement, "F009": f009_two_arms,
            "F010": f010_envcount, "F014": f014_pooled, "F015": f015_family_currency}

if __name__ == "__main__":
    wanted = [a for a in sys.argv[1:] if a in SECTIONS] or list(SECTIONS)
    for key in wanted:
        SECTIONS[key]()
    print()
