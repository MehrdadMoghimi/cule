#!/usr/bin/env python3
"""Measure cheap, training-free descriptors of each CuLE game.

Every finding in `benchmark_results/exploration` so far was measured on
Breakout.  Before spending GPU hours replicating them elsewhere it is worth
knowing *how* other games differ, in numbers rather than in reputation.  All of
these are computed from a uniform-random policy, so they cost seconds per game
and describe the environment rather than any learner:

    random_return      mean full-game raw score  (the zero point of every ratio)
    reward_density     fraction of agent steps carrying a nonzero raw reward
    reward_scale       mean absolute magnitude of the nonzero rewards
    reward_cv          coefficient of variation of the nonzero reward
                       magnitudes -- how much the standard `sign()` wrapper
                       throws away (0 means clipping is lossless)
    frame_delta        mean |pixel change| between consecutive agent steps,
                       normalised: how fast the observation stream moves
    novelty_rate       distinct coarse states per 1,000 steps, measured over the
                       second half of the rollout so the startup transient is
                       excluded: how quickly a random walk keeps finding states
                       it has not seen
    eff_rank           participation ratio of the observation covariance
                       spectrum, exp(H(eigenvalues)): how many dimensions the
                       visited state manifold occupies.  It has the advantage of
                       not depending on a quantisation grid and the drawback,
                       measured rather than assumed, of being dominated by global
                       motion -- across the 17 games it correlates -0.55 with
                       `frame_delta`, and the games with a scrolling or flashing
                       full-screen background (BeamRider 2.1, Enduro 1.3,
                       Assault 1.8) come out looking nearly one-dimensional.
                       Prefer `novelty_rate` where the two disagree.

`novelty_rate` is the descriptor this was written for.  The working hypothesis
is that a game whose transitions are highly redundant pays a learner in gradient
steps, and a game that keeps producing genuinely new states pays it in fresh
data -- which would make F015's on-policy/off-policy "currency" split partly a
property of Breakout rather than of the algorithms.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "cleanrl"))

from cule_env import make_cule_env  # noqa: E402

GAMES = [
    "Pong", "Breakout", "SpaceInvaders", "BeamRider", "Seaquest", "Qbert",
    "MsPacman", "Asterix", "Krull", "Freeway", "Enduro", "Boxing",
    "Riverraid", "Frostbite", "Alien", "Assault", "Amidar", "Gopher",
]


def descriptors(game: str, num_envs: int, steps: int, seed: int, device: str) -> dict:
    env_id = f"{game}NoFrameskip-v4"
    t0 = time.time()
    env = make_cule_env(env_id, num_envs, device, seed=seed,
                        episodic_life=False, clip_rewards=False, stats_interval=64)
    obs, _ = env.reset(seed=seed)
    n_actions = env.single_action_space.n

    gen = torch.Generator(device=device).manual_seed(seed)
    # A 64-bit random projection of a coarsened frame stands in for a state id.
    # 12x12 cells, 8 grey levels: fine enough to separate game situations,
    # coarse enough that sprite-level jitter does not count as a new state.
    grid = 20
    cells = grid * grid
    proj = torch.randint(-(2**62), 2**62, (cells,), device=device,
                         dtype=torch.int64, generator=gen)

    returns, lengths = [], []
    rew_sum = torch.zeros((), device=device)
    rew_abs = torch.zeros((), device=device)
    rew_sq = torch.zeros((), device=device)
    rew_nonzero = torch.zeros((), device=device)
    delta_sum = torch.zeros((), device=device)
    prev = obs[:, -1].float()
    seen: set[int] = set()
    novel_probe = min(8, num_envs)          # hash only a few envs; a rate needs no more
    hashes = []
    feats = []                              # downsampled frames for the covariance

    for t in range(steps):
        actions = torch.randint(0, n_actions, (num_envs,), device=device, generator=gen)
        obs, reward, term, trunc, infos = env.step(actions)
        rew_sum += reward.sum()
        rew_abs += reward.abs().sum()
        rew_sq += reward.square().sum()
        rew_nonzero += (reward != 0).sum()
        cur = obs[:, -1].float()
        delta_sum += (cur - prev).abs().mean()
        prev = cur
        coarse = torch.nn.functional.adaptive_avg_pool2d(
            obs[:novel_probe, -1].float().unsqueeze(1), grid).squeeze(1)
        code = (coarse / 16.0).to(torch.int64).clamp_(0, 15).reshape(novel_probe, cells)
        hashes.append((code * proj).sum(1))
        if t % 4 == 0:
            feats.append(coarse.reshape(novel_probe, cells))
        for info in infos.get("final_info", []) or []:
            if info is not None:
                returns.append(info["episode"]["r"])
                lengths.append(info["episode"]["l"])

    stacked = torch.stack(hashes)                     # (steps, probe)
    half = steps // 2
    first = set(stacked[:half].flatten().tolist())
    second = stacked[half:].flatten().tolist()
    novel = sum(1 for h in second if h not in first and not first.add(h))
    # Participation ratio of the observation covariance spectrum.  Centring
    # first matters: without it the mean frame dominates and every game looks
    # one-dimensional.
    feat = torch.cat(feats).double()
    feat -= feat.mean(0, keepdim=True)
    cov = (feat.T @ feat) / max(1, feat.shape[0] - 1)
    ev = torch.linalg.eigvalsh(cov).clamp_min(0)
    p = ev / ev.sum().clamp_min(1e-30)
    p = p[p > 0]
    eff_rank = float(torch.exp(-(p * p.log()).sum()))

    total = float(steps * num_envs)
    n_ep = max(1, len(returns))
    nz = max(1.0, float(rew_nonzero))
    mean_abs = float(rew_abs) / nz
    var_abs = max(0.0, float(rew_sq) / nz - mean_abs ** 2)
    out = {
        "game": game,
        "actions": int(n_actions),
        "episodes": len(returns),
        "random_return": round(sum(returns) / n_ep, 3) if returns else None,
        "episode_len": round(sum(lengths) / n_ep, 1) if lengths else None,
        "reward_density": round(float(rew_nonzero) / total, 6),
        "reward_scale": round(mean_abs, 4),
        "reward_cv": round(var_abs ** 0.5 / max(1e-9, mean_abs), 4),
        "reward_per_step": round(float(rew_sum) / total, 6),
        "frame_delta": round(float(delta_sum) / steps / 255.0, 6),
        "novelty_rate": round(1000.0 * novel / (half * novel_probe), 3),
        "eff_rank": round(eff_rank, 2),
        "seconds": round(time.time() - t0, 1),
    }
    env.close()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", nargs="*", default=GAMES)
    ap.add_argument("--num-envs", type=int, default=64)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=str(ROOT / "benchmark_results" / "exploration"
                                        / "game_descriptors.jsonl"))
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        done = {json.loads(l)["game"] for l in out.open() if l.strip()}

    for game in args.games:
        if game in done:
            print(f"skip {game} (already recorded)", flush=True)
            continue
        try:
            row = descriptors(game, args.num_envs, args.steps, args.seed, args.device)
        except Exception as exc:                      # a game CuLE cannot run
            row = {"game": game, "error": f"{type(exc).__name__}: {exc}"}
        with out.open("a") as fh:
            fh.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
