#!/usr/bin/env python3
"""Check that CuLE reports a finished episode for every game, not just some.

`CuLEVectorEnv.step` decides a game is over with

    game_over = terminated & raw_info["ale.lives"].eq(0)

and only then publishes `episode/r`.  Every learning curve in this repository
is built from that signal, so if the conjunct is wrong for a game, that game's
curve is silently empty or wrong -- and nothing here has ever trained on
anything but Breakout, where it demonstrably works.

This probe steps a random policy and counts, per game and per `episodic_life`
setting, how often `terminated` fires and how often the conjunct fires with it.
A game where terminations are plentiful but game-overs are zero is the failure
mode worth knowing about before spending GPU hours on cross-game scores.
"""

from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "cleanrl"))
from cule_env import make_cule_env  # noqa: E402


def probe(game: str, episodic_life: bool, num_envs: int, steps: int, device: str) -> dict:
    env = make_cule_env(f"{game}NoFrameskip-v4", num_envs, device, seed=1,
                        episodic_life=episodic_life, clip_rewards=False, stats_interval=1)
    obs, _ = env.reset(seed=1)
    gen = torch.Generator(device=device).manual_seed(1)
    n = env.single_action_space.n
    term = torch.zeros((), device=device)
    over = torch.zeros((), device=device)
    lives_at_term: dict[int, int] = {}
    reported = 0
    for _ in range(steps):
        a = torch.randint(0, n, (num_envs,), device=device, generator=gen)
        obs, r, t, tr, infos = env.step(a)
        lives = infos["lives"]
        term += t.sum()
        over += (t & lives.eq(0)).sum()
        if bool(t.any()):
            for lv in lives[t].tolist():
                lives_at_term[int(lv)] = lives_at_term.get(int(lv), 0) + 1
        reported += sum(1 for i in (infos.get("final_info") or []) if i is not None)
    env.close()
    return {"game": game, "episodic_life": episodic_life,
            "terminations": int(term), "game_overs": int(over), "reported": reported,
            "lives_at_termination": dict(sorted(lives_at_term.items()))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", nargs="*", default=["Breakout", "Qbert", "Seaquest", "MsPacman"])
    ap.add_argument("--num-envs", type=int, default=32)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    for game in args.games:
        for el in (True, False):
            try:
                print(json.dumps(probe(game, el, args.num_envs, args.steps, args.device)), flush=True)
            except Exception as exc:
                print(json.dumps({"game": game, "episodic_life": el,
                                  "error": f"{type(exc).__name__}: {exc}"}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
