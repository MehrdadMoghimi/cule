#!/usr/bin/env python3
"""H006: separate 'more optimizer steps' from 'smaller batch'.

H005 sweeps batch size at a fixed replay ratio, which is what the highenv
ladder effectively did -- but it moves two things at once, because the trainer
derives updates from the ratio: a smaller batch is simultaneously more gradient
steps AND a smaller gradient step.  Obando-Ceron, Bellemare and Castro (NeurIPS
2023, arXiv:2310.03882) show the second factor alone is worth real performance
in the DQN family, so H005 on its own cannot support the claim F006 makes.

This is the other arm.  Optimizer steps are pinned at 62,500 for every
configuration and the replay ratio is moved to compensate, so batch size varies
64x while the number of gradient steps -- and the transition budget, and the
updates per vector step, which works out to 64 in every cell -- does not.

    batch    32, ratio   2.0 -> 62,500 updates
    batch  2048, ratio 128.0 -> 62,500 updates

Read against H005:

    H005 varies, H006 flat   -> optimizer-step count drives it; F006 stands.
    both vary                -> batch size matters in its own right; F006 is at
                                best half the story and the small-batch
                                literature is the better explanation.
    H005 flat                -> F006 is wrong.

DQN is a 1-step trainer, which matters: the small-batch effect in that paper is
reported to appear mainly WITH multi-step returns, so a flat H006 here is the
outcome that paper would also predict, and separating the two claims needs the
multi-step follow-up (H007) rather than this pair alone.
"""

from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from exp_runner import run_experiment

ENVS = 1024
BUDGET = 1_000_000
UPDATES = 62_500           # midway up H005's ladder, on the safe side of its cliff
BATCHES = (32, 64, 128, 256, 512, 1024, 2048)
BUFFER = 131_072


def specs():
    out = []
    for batch in BATCHES:
        ratio = UPDATES * batch / BUDGET
        out.append({
            "name": f"dqn-fixedupd-b{batch}",
            "algorithm": "dqn",
            "variant": "torchcompile",
            "num_envs": ENVS,
            "total_timesteps": BUDGET,
            "seed": 1,
            "batch_size": batch,
            "replay_ratio": ratio,
            "planned_updates": UPDATES,
            "flags": ["--batch-size", batch, "--replay-ratio", ratio,
                      "--buffer-size", BUFFER],
        })
    # GPU work is proportional to batch here, so cheapest first.
    return sorted(out, key=lambda s: s["batch_size"])


if __name__ == "__main__":
    results = run_experiment("h006_fixed_updates", specs(), timeout=5400.0,
                             dry_run="--dry-run" in sys.argv)
    print()
    print(f"{'batch':>6s} {'ratio':>8s} {'updates':>9s} {'score':>8s} {'peak':>8s} {'sps':>7s} {'wall':>7s}")
    for r in sorted([r for r in results if r.get("score") is not None],
                    key=lambda r: r["batch_size"]):
        print(f"{r['batch_size']:6d} {r['replay_ratio']:8.1f} {r['planned_updates']:9,d} "
              f"{r['score']:8.2f} {r['peak']:8.1f} {r['train_sps'] or 0:7d} {r['wall_seconds']:6.0f}s")
