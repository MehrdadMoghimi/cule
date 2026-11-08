#!/usr/bin/env python3
"""H005: does score track optimizer steps, with transitions held fixed?

F006 found that across both Breakout protocols the count of gradient steps
predicts the score (Spearman +0.433, p=0.013) while transitions (-0.197) and
replay ratio (+0.185) do not, and that the collapse of Rainbow, IQN, FQF, DER,
SPR, BBF and BTR at 1,024 environments coincides with their optimizer-step
count falling below roughly 40-50k.  That evidence is observational: a trainer
with a cheaper update also collects more transitions, so the two are entangled.

This experiment separates them.  One trainer, one environment count, one
transition budget, one replay ratio -- only the batch size moves.  Because the
trainer derives `learner_updates_per_vector_step = replay_ratio * num_envs /
batch_size`, halving the batch doubles the optimizer steps while leaving the
data, the sampling rate and the schedule identical:

    batch    32 ->  250,000 optimizer steps   (what the 1M protocol gave DQN)
    batch  2048 ->    3,906 optimizer steps
    everything else identical

Prediction registered in advance: if F006 is right the score falls off a cliff
somewhere between 31k and 62k optimizer steps (batch 256 -> 128).  If score is
instead flat in batch size, F006 is wrong and the 1,024-environment collapse
has another cause.

Scored on transitions, not seconds, so the result is valid even while another
job shares the GPU.
"""

from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from exp_runner import run_experiment

ENVS = 1024
BUDGET = 1_000_000          # matches the 1m benchmark exactly
RATIO = 8.0                 # the published DQN-line replay ratio
BATCHES = (32, 64, 128, 256, 512, 1024, 2048)
BUFFER = 131_072            # the GPU-resident cap; constant so it is not a variable


def specs():
    out = []
    for batch in BATCHES:
        updates = int(BUDGET * RATIO / batch)
        out.append({
            "name": f"dqn-b{batch}",
            "algorithm": "dqn",
            "variant": "torchcompile",
            "num_envs": ENVS,
            "total_timesteps": BUDGET,
            "seed": 1,
            "batch_size": batch,
            "replay_ratio": RATIO,
            "planned_updates": updates,
            "flags": ["--batch-size", batch, "--replay-ratio", RATIO,
                      "--buffer-size", BUFFER],
        })
    # Smallest batches cost the most wall clock, so run cheap ones first: a
    # partial result still spans the predicted cliff.
    return sorted(out, key=lambda s: -s["batch_size"])


if __name__ == "__main__":
    results = run_experiment("h005_batch_updates", specs(), timeout=5400.0,
                             dry_run="--dry-run" in sys.argv)
    print()
    print(f"{'batch':>6s} {'planned updates':>16s} {'score':>8s} {'peak':>8s} {'sps':>7s} {'wall':>7s}")
    for r in sorted([r for r in results if r.get("score") is not None],
                    key=lambda r: -r["batch_size"]):
        print(f"{r['batch_size']:6d} {r['planned_updates']:16,d} {r['score']:8.2f} "
              f"{r['peak']:8.1f} {r['train_sps'] or 0:7d} {r['wall_seconds']:6.0f}s")
