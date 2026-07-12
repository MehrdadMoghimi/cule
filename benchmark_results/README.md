# RTX 4090 throughput search

Measured on 2026-07-12 with the `cule312` environment, PyTorch 2.13.0+cu130,
an NVIDIA GeForce RTX 4090 (24 GB), and `PongNoFrameskip-v4`.  FPS includes the
complete rollout and optimizer workload and excludes evaluation and file I/O.

| Algorithm | Best stable configuration | Throughput |
|---|---|---:|
| A2C | 2,800 envs, 5 steps, normalization off | 66,083 FPS median (61,955–66,387) |
| V-trace | 4,000 envs, 5 steps, 80 minibatches, 1 step/update, normalization off | 64,817 FPS median (64,494–66,772) |
| PPO | 2,000 envs, 5 steps, 3 epochs, normalization off | 20,845 FPS (single run) |

PPO at 2,800 and 4,000 environments became numerically unstable during the
short benchmark (invalid action probabilities), so those failed trials are
retained in `rtx4090_quick.jsonl` rather than ranked.

Recommended maximum-throughput launch:

```
cd examples/a2c
conda run -n cule312 python a2c_main.py \
  --env-name PongNoFrameskip-v4 --use-cuda-env \
  --num-ales 2800 --num-steps 5 \
  --t-max 8000000 --evaluation-interval 2000000
```

Raw results:

- `rtx4090_quick.jsonl`: broad and focused searches, including failures.
- `rtx4090_a2c_repeat.jsonl`: three repeats of the winning A2C setup.
- `rtx4090_vtrace_repeat.jsonl`: three repeats of the winning V-trace setup.

## Pong time-to-solve check

For learning runs, "solved" means a mean reward of at least +18 over 10 full
evaluation episodes.  These are single-seed diagnostic runs, so they show the
direction and scale of the improvement rather than a statistically definitive
optimum.

| Configuration | Frames to solve | Training time | End-to-end wall time |
|---|---:|---:|---:|
| V-trace baseline, lr=0.00065 | 2,000,400 | 71.3 s | about 160 s |
| V-trace, lr=0.001 | 1,500,000 | 55.0 s | 133.5 s |
| V-trace, lr=0.0015 | 1,500,000 | 53.8 s | 118.1 s |
| Throughput A2C, 2,800 envs, normalization off | not solved by 5M | 75.8 s at 4.5M | about 131 s total |

The original V-trace hyperparameters are healthy.  Raising the learning rate
to 0.001 reduced both frames-to-solve and training time by roughly 23–25%.
The 0.0015 result was slightly faster but should be treated as a more aggressive
single-seed candidate.  At 0.002, evaluation plateaued just below the +18
threshold, so increasing the rate further was not useful.

The throughput-only A2C winner stayed at -21 throughout the run.  Its huge
on-policy batch reduced update frequency enough to eliminate useful learning,
which confirms that raw FPS must not be used as the selection objective.

Recommended learning-oriented command:

```
cd examples/vtrace
conda run -n cule312 python vtrace_main.py \
  --env-name PongNoFrameskip-v4 --normalize --use-cuda-env \
  --num-ales 1200 --num-steps 20 --num-steps-per-update 1 \
  --num-minibatches 20 --lr 0.001 \
  --t-max 8000000 --evaluation-interval 1000000
```

Use `--solve-reward 18 --skip-initial-evaluation` to stop automatically after
an evaluation crosses the solve threshold.  A one-million-frame evaluation
interval avoids spending wall time evaluating policies that are still weak.
