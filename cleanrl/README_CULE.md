# CleanRL with CuLE

The Atari scripts in this directory accept `--env-backend cule`. The shared
adapter keeps batched on-policy rollouts on the GPU and applies the same core
preprocessing as the original CleanRL environments: grayscale 84x84 frames,
four-frame stacking (one frame for recurrent PPO), frame skip/max-pooling,
episodic life, and clipped training rewards.

This integration contains code adapted from
[CleanRL](https://github.com/vwxyzjn/cleanrl) and
[LeanRL](https://github.com/meta-pytorch/LeanRL). Both projects are available
under the MIT License; their copyright and license notices are preserved in
[LICENSE.md](LICENSE.md).

Examples:

```bash
conda run -n cule312 python cleanrl/ppo_atari.py \
  --env-backend cule --env-id PongNoFrameskip-v4 \
  --num-envs 256 --num-steps 32

conda run -n cule312 python cleanrl/pqn_atari_envpool.py \
  --env-backend cule --env-id PongNoFrameskip-v4 \
  --num-envs 256 --num-steps 32

conda run -n cule312 python cleanrl/dqn_atari.py \
  --env-backend cule --env-id PongNoFrameskip-v4 --num-envs 256 \
  --batch-size 512 --replay-ratio 1

conda run -n cule312 python cleanrl/rainbow_atari.py \
  --env-backend cule --env-id PongNoFrameskip-v4 --num-envs 256 \
  --batch-size 512 --replay-ratio 1

conda run -n cule312 python cleanrl/dqn_torchcompile.py \
  --env-id PongNoFrameskip-v4 --compile

conda run -n cule312 python cleanrl/ppo_atari_envpool_torchcompile.py \
  --env-id PongNoFrameskip-v4 --compile
```

PQN, C51, and Rainbow also have paired torch.compile entry points. Omitting
the --compile flag runs the same implementation eagerly, which makes a matched
throughput comparison straightforward. The PPO, DQN, PQN, and Rainbow variants
additionally accept `--cudagraphs`, which captures the fixed-shape policy and
learner update with tensordict's `CudaGraphModule` (combinable with
`--compile`; PER sampling, priority-tree writes, and CuLE stepping stay
outside the capture):

~~~bash
conda run -n cule312 python cleanrl/pqn_atari_envpool_torchcompile.py \
  --env-backend cule --env-id PongNoFrameskip-v4 \
  --num-envs 256 --num-steps 32 --compile

conda run -n cule312 python cleanrl/c51_atari_torchcompile.py \
  --env-backend cule --env-id PongNoFrameskip-v4 \
  --num-envs 256 --batch-size 512 --replay-ratio 1 --compile

conda run -n cule312 python cleanrl/rainbow_atari_torchcompile.py \
  --env-backend cule --env-id PongNoFrameskip-v4 \
  --num-envs 256 --batch-size 512 --replay-ratio 1 --compile
~~~

The scripts used to benchmark these trainers (throughput sweeps, CuLE vs
EnvPool comparisons, and fixed-budget learning runs) live in
[benchmarks/](../benchmarks/); headline results are summarized in the
[project README](../README.md).

The `torchcompile` DQN, C51, and Rainbow variants use TorchRL's GPU-resident
`LazyTensorStorage` replay. Rainbow retains prioritized replay with a CUDA
sum/min tree; the installed TorchRL wheel's native prioritized-replay extension
is not compatible with this PyTorch build. This checkout pins TorchRL 0.13.2 in
both `requirements.txt` and the conda environment definition.

All bundled algorithms support vectorized CuLE collection. `--cule-device auto`
uses the training GPU for 32 or more environments and the CPU for smaller
batches. DQN, C51, Rainbow, and discrete SAC decouple collection from learning:
`--learner-updates-per-vector-step` directly controls optimizer launches and may
be fractional. `--replay-ratio` is often easier to compare across environment
and minibatch counts; it means sampled replay items per newly collected
transition and overrides the direct update setting:

```text
updates per vector step = replay ratio * num envs / batch size
```

For example, 256 environments, batch size 512, and replay ratio 1 performs half
an update per vector step. Target-network frequencies are also expressed in
learner updates, so changing the actor count no longer silently changes the
number of gradient steps between target refreshes.

Use `--max-training-seconds` for equal-wall-clock comparisons and
`--solve-reward` plus `--solve-window` for early stopping. The trainers report
SPS, learner updates/s, effective update-to-data ratio, replay ratio, and moving
episodic return.

The regular off-policy trainers use frame-efficient replay. Only one 84x84
`uint8` frame is stored per transition; four-frame observations are reconstructed
when sampled without crossing episode boundaries. A one-million-transition
replay therefore uses about 7.1 GB for frames instead of roughly 28 GB for
stored four-frame observations, or 56 GB when both current and next stacks are
stored. The TorchRL-backed torchcompile DQN, C51, and Rainbow variants instead
store full current and next stacks on the training GPU to remove replay
host-to-device transfers (about 5.3 GB for 100,000 `uint8` transitions before
model and allocator overhead). Rainbow additionally uses vectorized prioritized
replay and independent n-step trajectories for every environment.

## RTX 4090 throughput snapshot

Measured with PyTorch 2.13.0+cu130. `SPS` is full training-loop throughput,
including environment interaction, inference, loss computation,
backpropagation, and optimizer updates. Compare backends within a row; the
algorithms use different update workloads, so cross-row ranking is not a
learning-efficiency comparison.

> The CuLE rows below were measured before the warp-cooperative frame
> renderer, which raises CuLE throughput a further 24-48% depending on the
> workload (re-measured compiled PPO: 16,112 SPS at 256 envs, up from
> 12,301). Rerun `benchmarks/benchmark_cule_envpool.py` for current numbers.

| Algorithm | Backend | Workload | Final SPS |
|---|---|---:|---:|
| PPO | CuLE GPU | 256 envs, 32 steps, 4 epochs | 10,580 |
| PPO | Gymnasium SyncVectorEnv | same | 1,330 |
| recurrent PPO | CuLE GPU | 256 envs, 32 steps, 4 epochs | 8,184 |
| PQN | CuLE GPU | 256 envs, 32 steps, 4 epochs | 10,372 |
| PQN | EnvPool 1.2.5 | same | 14,882 |
| DQN | CuLE GPU | 256 envs, batch 32, 1 update/vector step | 10,382 |
| C51 | CuLE GPU | 256 envs, batch 32, 1 update/vector step | 9,788 |
| Rainbow | CuLE GPU | 256 envs, batch 32, 1 update/vector step | 9,140 |
| discrete SAC | CuLE GPU | 256 envs, batch 64, 1 update/vector step | 6,774 |
| DQN | CuLE GPU | 256 envs, batch 512, replay ratio 1 | 8,407 |
| C51 | CuLE GPU | 256 envs, batch 512, replay ratio 1 | 7,650 |
| Rainbow | CuLE GPU | 256 envs, batch 512, replay ratio 1 | 7,300 |
| discrete SAC | CuLE GPU | 256 envs, batch 512, replay ratio 1 | 7,075 |

At this tested batch size, CuLE made PPO about 8x faster than synchronous
Gymnasium. EnvPool remained about 44% faster than CuLE for PQN. The new
off-policy measurements used 32,768 total transitions and 2,048 random-start
transitions. The old transition-coupled implementation forced 64 updates after
each 256-environment step and reached only 926/665/479/388 SPS for
DQN/C51/Rainbow/SAC in shorter smoke runs. One update per vector step is about
11-19x faster, but it also lowers replay intensity; the batch-512 replay-ratio-1
rows are the more useful starting points for learning experiments.

## Short Pong wall-clock check

A single-seed, 60-second DQN check was used to reject throughput-only settings.
It is too short to establish a final learning winner, but it shows why SPS is
not the objective by itself.

| Envs | Batch | Updates/vector step | Replay ratio | Final SPS | Recent return | Best 20-episode mean |
|---:|---:|---:|---:|---:|---:|---:|
| 256 | 32 | 1 | 0.125 | 7,912 | -20.60 | -20.15 |
| 256 | 512 | 0.5 | 1 | 8,699 | -20.25 | -19.85 |
| 256 | 1024 | 2 | 8 | 2,284 | no completed game | no completed game |
| 128 | 512 | 0.5 | 2 | 4,463 | -20.40 | -20.30 |
| 64 | 512 | 0.5 | 4 | 2,303 | -20.55 | -19.80 |

The 256-env, batch-512, replay-ratio-1 configuration is therefore the current
starting point, not a claim of a solved optimum. Longer runs should sweep replay
ratio around 1 while ranking configurations by wall-clock time to a return
threshold.
