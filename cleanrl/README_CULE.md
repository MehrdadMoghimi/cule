# CleanRL with CuLE

The Atari scripts in this directory accept `--env-backend cule`. The shared
adapter keeps batched on-policy rollouts on the GPU and applies the same core
preprocessing as the original CleanRL environments: grayscale 84x84 frames,
four-frame stacking (one frame for recurrent PPO), frame skip/max-pooling,
episodic life, and clipped training rewards.

This integration contains code adapted from
[CleanRL](https://github.com/vwxyzjn/cleanrl) and
[LeanRL](https://github.com/meta-pytorch/LeanRL), plus Atari wrappers from
[stable-baselines3](https://github.com/DLR-RM/stable-baselines3). All three are
MIT licensed; their copyright and license notices are preserved in
[LICENSE.md](LICENSE.md).

Every script states its own provenance in its header: which upstream file it
was adapted from, and — for the algorithms this fork ported itself (QR-DQN,
IQN, FQF, DER, DrQ(ε), SPR, M-IQN, BBF) — which official implementation the
algorithm and hyperparameters came from. [LICENSE.md](LICENSE.md) summarizes
both in a table.

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

Every bundled algorithm has a paired torch.compile entry point: PPO
(`ppo_atari_envpool_torchcompile.py`), recurrent PPO
(`ppo_atari_lstm_torchcompile.py`), DQN, C51, QR-DQN, IQN, FQF, Rainbow, PQN,
discrete SAC (`sac_atari_torchcompile.py`), and QDagger
(`qdagger_dqn_atari_impalacnn_torchcompile.py`). Omitting the --compile flag
runs the same implementation eagerly, which makes a matched throughput
comparison straightforward. All variants additionally accept `--cudagraphs`,
which captures the fixed-shape policy and learner update with tensordict's
`CudaGraphModule` (combinable with `--compile`; PER sampling, priority-tree
writes, and CuLE stepping stay outside the capture):

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

conda run -n cule312 python cleanrl/sac_atari_torchcompile.py \
  --env-backend cule --env-id PongNoFrameskip-v4 \
  --num-envs 256 --batch-size 512 --replay-ratio 1 --compile

conda run -n cule312 python cleanrl/ppo_atari_lstm_torchcompile.py \
  --env-backend cule --env-id PongNoFrameskip-v4 \
  --num-envs 256 --num-steps 32 --compile
~~~

## Distributional RL: C51, QR-DQN, IQN, FQF

Alongside C51, the repository bundles the quantile-based distributional
family: QR-DQN (`qrdqn_atari.py`, quantile regression over N fixed quantile
midpoints), IQN (`iqn_atari.py`, implicit quantiles via a cosine embedding of
sampled taus), and FQF (`fqf_atari.py`, a fraction-proposal network trained
with its own RMSprop optimizer on the W1 gradient). All three support the
`gymnasium`, `cule`, and `envpool` backends in both the eager and the
torch.compile variants, share the C51 trainer structure (decoupled
collection, `--replay-ratio`, benchmark mode), and default to the paper
hyperparameters (learning rate 5e-5; QR-DQN N=200, IQN N=N'=64/K=32,
FQF N=32 with fraction learning rate 2.5e-9):

```bash
conda run -n cule312 python cleanrl/qrdqn_atari.py \
  --env-backend cule --env-id PongNoFrameskip-v4 --num-envs 256 \
  --batch-size 512 --replay-ratio 1

conda run -n cule312 python cleanrl/iqn_atari_torchcompile.py \
  --env-backend cule --env-id PongNoFrameskip-v4 \
  --num-envs 256 --batch-size 512 --replay-ratio 1 --compile --cudagraphs

conda run -n cule312 python cleanrl/fqf_atari_torchcompile.py \
  --env-backend envpool --env-id Pong-v5 \
  --num-envs 256 --batch-size 512 --replay-ratio 1 --compile
```

IQN and FQF draw their quantile samples inside the compiled/captured
regions; PyTorch's graph-safe Philox RNG keeps replays statistically fresh
under `--cudagraphs`. FQF's two losses touch disjoint parameter sets, so the
learner performs a single backward over their sum, which keeps the update
region compilable.

## Atari-100K: DER and DrQ(ε)

`der_atari.py` (Data-Efficient Rainbow, van Hasselt et al. 2019) and
`drq_atari.py` (DrQ(ε), Kostrikov et al. 2020 with the ε-greedy evaluation of
Agarwal et al. 2021) are ported from the official Dopamine implementations in
`dopamine/labs/atari_100k` (`DER.gin`, `DrQ_eps.gin`). The defaults reproduce
the official single-environment configuration: a 100K-step budget, batch 32,
one update per environment step, n-step 10, Adam 1e-4 with epsilon 1.5e-4,
learning starts at 1,600 transitions, and Dopamine's warmup-then-linear-decay
exploration schedule. DER keeps the full Rainbow components with the labs'
prioritized-replay scheme (priorities `sqrt(loss)`, fixed `1/sqrt(prob)`
importance weights — alpha = beta = 0.5, no annealing); DrQ(ε) is Efficient
DQN (dueling, double, non-distributional, uniform n-step replay, target sync
every update) with DrQ random-shift/intensity augmentation enabled by
default. Both accept `--data-augmentation`, support all three backends, and
keep the family's `--replay-ratio` semantics for vectorized runs:

```bash
conda run -n cule312 python cleanrl/der_atari.py \
  --env-id BreakoutNoFrameskip-v4   # official 100K single-env configuration

conda run -n cule312 python cleanrl/drq_atari.py \
  --env-backend cule --env-id PongNoFrameskip-v4 \
  --num-envs 64 --batch-size 64 --replay-ratio 32   # vectorized collection
```

Note that the Atari-100K protocol is defined for a single environment; with
vectorized collection, keep `--replay-ratio` near the official 32 sampled
items per transition if sample-efficiency comparability matters.

## Atari-100K: SPR, M-IQN, and BBF

The rest of the sample-efficiency family is ported from the author releases:

- `spr_atari.py` (Schwarzer et al. 2021, from `mila-iqia/spr`): the DER-style
  Rainbow base (noisy std 0.5, dueling hidden 256, double DQN, C51, PER with
  beta annealed to 1) plus the SPR objective — a convolutional transition
  model rolled K=5 steps, the q_l1 projection (deterministic first dueling
  layers), a linear predictor, an EMA target encoder/projection (tau 0.01),
  and a normalized-L2 matching loss with weight 5, masked past terminals.
  Two updates per environment step, target sync every update, grad-norm clip
  10 per official parameter group, shift+intensity augmentation on learner
  and behavior-policy inputs. Deviations from the official code: the inert
  reward predictor (loss weight 0) is omitted, and n-step windows crossing a
  terminal are excluded from sampling rather than truncated.
- `miqn_atari.py` (Munchausen-IQN, Vieillard et al. 2020, from
  google-research/munchausen_rl): IQN with N = N' = K = 32 whose target adds
  `alpha * clip(tau * ln pi(a|s), -1, 0)` to the reward and bootstraps the
  soft expectation `sum_a pi(a|s')(z_j(s',a) - tau ln pi(a|s'))` with
  `pi = softmax(Q/tau)` from the target network (alpha 0.9, tau 0.03). The
  behavior policy samples `softmax(Q/tau)` via Gumbel-max (official
  `interact='stochastic'`); pass `--interact greedy` for argmax.
- `bbf_atari.py` (Schwarzer et al. 2023, from Google Research's
  `bigger_better_faster`, `configs/BBF.gin`): Impala-CNN encoder (width 4,
  two residual blocks per stage, min-max renormalized latents), a 2048-unit
  projection as the shared hidden layer of dueling C51 heads, the SPR
  objective with targets from the single EMA target network (tau 0.005),
  action selection from the target network, DrQ augmentation, prioritized
  replay, AdamW (weight decay 0.1), exponential anneals of the update
  horizon 10 -> 3 and gamma 0.97 -> 0.997 over the 10k gradient steps after
  each reset, and shrink-and-perturb resets (0.5/0.5 on encoder and
  transition model, hard resets elsewhere) every 20k gradient steps.

All three support the `gymnasium`, `cule`, and `envpool` backends and have
paired torch.compile entry points (`*_torchcompile.py`) accepting `--compile`
and `--cudagraphs`. Their sequence replay stays on the host; the compiled
learner update covers the full loss (including the K-step rollout, and for
BBF the EMA target update). BBF's shrink-and-perturb resets zero the AdamW
moments in place under CUDA graphs so captured graphs stay valid across
resets. Compiled speedups are largest for SPR (about +80% at 64 envs on an
RTX 4090; its rollout is launch-bound) and smaller for BBF (compute-bound in
the width-4 Impala encoder).

## Recent algorithms

Seven more trainers were ported from recent papers, each with the usual
`*_torchcompile.py` twin and the same three backends:
`hadamax_pqn_atari_envpool.py`, `btr_atari.py`, `stream_q_atari.py`,
`stream_ac_atari.py`, `r2d2_atari.py`, `mrq_atari.py` and `disco_atari.py`.
[ALGORITHM_LINEAGE.md](ALGORITHM_LINEAGE.md) is the reference for all seven: it
records which existing file each one descends from, what it inherits unchanged,
what it changes, and how the equivalence tests pin it to its official
implementation.

`disco_atari.py` is the one with an extra setup step. It runs DeepMind's
*discovered* update rule, so it needs the published meta-parameters; they are
downloaded once to `~/.cache/cule-disco/disco_103.npz` on first run, or can be
pointed at an existing copy with `--meta-weights`.

Three 2026 papers were added afterwards, eager-only for now:

- `ibdqn_atari.py` (+ twin) — DQN with the **mean-expansion layer**
  (arXiv:2606.29806), a final layer with no parameters that shares each TD error
  across all actions. `--mean-scaling-coefficient` defaults to `k = n`; `0` is
  plain DQN. The same flag on `iqn_atari.py` and its twin gives IB-IQN, and
  defaults to off so IQN is untouched.
- `ppo_rv_atari.py` — PPO whose critic learns **value differences**
  `Delta(s_i, s_j)` instead of values, with advantages rebuilt from them
  (R-GAE, arXiv:2607.21120). Diffed against the authors' code: 28/28.
- `endpoint_ddqn_atari.py` — Double DQN with **Endpoint Replay**
  (arXiv:2607.25123): a small recency buffer plus a coreset of chained n-step
  transitions, trained with expectile Sarsa.

`benchmarks/me_layer_gridworld.py` reruns the mean-expansion paper's tabular
gridworld experiment; the outcome, including what did not reproduce, is in
[ALGORITHM_LINEAGE.md](ALGORITHM_LINEAGE.md).

## QDagger

`qdagger_dqn_atari_impalacnn.py` distills a pretrained Nature-CNN DQN teacher
into an Impala-CNN student (offline phase on a teacher-generated replay, then
online fine-tuning with an annealed distillation coefficient). It supports
vectorized collection on all three backends — `gymnasium`, `cule`, and
`envpool` — with teacher evaluation and replay generation running on the same
vectorized backend. The teacher checkpoint comes from the CleanRL Hugging
Face hub by default (`--teacher-policy-hf-repo`), or from a local file via
`--teacher-model-path`. Note that the default hub repo name is derived from
`--env-id`, so with the EnvPool backend (`Breakout-v5`-style ids) pass the
repo or a local checkpoint explicitly.

```bash
conda run -n cule312 python cleanrl/qdagger_dqn_atari_impalacnn.py \
  --env-backend cule --env-id BreakoutNoFrameskip-v4 \
  --num-envs 256 --batch-size 512 --replay-ratio 1

conda run -n cule312 python cleanrl/qdagger_dqn_atari_impalacnn_torchcompile.py \
  --env-backend envpool --env-id Breakout-v5 \
  --teacher-policy-hf-repo cleanrl/BreakoutNoFrameskip-v4-dqn_atari-seed1 \
  --num-envs 256 --batch-size 512 --replay-ratio 1 --compile --cudagraphs
```

The torchcompile QDagger variant keeps both the teacher replay and the online
replay on the training GPU (TorchRL `LazyTensorStorage`), so its default
`--teacher-steps`, `--offline-steps`, and `--buffer-size` are 100k rather
than the eager trainer's 500k/500k/1M (full stacked transitions cost about
53 MB per 1k transitions).

The CuLE vector wrapper steps the backend asynchronously by default (no host
synchronization per step; outputs stay stream-ordered, and host-side reads
such as episode logging synchronize on access). Pass `sync_step=True` to
`CuLEVectorEnv` to restore fully synchronous stepping. On an RTX 4090 at 256
envs asynchronous stepping is worth about +5.7% PQN SPS eagerly and +2.5%
with `--compile --cudagraphs`.

The scripts used to benchmark these trainers (throughput sweeps, CuLE vs
EnvPool comparisons, and fixed-budget learning runs) live in
[benchmarks/](../benchmarks/); headline results are summarized in the
[project README](../README.md).

The `torchcompile` DQN, C51, QR-DQN, IQN, FQF, and Rainbow variants use
TorchRL's GPU-resident `LazyTensorStorage` replay. Rainbow retains prioritized replay with a CUDA
sum/min tree; the installed TorchRL wheel's native prioritized-replay extension
is not compatible with this PyTorch build. This checkout pins TorchRL 0.13.2 in
both `requirements.txt` and the conda environment definition.

All bundled algorithms support vectorized CuLE collection. `--cule-device auto`
uses the training GPU for 32 or more environments and the CPU for smaller
batches. DQN, C51, QR-DQN, IQN, FQF, Rainbow, discrete SAC, and QDagger
decouple collection from learning:
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
