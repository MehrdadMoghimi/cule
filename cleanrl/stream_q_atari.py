# Stream Q(lambda): "Streaming Deep Reinforcement Learning Finally Works"
# (Elsayed et al., ICLR 2025, https://arxiv.org/abs/2410.14606).
#
# Reimplemented from the paper (Algorithm 2 `stream Q(lambda)`, Algorithm 3
# `ObGD`, and Section 4's SparseInit / LayerNorm / reward-scaling recipe).  The
# reference implementation at https://github.com/mohmdelsayed/streaming-drl is
# published under CC BY-NC 4.0, so no code is copied from it and none is
# vendored into this repository; `tests/test_stream_equivalence.py` checks this
# port against the paper's equations instead.
#
# Relative to dqn_atari.py -- the closest existing trainer, since both are
# epsilon-greedy Q-learning that step and update once per transition -- stream
# Q(lambda) *removes* rather than adds machinery:
#
#   * no replay buffer      -> each transition is learned from once, then dropped
#   * no target network     -> the bootstrap uses the online network
#   * no minibatches        -> the update is a per-stream rank-1 trace update
#   * SGD/Adam -> ObGD      -> eligibility traces plus an overshooting bound
#   * SparseInit + LayerNorm + observation normalisation + reward scaling
#
# The backend scaffolding (envpool adapter, `completed_episode_infos`) is the
# one used by miqn_atari.py.  Supports gymnasium, cule, and envpool.
#
# The trainer skeleton is adapted from CleanRL
# (https://github.com/vwxyzjn/cleanrl, MIT; license in cleanrl/LICENSE.md).
#
# CONFIRMED against the official implementation:
# `tests/crosscheck/check_stream.py` runs mohmdelsayed/streaming-drl's own ObGD,
# sparse_init and network next to this file's. 83/83 components match on CPU and
# CUDA: sparse_init bit-exact, the network forward pass, ObGD's parameter
# updates and eligibility traces over five steps including a trace reset and the
# overshooting bound, the Q(lambda) TD error, and the per-stream gradient of
# -Q(s, a) against a plain backward. Upstream is single-stream, so the check
# pins --num-envs 1; the vectorised path reduces to it exactly.
"""Stream Q(lambda) on Atari: replay-free, target-network-free, batch-free.

Streaming RL is defined for a single stream of experience.  This port keeps that
exactly at `--num-envs 1`; for `--num-envs N` it maintains N *independent*
eligibility traces (one per environment, the streams never mix) and reduces
their parameter updates with `--stream-reduction`.  `mean` preserves ObGD's
per-step overshooting bound for the combined update and is the default; `sum`
applies each stream's bounded update in full.  Both are identical at N = 1.
"""

import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass

try:
    import envpool
except ImportError:
    envpool = None
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tyro
from torch.func import functional_call, grad, vmap
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from cule_env import (
    CuLEVectorEnv,
    done_tensor,
    frame_stack_observation,
    grayscale_observation,
    resolve_cule_device,
    step_env,
    to_numpy,
    to_tensor,
)

from cleanrl_utils.atari_wrappers import (
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)
from cleanrl_utils.episode_stats import EpisodeStats


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances"""
    env_backend: str = "gymnasium"
    """environment backend: `gymnasium`, `cule`, or `envpool`"""
    cule_device: str = "auto"
    """CuLE device; auto uses CUDA for 32+ envs and CPU for smaller batches"""

    # Algorithm specific arguments
    env_id: str = "BreakoutNoFrameskip-v4"
    """the id of the environment"""
    total_timesteps: int = 10000000
    """total timesteps of the experiments"""
    learning_rate: float = 1.0
    """the ObGD base step size (the paper's lr = 1.0)"""
    num_envs: int = 1
    """the number of parallel streams; 1 is the published single-stream setting"""
    stream_reduction: str = "mean"
    """how N streams combine their updates: `mean` keeps ObGD's overshooting
    bound for the combined step, `sum` applies each stream's step in full"""
    hidden_size: int = 256
    """the width of the penultimate layer"""
    sparsity: float = 0.9
    """the SparseInit fraction of zeroed incoming weights per unit"""
    gamma: float = 0.99
    """the discount factor gamma"""
    lamda: float = 0.8
    """the eligibility trace decay lambda"""
    kappa: float = 2.0
    """the ObGD overshooting-bound coefficient"""
    start_e: float = 1.0
    """the starting epsilon for exploration"""
    end_e: float = 0.01
    """the ending epsilon for exploration"""
    exploration_fraction: float = 0.05
    """the fraction of `total-timesteps` it takes from start-e to go end-e"""
    normalize_observations: bool = True
    """whether to standardise observations with running per-pixel statistics"""
    scale_rewards: bool = True
    """whether to divide rewards by the running std of the discounted reward trace"""
    clip_rewards: bool = False
    """whether to sign-clip rewards; the paper scales rather than clips"""
    max_training_seconds: float = 0.0
    """wall-clock training limit; zero disables it"""
    solve_reward: float | None = None
    """stop when the moving episodic return reaches this value"""
    solve_window: int = 20
    """number of completed episodes in the solve moving average"""
    benchmark: bool = False
    """run a fixed warmup/measurement window and print a JSON benchmark result"""
    benchmark_warmup_iterations: int = 10
    """vector-environment steps excluded from benchmark timing"""
    benchmark_measure_iterations: int = 30
    """vector-environment steps included in benchmark timing"""


def make_env(env_id, seed, idx, capture_video, run_name, clip_rewards):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)

        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        if clip_rewards:
            from cleanrl_utils.atari_wrappers import ClipRewardEnv

            env = ClipRewardEnv(env)
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = grayscale_observation(env)
        env = frame_stack_observation(env, 4)

        env.action_space.seed(seed)
        return env

    return thunk


class RecordEpisodeStatistics(gym.Wrapper):
    """Expose EnvPool spaces and full-game statistics to the trainer."""

    def __init__(self, env):
        super().__init__(env)
        self.num_envs = getattr(env, "num_envs", 1)
        self.single_action_space = getattr(env, "single_action_space", env.action_space)
        self.single_observation_space = getattr(env, "single_observation_space", env.observation_space)

    def reset(self, **kwargs):
        observations = super().reset(**kwargs)
        self.episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        self.returned_episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.returned_episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        return observations

    def step(self, action):
        result = super().step(action)
        if len(result) == 5:
            observations, rewards, terminations, truncations, infos = result
            dones = np.logical_or(terminations, truncations)
        else:
            observations, rewards, dones, infos = result
        self.episode_returns += infos["reward"]
        self.episode_lengths += 1
        self.returned_episode_returns[:] = self.episode_returns
        self.returned_episode_lengths[:] = self.episode_lengths
        game_over = np.logical_and(dones, np.asarray(infos["lives"]) == 0)
        self.episode_returns *= 1 - game_over
        self.episode_lengths *= 1 - game_over
        infos["r"] = self.returned_episode_returns
        infos["l"] = self.returned_episode_lengths
        if len(result) == 5:
            return observations, rewards, terminations, truncations, infos
        return observations, rewards, dones, infos


def completed_episode_infos(infos, dones):
    """Normalize per-backend episode statistics to the `final_info` format."""
    if "final_info" in infos:
        return infos
    if "r" in infos:  # EnvPool RecordEpisodeStatistics
        game_over = to_numpy(dones).astype(bool) & (np.asarray(infos["lives"]) == 0)
        if game_over.any():
            return {
                "final_info": [
                    {"episode": {"r": float(infos["r"][index]), "l": int(infos["l"][index])}}
                    for index in np.flatnonzero(game_over)
                ]
            }
    return {}


def sparse_init(tensor: torch.Tensor, sparsity: float) -> torch.Tensor:
    """SparseInit: uniform(+-1/sqrt(fan_in)) with a fraction of inputs zeroed.

    Each output unit keeps an independently drawn random subset of its incoming
    weights, so the fan-in is reduced without making units share a mask.
    """
    with torch.no_grad():
        if tensor.dim() == 2:
            fan_out, fan_in = tensor.shape
            flat = tensor
        elif tensor.dim() == 4:
            channels_out, channels_in, height, width = tensor.shape
            fan_out, fan_in = channels_out, channels_in * height * width
            flat = tensor.reshape(channels_out, fan_in)
        else:
            raise ValueError("SparseInit supports 2- and 4-dimensional tensors only")
        bound = math.sqrt(1.0 / fan_in)
        tensor.uniform_(-bound, bound)
        num_zeros = int(math.ceil(sparsity * fan_in))
        for unit in range(fan_out):
            # The permutation is drawn on the CPU generator, as upstream does,
            # so a seeded run initialises identically on CPU and on GPU.
            zero_indices = torch.randperm(fan_in)[:num_zeros].to(tensor.device)
            flat[unit, zero_indices] = 0.0
    return tensor


class LayerNormalization(nn.Module):
    """Parameter-free LayerNorm over every non-batch axis.

    The paper normalises the whole activation tensor of a layer and keeps no
    learnable scale or bias, which is what holds the pre-activations at unit
    scale without a gain that the trace update could blow up.
    """

    def __init__(self, num_axes: int):
        super().__init__()
        self.num_axes = num_axes

    def forward(self, x):
        return F.layer_norm(x, x.shape[-self.num_axes :])

    def extra_repr(self) -> str:
        return f"num_axes={self.num_axes}"


# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    """Stream Q(lambda)'s value network.

    Wider strides than the Nature CNN (5/3/2 instead of 4/2/1) shrink the trunk
    to a 64x2x2 = 256-unit feature vector; the whole network is ~0.15M
    parameters, which is what makes one eligibility trace per stream cheap.
    Accepts batched (N, 4, 84, 84) or unbatched (4, 84, 84) input, so the same
    module serves both the vectorised policy and the per-stream `vmap` of the
    gradient.
    """

    def __init__(self, env, hidden_size=256, sparsity=0.9):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=5),
            LayerNormalization(3),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, 4, stride=3),
            LayerNormalization(3),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, 3, stride=2),
            LayerNormalization(3),
            nn.LeakyReLU(),
            nn.Flatten(start_dim=-3),
            nn.Linear(256, hidden_size),
            LayerNormalization(1),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, int(env.single_action_space.n)),
        )
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                sparse_init(module.weight, sparsity)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        return self.network(x)


class RunningMeanStd:
    """Welford per-element mean/variance over a stream of samples."""

    def __init__(self, shape, device, epsilon: float = 1e-8):
        self.mean = torch.zeros(shape, device=device, dtype=torch.float64)
        self.sum_squares = torch.zeros(shape, device=device, dtype=torch.float64)
        self.var = torch.ones(shape, device=device, dtype=torch.float64)
        self.count = 0
        self.epsilon = epsilon

    def update(self, samples: torch.Tensor) -> None:
        """Fold in a leading batch of samples one at a time (Welford)."""
        for sample in samples.to(torch.float64):
            if self.count == 0:
                self.mean = sample.clone()
                self.sum_squares = torch.zeros_like(sample)
                self.count = 1
                continue
            self.count += 1
            delta = sample - self.mean
            self.mean = self.mean + delta / self.count
            self.sum_squares = self.sum_squares + delta * (sample - self.mean)
            self.var = self.sum_squares / (self.count - 1)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return ((x.to(torch.float64) - self.mean) / torch.sqrt(self.var + self.epsilon)).float()

    def scale(self, x: torch.Tensor) -> torch.Tensor:
        return (x.to(torch.float64) / torch.sqrt(self.var + self.epsilon)).float()


class ObGD:
    """Overshooting-bounded Gradient Descent with per-stream eligibility traces.

    One trace per parameter per stream.  Per step, for each stream i:

        z_i  <- gamma * lambda * z_i + grad_i
        M_i  <- max(|delta_i|, 1) * ||z_i||_1 * lr * kappa
        eta_i <- lr / M_i  if M_i > 1 else lr
        theta <- theta - reduce_i( eta_i * delta_i * z_i )

    The `eta_i <- lr / M_i` clamp is the overshooting bound: it guarantees the
    update cannot move the TD error past zero, which is what lets the algorithm
    run at lr = 1 without a replay buffer to average over.
    """

    def __init__(self, params, num_streams, lr=1.0, gamma=0.99, lamda=0.8, kappa=2.0, reduction="mean"):
        if reduction not in {"mean", "sum"}:
            raise ValueError("reduction must be 'mean' or 'sum'")
        self.params = list(params)
        self.num_streams = num_streams
        self.lr = lr
        self.gamma = gamma
        self.lamda = lamda
        self.kappa = kappa
        self.reduction = reduction
        self.traces = [
            torch.zeros((num_streams,) + p.shape, device=p.device, dtype=p.dtype) for p in self.params
        ]

    @torch.no_grad()
    def step(self, grads, delta: torch.Tensor, reset: torch.Tensor) -> torch.Tensor:
        """Apply one streaming update.

        `grads` are per-stream gradients shaped (streams, *param_shape), `delta`
        and `reset` are (streams,).  Returns the per-stream step sizes.
        """
        decay = self.gamma * self.lamda
        trace_l1 = torch.zeros(self.num_streams, device=delta.device, dtype=delta.dtype)
        for trace, gradient in zip(self.traces, grads):
            trace.mul_(decay).add_(gradient)
            trace_l1 += trace.abs().flatten(1).sum(1)

        delta_bar = delta.abs().clamp_min(1.0)
        bound = delta_bar * trace_l1 * self.lr * self.kappa
        step_size = torch.where(bound > 1.0, self.lr / bound, torch.full_like(bound, self.lr))

        scale = step_size * delta
        if self.reduction == "mean":
            scale = scale / self.num_streams
        # Watkins' Q(lambda): cut the trace at episode ends and after any
        # exploratory (non-greedy) action, because the trace credits the greedy
        # policy's returns only.  Applied as an unconditional mask multiply so
        # the whole step stays CUDA-graph capturable.
        keep = (~reset).float()
        for parameter, trace in zip(self.params, self.traces):
            broadcast = (-1,) + (1,) * (trace.dim() - 1)
            parameter.sub_((scale.view(broadcast) * trace).sum(0))
            trace.mul_(keep.view(broadcast))
        return step_size


def linear_schedule(start_e: float, end_e: float, duration: float, t: int) -> float:
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


def make_per_stream_grad_fn(q_network):
    """Return a function giving d(-Q(s, a))/d(theta) separately for each stream.

    `vmap` over `grad` gives exact per-stream gradients in one pass, which the
    single shared backward of a batched loss cannot: each stream needs its own
    trace, and the traces never mix.
    """
    buffers = {name: buffer for name, buffer in q_network.named_buffers()}

    def negative_q(params, observation, action_onehot):
        q_values = functional_call(q_network, (params, buffers), (observation,))
        # A one-hot contraction rather than `q_values[action]`: vmap can trace
        # neither indexing by a batched index nor `F.one_hot`'s scatter, so the
        # one-hot is built by the caller outside the transform.
        return -(q_values * action_onehot).sum()

    return vmap(grad(negative_q), in_dims=(None, 0, 0))


if __name__ == "__main__":
    process_start = time.perf_counter()
    args = tyro.cli(Args)
    if args.num_envs < 1:
        raise ValueError("num_envs must be positive")
    if not 0.0 <= args.sparsity < 1.0:
        raise ValueError("sparsity must be in [0, 1)")
    if args.exploration_fraction <= 0:
        raise ValueError("exploration_fraction must be positive")
    if args.max_training_seconds < 0:
        raise ValueError("max_training_seconds must be non-negative")
    if args.benchmark_warmup_iterations < 0:
        raise ValueError("benchmark_warmup_iterations cannot be negative")
    if args.benchmark_measure_iterations < 1:
        raise ValueError("benchmark_measure_iterations must be positive")
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = None if args.benchmark else SummaryWriter(f"runs/{run_name}")
    if writer is not None:
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    # env setup.  The paper scales rewards by the running return std instead of
    # sign-clipping them, so reward clipping is off by default on every backend.
    if args.env_backend == "cule":
        env_device = resolve_cule_device(args.cule_device, device, args.num_envs)
        if args.capture_video:
            raise ValueError("CuLE backend does not support CleanRL video capture")
        envs = CuLEVectorEnv(
            args.env_id, args.num_envs, env_device, seed=args.seed, clip_rewards=args.clip_rewards
        )
    elif args.env_backend == "envpool":
        if envpool is None:
            raise ImportError("EnvPool backend requested; install envpool or pass --env-backend cule")
        envs = RecordEpisodeStatistics(
            envpool.make(
                args.env_id,
                env_type="gym",
                num_envs=args.num_envs,
                episodic_life=True,
                reward_clip=args.clip_rewards,
                seed=args.seed,
            )
        )
    elif args.env_backend == "gymnasium":
        envs = gym.vector.SyncVectorEnv(
            [
                make_env(args.env_id, args.seed + i, i, args.capture_video, run_name, args.clip_rewards)
                for i in range(args.num_envs)
            ]
        )
    else:
        raise ValueError(f"unsupported environment backend: {args.env_backend}")
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"
    n_actions = int(envs.single_action_space.n)

    q_network = QNetwork(envs, hidden_size=args.hidden_size, sparsity=args.sparsity).to(device)
    optimizer = ObGD(
        q_network.parameters(),
        num_streams=args.num_envs,
        lr=args.learning_rate,
        gamma=args.gamma,
        lamda=args.lamda,
        kappa=args.kappa,
        reduction=args.stream_reduction,
    )
    per_stream_grad = make_per_stream_grad_fn(q_network)
    observation_stats = RunningMeanStd(envs.single_observation_space.shape, device)
    reward_stats = RunningMeanStd((), device)
    reward_trace = torch.zeros(args.num_envs, device=device)

    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    reset_result = envs.reset(seed=args.seed) if args.env_backend != "envpool" else envs.reset()
    raw_obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result
    obs = to_tensor(raw_obs, device, torch.float32)
    if args.normalize_observations:
        observation_stats.update(obs)
        obs = observation_stats.normalize(obs)
    global_step = 0
    episode_stats = EpisodeStats(args.solve_window, args.solve_reward)
    next_log_step = max(10000, args.num_envs)
    num_vector_steps = int(np.ceil(args.total_timesteps / args.num_envs))
    if args.benchmark:
        num_vector_steps = args.benchmark_warmup_iterations + args.benchmark_measure_iterations
    benchmark_start = None
    benchmark_start_step = None
    last_delta = None
    last_step_size = None
    for vector_step in range(num_vector_steps):
        if args.benchmark and vector_step == args.benchmark_warmup_iterations:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            benchmark_start = time.perf_counter()
            benchmark_start_step = global_step
        if args.max_training_seconds and time.time() - start_time >= args.max_training_seconds:
            break

        # ALGO LOGIC: put action logic here
        epsilon = linear_schedule(
            args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step
        )
        with torch.no_grad():
            greedy_actions = torch.argmax(q_network(obs), dim=-1)
        random_actions = torch.randint(n_actions, (args.num_envs,), device=device)
        explore = torch.rand(args.num_envs, device=device) < epsilon
        actions = torch.where(explore, random_actions, greedy_actions)
        # A trace reset is due only when the action actually differs from the
        # greedy one, not merely because the coin came up "explore".
        non_greedy = actions != greedy_actions

        # TRY NOT TO MODIFY: execute the game and log data.
        step_result = step_env(envs, actions)
        if len(step_result) == 5:
            next_raw_obs, rewards, terminations, truncations, infos = step_result
        else:
            next_raw_obs, rewards, terminations, infos = step_result
            truncations = np.zeros_like(np.asarray(terminations), dtype=bool)
        terminated = to_tensor(terminations, device, torch.bool)
        transition_dones = done_tensor(terminations, truncations, device).bool()
        global_step += args.num_envs

        next_obs = to_tensor(next_raw_obs, device, torch.float32)
        if args.normalize_observations:
            observation_stats.update(next_obs)
            next_obs = observation_stats.normalize(next_obs)
        rewards = to_tensor(rewards, device, torch.float32).view(-1)
        if args.scale_rewards:
            # The scale is the running std of the discounted reward trace, not
            # of the raw reward, so it tracks the magnitude of returns.
            reward_trace = reward_trace * args.gamma * (~transition_dones).float() + rewards
            reward_stats.update(reward_trace)
            rewards = reward_stats.scale(rewards)

        solved = False
        if not args.benchmark:
            solved = episode_stats.update(
                completed_episode_infos(infos, transition_dones), global_step, writer
            )

        # ALGO LOGIC: training -- one update per transition, no replay.
        with torch.no_grad():
            q_sa = q_network(obs).gather(1, actions.view(-1, 1)).squeeze(1)
            next_max_q = q_network(next_obs).max(dim=-1).values
            # `terminated` and not `done`: a truncated episode still bootstraps.
            td_target = rewards + args.gamma * next_max_q * (~terminated).float()
            delta = td_target - q_sa

        params = {name: parameter.detach() for name, parameter in q_network.named_parameters()}
        grads = per_stream_grad(params, obs, F.one_hot(actions, n_actions).float())
        last_step_size = optimizer.step(
            [grads[name] for name, _ in q_network.named_parameters()],
            delta,
            torch.logical_or(transition_dones, non_greedy),
        )
        last_delta = delta

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        if not args.benchmark and global_step >= next_log_step:
            sps = int(global_step / (time.time() - start_time))
            writer.add_scalar("losses/td_error", last_delta.abs().mean().item(), global_step)
            writer.add_scalar("losses/q_values", q_sa.mean().item(), global_step)
            writer.add_scalar("charts/step_size", last_step_size.mean().item(), global_step)
            writer.add_scalar("charts/epsilon", epsilon, global_step)
            writer.add_scalar("charts/SPS", sps, global_step)
            print("SPS:", sps)
            next_log_step = global_step + max(10000, args.num_envs)
        if solved:
            break

    if args.benchmark:
        if benchmark_start is None or benchmark_start_step is None:
            raise RuntimeError("benchmark measurement window did not start")
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        benchmark_end = time.perf_counter()
        measured_steps = global_step - benchmark_start_step
        measured_seconds = benchmark_end - benchmark_start
        result = {
            "algorithm": "stream_q",
            "backend": args.env_backend,
            "benchmark": "full_training_loop",
            "compile": False,
            "env_device": str(getattr(envs, "device", "cpu")),
            "env_id": args.env_id,
            "learner_updates": measured_steps // args.num_envs,
            "measure_iterations": args.benchmark_measure_iterations,
            "measured_seconds": measured_seconds,
            "measured_steps": measured_steps,
            "num_envs": args.num_envs,
            "peak_cuda_memory_mb": (
                torch.cuda.max_memory_allocated(device) / (1024**2) if device.type == "cuda" else 0.0
            ),
            "process_seconds": benchmark_end - process_start,
            "replay_backend": "none_streaming",
            "schema_version": 1,
            "sps": measured_steps / measured_seconds,
            "stream_reduction": args.stream_reduction,
            "ups": (measured_steps / args.num_envs) / measured_seconds,
            "warmup_iterations": args.benchmark_warmup_iterations,
        }
        print(f"BENCHMARK_RESULT {json.dumps(result, sort_keys=True)}", flush=True)
    else:
        elapsed = time.time() - start_time
        print("SPS:", int(global_step / max(elapsed, 1e-9)))
        episode_stats.print_summary()

    envs.close()
    if writer is not None:
        writer.close()
