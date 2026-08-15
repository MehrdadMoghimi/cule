# Endpoint Replay on Double DQN: a small recency buffer plus a coreset of
# chained n-step transitions, so that every bootstrap target stays anchored.
#
# "Endpoint Replay: Compressing the Recency Buffer in Deep Reinforcement
# Learning", Mohammad Panahi, Ashrafi, Du, Patterson, White & White,
# Reinforcement Learning Journal / RLC 2026 (arXiv:2607.25123).
#
# NOT cross-checked against official code. github.com/panahiparham/endpoint-replay
# exists but holds only a LICENSE and a README reading "under construction and
# will be available by the start of the Reinforcement Learning Conference 2026
# (August 15, 2026)" as of 2026-08-08. This file is written from the paper, and
# tests/test_endpoint_replay_equivalence.py asserts the properties the paper
# states rather than a numerical diff against the authors' code.
#
# One reading had to be chosen where the paper is self-inconsistent. Algorithm 1
# line 16 says "Pop the first transition from Dlag" after emitting a summary,
# which would emit one coreset entry per step and compress nothing. Section 3.3
# ("The lag buffer would simply reset"), Section 3.2 ("compress a buffer of size
# m to a coreset of size m/n"), and Section 5 ("the coreset sub-samples every 10
# steps") all require the window to be cleared, which is also what produces the
# chain s_0 -> s_n -> s_2n whose endpoints anchor each other. This file clears.
#
# The trainer body is CleanRL's cleanrl/dqn_atari.py
# (https://github.com/vwxyzjn/cleanrl, MIT; license in cleanrl/LICENSE.md),
# with the Double DQN target of van Hasselt et al. (2016) and the paper's
# Dopamine-style hyperparameters. Supports gymnasium, cule, and envpool.
import json
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
import torch.optim as optim
import tyro
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from cule_env import (
    done_tensor,
    frame_stack_observation,
    grayscale_observation,
    make_cule_env,
    resolve_cule_device,
    step_env,
    to_numpy,
    to_tensor,
)

from cleanrl_utils.atari_wrappers import (
    ClipRewardEnv,
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
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    env_backend: str = "gymnasium"
    """environment backend: `gymnasium`, `cule`, or `envpool`"""
    cule_device: str = "auto"
    """CuLE device; auto uses CUDA for 32+ envs and CPU for smaller batches"""

    # Algorithm specific arguments
    env_id: str = "BreakoutNoFrameskip-v4"
    """the id of the environment"""
    total_timesteps: int = 50000000
    """total timesteps of the experiments (the paper runs 50M frames)"""
    learning_rate: float = 6.25e-5
    """the learning rate of the optimizer"""
    adam_eps: float = 1.5e-4
    """Adam epsilon (the paper's Dopamine-style setting)"""
    num_envs: int = 1
    """the number of parallel game environments"""
    recency_buffer_size: int = 10000
    """capacity of the recency buffer D_r"""
    coreset_buffer_size: int = 10000
    """capacity of the coreset buffer D_c; the paper's 50x setting is 10k + 10k"""
    n_step: int = 10
    """the length n of each chained coreset transition"""
    expectile: float = 0.7
    """the expectile parameter tau used on coreset samples"""
    gamma: float = 0.99
    """the discount factor gamma"""
    target_network_frequency: int = 2000
    """learner updates between target-network updates (32k frames / 16 frames per update)"""
    batch_size: int = 32
    """total minibatch size, split between the two buffers"""
    coreset_batch_size: int = 4
    """how many of the `batch_size` samples come from the coreset (the paper's 7:1 ratio)"""
    start_e: float = 1
    """the starting epsilon for exploration"""
    end_e: float = 0.01
    """the ending epsilon for exploration"""
    exploration_fraction: float = 0.02
    """fraction of `total_timesteps` over which epsilon anneals (1M frames of 50M)"""
    learning_starts: int = 20000
    """timestep to start learning (80k frames / 4)"""
    train_frequency: int = 4
    """environment steps between learner updates (16 frames / 4 per step)"""
    max_training_seconds: float = 0.0
    """wall-clock training limit; zero disables it"""
    solve_reward: float | None = None
    """stop when the moving episodic return reaches this value"""
    solve_window: int = 20
    """number of completed episodes in the solve moving average"""
    benchmark: bool = False
    """run a fixed warmup/measurement window and print a JSON benchmark result"""
    benchmark_warmup_iterations: int = 10
    """vector environment steps excluded from benchmark timing"""
    benchmark_measure_iterations: int = 30
    """vector environment steps included in benchmark timing"""


def make_env(env_id, seed, idx, capture_video, run_name):
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
            returns = np.asarray(infos["r"])
            lengths = np.asarray(infos["l"])
            return {
                "final_info": [
                    {"episode": {"r": float(returns[i]), "l": int(lengths[i])}} if game_over[i] else None
                    for i in range(len(game_over))
                ]
            }
    return infos


class QNetwork(nn.Module):
    """DDQN's original architecture (van Hasselt et al., 2016) -- the Nature CNN."""

    def __init__(self, env):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, env.single_action_space.n),
        )

    def forward(self, x):
        return self.network(x / 255.0)


def linear_schedule(start_e: float, end_e: float, duration: float, t: int) -> float:
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


def expectile_loss(prediction, target, tau):
    """Asymmetric squared error: tau on upward deviations, 1 - tau on downward.

    l_tau(u) = tau * 1[d >= 0] d^2 + (1 - tau) * 1[d < 0] d^2 for d = target -
    prediction. tau = 0.5 gives half the usual squared error, so the paper's
    "no expectile" ablation is `expectile=0.5` up to that factor of two. Fitting
    tau > 0.5 pulls the estimate toward the upper tail of the n-step returns,
    which is what offsets the pessimism of targets collected under older,
    weaker policies (Section 3.2).
    """
    deviation = target - prediction
    weight = torch.where(deviation >= 0, tau, 1.0 - tau)
    return weight * deviation.pow(2)


class EndpointReplayBuffer:
    """Recency ring + per-env lag window + coreset of chained n-step endpoints.

    Storage layout. The recency buffer keeps one 84x84 frame per step and
    rebuilds stacks on demand, as the rest of this repository's Atari buffers
    do. The coreset cannot do that: its two states are n steps apart and every
    frame in between has been dropped, so each coreset row stores two complete
    stacked observations.

    Eviction margin. A transition's stack needs the `frame_stack - 1` frames
    before it, so the very oldest row in a ring can no longer be reconstructed.
    The ring is therefore allocated `frame_stack + 2` rows larger than the
    recency capacity and transitions are handed to the lag window while their
    history is still intact.
    """

    def __init__(
        self,
        recency_size,
        coreset_size,
        observation_space,
        action_space,
        device,
        n_envs=1,
        n_step=10,
        gamma=0.99,
        frame_stack=4,
    ):
        if len(observation_space.shape) != 3 or observation_space.shape[0] != frame_stack:
            raise ValueError(f"expected ({frame_stack}, H, W) Atari observations, got {observation_space.shape}")
        if not hasattr(action_space, "n"):
            raise ValueError("EndpointReplayBuffer only supports discrete actions")
        if n_step < 1:
            raise ValueError("n_step must be positive")

        self.device = torch.device(device)
        self.n_envs = int(n_envs)
        self.n_step = int(n_step)
        self.gamma = float(gamma)
        self.frame_stack = int(frame_stack)
        self.height, self.width = observation_space.shape[-2:]

        # Recency ring, in per-env rows.
        self.recency_rows = max(int(np.ceil(recency_size / self.n_envs)), self.n_step + 1)
        self.recency_size = self.recency_rows * self.n_envs
        # The margin has to cover the lag window as well as the frame stack: a
        # window's first transition is evicted up to n_step - 1 steps before the
        # window fills and its stack is materialized, and that stack reaches
        # frame_stack - 1 frames further back still. Sizing the ring only for
        # the frame stack silently rebuilds those observations from whatever
        # newer frames have since landed in those rows.
        self.ring_rows = self.recency_rows + self.n_step + self.frame_stack + 2
        self.frames = np.zeros((self.ring_rows, self.n_envs, self.height, self.width), dtype=np.uint8)
        self.actions = np.zeros((self.ring_rows, self.n_envs), dtype=np.int64)
        self.next_actions = np.zeros((self.ring_rows, self.n_envs), dtype=np.int64)
        self.rewards = np.zeros((self.ring_rows, self.n_envs), dtype=np.float32)
        self.dones = np.zeros((self.ring_rows, self.n_envs), dtype=np.bool_)

        # Coreset, in whole rows (no per-env axis: chains are per env but pooled).
        self.coreset_size = int(coreset_size)
        self.coreset_obs = np.zeros(
            (self.coreset_size, self.frame_stack, self.height, self.width), dtype=np.uint8
        )
        self.coreset_next_obs = np.zeros(
            (self.coreset_size, self.frame_stack, self.height, self.width), dtype=np.uint8
        )
        self.coreset_actions = np.zeros(self.coreset_size, dtype=np.int64)
        self.coreset_next_actions = np.zeros(self.coreset_size, dtype=np.int64)
        self.coreset_returns = np.zeros(self.coreset_size, dtype=np.float32)
        self.coreset_discounts = np.zeros(self.coreset_size, dtype=np.float32)
        self.coreset_pos = 0
        self.coreset_count = 0

        # Per-env lag window: the indices of transitions waiting to be summarized.
        self.lag = [[] for _ in range(self.n_envs)]

        self.steps = 0
        self.initialized = False

    # -- insertion ---------------------------------------------------------

    @staticmethod
    def _numpy(value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
        return np.asarray(value)

    def _latest_frames(self, observations):
        if isinstance(observations, torch.Tensor):
            observations = observations[:, -1].detach().cpu().numpy()
        else:
            observations = np.asarray(observations)[:, -1]
        return observations.reshape(self.n_envs, self.height, self.width).astype(np.uint8, copy=False)

    def initialize(self, observations):
        self.frames[0] = self._latest_frames(observations)
        self.initialized = True

    def add(self, next_observations, actions, next_actions, rewards, dones):
        """Store one vector step, then hand any expired transitions to the lag windows."""
        if not self.initialized:
            raise RuntimeError("call buffer.initialize(initial_observations) before add")
        row = self.steps % self.ring_rows
        self.actions[row] = self._numpy(actions).reshape(self.n_envs)
        self.next_actions[row] = self._numpy(next_actions).reshape(self.n_envs)
        self.rewards[row] = self._numpy(rewards).reshape(self.n_envs)
        self.dones[row] = self._numpy(dones).reshape(self.n_envs).astype(bool, copy=False)
        self.frames[(self.steps + 1) % self.ring_rows] = self._latest_frames(next_observations)
        self.steps += 1

        expired = self.steps - self.recency_rows - 1
        if expired >= 0:
            self._push_to_lag(expired)

    def _push_to_lag(self, index):
        """Move transition `index` out of recency and into each env's lag window."""
        for env in range(self.n_envs):
            window = self.lag[env]
            window.append(index)
            terminated = bool(self.dones[index % self.ring_rows, env])
            if len(window) == self.n_step or terminated:
                self._emit_coreset_entry(env, window)
                window.clear()

    def _emit_coreset_entry(self, env, window):
        """Summarize a k-step window into one chained endpoint tuple."""
        first, last = window[0], window[-1]
        accumulated, discount = 0.0, 1.0
        for index in window:
            row = index % self.ring_rows
            accumulated += discount * float(self.rewards[row, env])
            # White (2017): the discount is zero on termination, so the product
            # collapses to zero exactly when the window ends in a terminal state
            # and the stored target needs no separate done flag.
            discount *= 0.0 if self.dones[row, env] else self.gamma

        slot = self.coreset_pos
        self.coreset_obs[slot] = self._stack(first, env)
        self.coreset_next_obs[slot] = self._stack(last + 1, env)
        self.coreset_actions[slot] = self.actions[first % self.ring_rows, env]
        self.coreset_next_actions[slot] = self.next_actions[last % self.ring_rows, env]
        self.coreset_returns[slot] = accumulated
        self.coreset_discounts[slot] = discount
        self.coreset_pos = (slot + 1) % self.coreset_size
        self.coreset_count = min(self.coreset_count + 1, self.coreset_size)

    def _stack(self, frame_index, env):
        """Rebuild the stacked observation whose newest frame is `frame_index`."""
        offsets = np.arange(self.frame_stack - 1, -1, -1)
        rows = (frame_index - offsets) % self.ring_rows
        stack = self.frames[rows, env].copy()
        # Zero out frames from before the start of the episode.
        for channel, offset in enumerate(offsets):
            if offset == 0:
                continue
            crossed = self.dones[
                (frame_index - np.arange(offset, 0, -1)) % self.ring_rows, env
            ].any()
            if crossed or frame_index - offset < 0:
                stack[channel] = 0
        return stack

    # -- sampling ----------------------------------------------------------

    def __len__(self):
        return self.recency_count + self.coreset_count

    @property
    def recency_count(self):
        return min(self.steps, self.recency_rows) * self.n_envs

    def _recency_indices(self, batch_size):
        newest = self.steps - 1
        oldest = max(self.steps - self.recency_rows, 0)
        indices = np.random.randint(oldest, newest + 1, size=batch_size)
        envs = np.random.randint(0, self.n_envs, size=batch_size)
        return indices, envs

    def sample_recency(self, batch_size):
        """One-step transitions (s, a, r, s', gamma) for the DDQN update."""
        indices, envs = self._recency_indices(batch_size)
        rows = indices % self.ring_rows
        observations = np.stack([self._stack(i, e) for i, e in zip(indices, envs)])
        next_observations = np.stack([self._stack(i + 1, e) for i, e in zip(indices, envs)])
        return (
            self._to_torch(observations),
            self._to_torch(self.actions[rows, envs]),
            self._to_torch(self.rewards[rows, envs]),
            self._to_torch(next_observations),
            self._to_torch(self.dones[rows, envs].astype(np.float32)),
        )

    def sample_coreset(self, batch_size):
        """Chained endpoints (s, a, g, gamma^k, s_end, a_end) for the Sarsa update."""
        if self.coreset_count == 0:
            return None
        indices = np.random.randint(0, self.coreset_count, size=batch_size)
        return (
            self._to_torch(self.coreset_obs[indices]),
            self._to_torch(self.coreset_actions[indices]),
            self._to_torch(self.coreset_returns[indices]),
            self._to_torch(self.coreset_discounts[indices]),
            self._to_torch(self.coreset_next_obs[indices]),
            self._to_torch(self.coreset_next_actions[indices]),
        )

    def _to_torch(self, array):
        return torch.as_tensor(array, device=self.device)


if __name__ == "__main__":
    process_start = time.perf_counter()
    args = tyro.cli(Args)
    if args.num_envs < 1:
        raise ValueError("num_envs must be positive")
    if not 0.0 < args.expectile < 1.0:
        raise ValueError("expectile must lie strictly between 0 and 1")
    if not 0 <= args.coreset_batch_size <= args.batch_size:
        raise ValueError("coreset_batch_size must fit inside batch_size")
    if args.benchmark_warmup_iterations < 0:
        raise ValueError("benchmark_warmup_iterations cannot be negative")
    if args.benchmark_measure_iterations < 1:
        raise ValueError("benchmark_measure_iterations must be positive")
    recency_batch_size = args.batch_size - args.coreset_batch_size
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

    # env setup
    if args.env_backend == "cule":
        env_device = resolve_cule_device(args.cule_device, device, args.num_envs)
        envs = make_cule_env(args.env_id, args.num_envs, env_device, args.seed, args.capture_video)
    elif args.env_backend == "envpool":
        if envpool is None:
            raise ImportError("EnvPool backend requested; install envpool or pass --env-backend cule")
        envs = RecordEpisodeStatistics(
            envpool.make(
                args.env_id,
                env_type="gym",
                num_envs=args.num_envs,
                episodic_life=True,
                reward_clip=True,
                seed=args.seed,
            )
        )
    elif args.env_backend == "gymnasium":
        envs = gym.vector.SyncVectorEnv(
            [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
        )
    else:
        raise ValueError(f"unsupported environment backend: {args.env_backend}")
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    q_network = QNetwork(envs).to(device)
    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate, eps=args.adam_eps)
    target_network = QNetwork(envs).to(device)
    target_network.load_state_dict(q_network.state_dict())

    rb = EndpointReplayBuffer(
        args.recency_buffer_size,
        args.coreset_buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        n_envs=args.num_envs,
        n_step=args.n_step,
        gamma=args.gamma,
    )
    start_time = time.time()

    def epsilon_greedy(observations, epsilon):
        """Algorithm 1 draws a' from pi_eps(s') at storage time, hence Sarsa."""
        greedy = torch.argmax(q_network(observations), dim=1)
        random_actions = torch.randint(envs.single_action_space.n, (args.num_envs,), device=device)
        explore = torch.rand(args.num_envs, device=device) < epsilon
        return torch.where(explore, random_actions, greedy)

    obs, _ = envs.reset(seed=args.seed)
    obs = to_tensor(obs, device)
    rb.initialize(obs)
    global_step = 0
    learner_updates = 0
    episode_stats = EpisodeStats(args.solve_window, args.solve_reward)
    next_target_update = args.target_network_frequency
    next_log_step = max(10000, args.num_envs)
    num_vector_steps = int(np.ceil(args.total_timesteps / args.num_envs))
    if args.benchmark:
        num_vector_steps = args.benchmark_warmup_iterations + args.benchmark_measure_iterations
    benchmark_start = benchmark_start_step = benchmark_start_updates = None

    epsilon = linear_schedule(
        args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, 0
    )
    with torch.no_grad():
        action = epsilon_greedy(obs.float(), epsilon)
    loss = torch.zeros((), device=device)

    for vector_step in range(num_vector_steps):
        if args.benchmark and vector_step == args.benchmark_warmup_iterations:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            benchmark_start = time.perf_counter()
            benchmark_start_step = global_step
            benchmark_start_updates = learner_updates
        if args.max_training_seconds and time.time() - start_time >= args.max_training_seconds:
            break

        next_obs, rewards, terminations, truncations, infos = step_env(envs, action)
        transition_dones = done_tensor(terminations, truncations, device).bool()
        next_obs = to_tensor(next_obs, device)
        global_step += args.num_envs

        epsilon = linear_schedule(
            args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step
        )
        with torch.no_grad():
            next_action = epsilon_greedy(next_obs.float(), epsilon)

        solved = False if args.benchmark else episode_stats.update(
            completed_episode_infos(infos, transition_dones), global_step, writer
        )
        rb.add(next_obs, action, next_action, rewards, transition_dones)
        obs, action = next_obs, next_action

        if global_step > args.learning_starts and vector_step % args.train_frequency == 0:
            observations, actions, step_rewards, next_observations, dones = rb.sample_recency(
                recency_batch_size
            )
            with torch.no_grad():
                # Double DQN: the online net picks the action, the target net scores it.
                best = torch.argmax(q_network(next_observations.float()), dim=1)
                bootstrap = target_network(next_observations.float()).gather(1, best.unsqueeze(1)).squeeze(1)
                recency_target = step_rewards + args.gamma * bootstrap * (1.0 - dones)
            predicted = q_network(observations.float()).gather(1, actions.unsqueeze(1)).squeeze(1)
            loss = ((recency_target - predicted) ** 2).mean()

            coreset = rb.sample_coreset(args.coreset_batch_size) if args.coreset_batch_size else None
            if coreset is not None:
                (
                    coreset_obs,
                    coreset_actions,
                    coreset_returns,
                    coreset_discounts,
                    coreset_next_obs,
                    coreset_next_actions,
                ) = coreset
                with torch.no_grad():
                    # Sarsa, not max: the stored a_end is the only action whose
                    # value at s_end is itself kept anchored by another chain
                    # element, so bootstrapping off any other action would query
                    # an unanchored value.
                    end_values = (
                        target_network(coreset_next_obs.float())
                        .gather(1, coreset_next_actions.unsqueeze(1))
                        .squeeze(1)
                    )
                    coreset_target = coreset_returns + coreset_discounts * end_values
                coreset_predicted = (
                    q_network(coreset_obs.float()).gather(1, coreset_actions.unsqueeze(1)).squeeze(1)
                )
                loss = loss + expectile_loss(coreset_predicted, coreset_target, args.expectile).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            learner_updates += 1

            if learner_updates >= next_target_update:
                target_network.load_state_dict(q_network.state_dict())
                next_target_update = (
                    learner_updates // args.target_network_frequency + 1
                ) * args.target_network_frequency

            if writer is not None and global_step >= next_log_step:
                sps = int(global_step / (time.time() - start_time))
                writer.add_scalar("losses/td_loss", loss.item(), global_step)
                writer.add_scalar("charts/SPS", sps, global_step)
                writer.add_scalar("charts/epsilon", epsilon, global_step)
                writer.add_scalar("buffers/recency", rb.recency_count, global_step)
                writer.add_scalar("buffers/coreset", rb.coreset_count, global_step)
                print("SPS:", sps)
                next_log_step = global_step + max(10000, args.num_envs)
        if solved:
            break

    if args.benchmark:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        benchmark_end = time.perf_counter()
        measured_steps = global_step - benchmark_start_step
        measured_updates = learner_updates - benchmark_start_updates
        measured_seconds = benchmark_end - benchmark_start
        result = {
            "algorithm": "endpoint_ddqn",
            "backend": args.env_backend,
            "batch_size": args.batch_size,
            "benchmark": "full_training_loop",
            "compile": False,
            "coreset_buffer_size": args.coreset_buffer_size,
            "env_id": args.env_id,
            "learner_updates": measured_updates,
            "measure_iterations": args.benchmark_measure_iterations,
            "measured_seconds": measured_seconds,
            "measured_steps": measured_steps,
            "n_step": args.n_step,
            "num_envs": args.num_envs,
            "peak_cuda_memory_mb": (
                torch.cuda.max_memory_allocated(device) / (1024**2) if device.type == "cuda" else 0.0
            ),
            "process_seconds": benchmark_end - process_start,
            "recency_buffer_size": args.recency_buffer_size,
            "schema_version": 1,
            "sps": measured_steps / measured_seconds,
            "ups": measured_updates / measured_seconds,
            "warmup_iterations": args.benchmark_warmup_iterations,
        }
        print(f"BENCHMARK_RESULT {json.dumps(result, sort_keys=True)}", flush=True)
    else:
        elapsed = time.time() - start_time
        print("SPS:", int(global_step / elapsed))
        print("learner updates:", learner_updates)
        print("coreset entries:", rb.coreset_count)
        episode_stats.print_summary()

    envs.close()
    if writer is not None:
        writer.close()
