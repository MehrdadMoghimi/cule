# torch.compile twin of r2d2_atari.py.
#
# R2D2: "Recurrent Experience Replay in Distributed Reinforcement Learning"
# (Kapturowski et al., ICLR 2019, https://openreview.net/forum?id=r1lyTjAqYX).
#
# DeepMind never released code for R2D2, so this is a reimplementation from the
# paper.  The distributed Ape-X actor/learner split is replaced by this repo's
# synchronous vectorised-env loop: the `num_envs` environments play the role of
# the actor pool, including Ape-X's per-actor epsilon ladder.
#
# Parents: rainbow_atari.py for the off-policy value machinery (prioritized
# replay, target network, double Q, n-step returns) and ppo_atari_lstm.py for
# the recurrent core.  What R2D2 adds on top of that pair:
#
#   * SEQUENCE replay -- the unit of storage is a length-(burn_in + seq_len)
#     slice of one environment's trajectory, not a single transition
#   * STORED recurrent state -- the LSTM state that actually produced the
#     behaviour is saved alongside each sequence, instead of restarting from
#     zeros at replay time
#   * BURN-IN -- the first `burn_in` steps of a replayed sequence are unrolled
#     without gradient, purely to let the stored (and now stale) state recover
#   * INVERTIBLE VALUE RESCALING -- h(x) = sign(x)(sqrt(|x|+1) - 1) + eps*x is
#     applied to the bootstrap so raw, unclipped rewards can be used
#   * MIXED PRIORITY -- p = eta * max_t |delta_t| + (1 - eta) * mean_t |delta_t|,
#     because a sequence has many TD errors and neither summary alone works
#   * the LSTM is fed the previous action and previous reward alongside the
#     convolutional features
#
# Supports gymnasium, cule, and envpool.  The trainer skeleton is adapted from
# CleanRL (https://github.com/vwxyzjn/cleanrl, MIT) and the compile / CUDA-graph
# structure follows LeanRL (https://github.com/meta-pytorch/LeanRL, MIT); both
# licenses are reproduced in cleanrl/LICENSE.md.
"""R2D2 on Atari with optional torch.compile and CUDA graphs.

The recurrent unroll is a fixed-length Python loop over `burn_in + seq_len +
n_step` steps, so it fully unrolls into the traced graph; both it and the
learner update are fixed-shape and capturable.  Sequence sampling and the
priority-tree writes stay eager, as in rainbow_atari_torchcompile.py.

Sequences may cross episode boundaries; the recurrent state is reset at those
boundaries during the replayed unroll rather than the sequence being rejected.
Requiring a done-free window would starve the buffer, because `EpisodicLifeEnv`
ends an "episode" on every lost life.
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
import torch.optim as optim
import tyro
from tensordict.nn import CudaGraphModule
from torch.utils.tensorboard import SummaryWriter

torch.set_float32_matmul_precision("high")

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
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)
from cleanrl_utils.buffers import AtariReplayBuffer, PrioritizedAtariReplayBuffer
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
    learning_rate: float = 1e-4
    """the learning rate of the optimizer"""
    adam_eps: float = 1e-3
    """Adam epsilon (the paper's 1e-3, far larger than the usual 1e-8)"""
    num_envs: int = 64
    """the number of parallel actors"""
    buffer_size: int = 100000
    """the replay memory size in individual transitions"""
    gamma: float = 0.997
    """the discount factor gamma"""
    n_step: int = 5
    """the number of steps in the bootstrapped return"""
    burn_in: int = 20
    """replayed steps unrolled without gradient to recover the recurrent state"""
    seq_len: int = 40
    """replayed steps that contribute to the loss"""
    lstm_size: int = 512
    """the LSTM hidden size"""
    batch_size: int = 32
    """the number of sequences per learner update"""
    target_network_frequency: int = 400
    """learner updates between target-network updates"""
    learning_starts: int = 50000
    """transitions collected before learning starts"""
    learner_updates_per_vector_step: float = 1.0
    """gradient updates accrued per vector environment step; may be fractional"""
    replay_ratio: float | None = None
    """sampled sequences per collected transition; overrides learner-updates-per-vector-step"""
    prioritized_replay_alpha: float = 0.9
    """alpha parameter for prioritized replay"""
    prioritized_replay_beta: float = 0.6
    """initial beta parameter for prioritized replay"""
    prioritized_replay_eps: float = 1e-6
    """epsilon added to priorities"""
    priority_eta: float = 0.9
    """mixing weight between the max and the mean absolute TD error of a sequence"""
    rescale_eps: float = 1e-3
    """the linear term of the invertible value rescaling h(x)"""
    actor_epsilon_base: float = 0.4
    """base of the Ape-X per-actor epsilon ladder, eps_i = base^(1 + 7i/(N-1))"""
    actor_epsilon_alpha: float = 7.0
    """exponent spread of the Ape-X per-actor epsilon ladder"""
    max_grad_norm: float = 40.0
    """the maximum gradient norm"""
    clip_rewards: bool = False
    """whether to sign-clip rewards; R2D2 rescales instead of clipping"""
    max_training_seconds: float = 0.0
    """wall-clock training limit; zero disables it"""
    solve_reward: float | None = None
    """stop when the moving episodic return reaches this value"""
    solve_window: int = 20
    """number of completed episodes in the solve moving average"""
    compile: bool = False
    """whether to compile the policy step and the learner update"""
    cudagraphs: bool = False
    """whether to wrap the policy step and the learner update in CudaGraphModule"""
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


def signed_hyperbolic(x: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    """h(x) = sign(x) (sqrt(|x| + 1) - 1) + eps * x.

    Compresses the value scale so a single set of hyperparameters works across
    games without clipping rewards; the eps*x term keeps h invertible.
    """
    return torch.sign(x) * (torch.sqrt(torch.abs(x) + 1.0) - 1.0) + eps * x


def signed_parabolic(x: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    """The exact inverse of `signed_hyperbolic`."""
    if eps == 0:
        return torch.sign(x) * (x * x + 2.0 * torch.abs(x))
    root = (torch.sqrt(1.0 + 4.0 * eps * (torch.abs(x) + 1.0 + eps)) - 1.0) / (2.0 * eps)
    return torch.sign(x) * (root * root - 1.0)


# ALGO LOGIC: initialize agent here:
class R2D2Network(nn.Module):
    """Nature CNN -> LSTM -> dueling head.

    The LSTM input is the convolutional feature vector concatenated with the
    one-hot previous action and the (rescaled) previous reward, which is what
    lets the recurrent state disambiguate states that look identical but follow
    different histories.
    """

    def __init__(self, env, lstm_size=512):
        super().__init__()
        self.n_actions = int(env.single_action_space.n)
        self.lstm_size = int(lstm_size)
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
        )
        self.lstm = nn.LSTMCell(512 + self.n_actions + 1, self.lstm_size)
        self.value_head = nn.Sequential(nn.Linear(self.lstm_size, 512), nn.ReLU(), nn.Linear(512, 1))
        self.advantage_head = nn.Sequential(
            nn.Linear(self.lstm_size, 512), nn.ReLU(), nn.Linear(512, self.n_actions)
        )

    def initial_state(self, batch_size, device):
        dtype = next(self.parameters()).dtype
        zeros = torch.zeros(batch_size, self.lstm_size, device=device, dtype=dtype)
        return zeros, zeros.clone()

    def step(self, observations, previous_actions, previous_rewards, state):
        """One recurrent step over a batch; returns (q_values, new_state)."""
        features = self.conv(observations / 255.0)
        context = torch.cat(
            [
                features,
                F.one_hot(previous_actions, self.n_actions).float(),
                previous_rewards.unsqueeze(-1),
            ],
            dim=-1,
        )
        hidden, cell = self.lstm(context, state)
        value = self.value_head(hidden)
        advantage = self.advantage_head(hidden)
        q_values = value + advantage - advantage.mean(dim=-1, keepdim=True)
        return q_values, (hidden, cell)

    def unroll(self, observations, previous_actions, previous_rewards, resets, state, grad_from=0):
        """Unroll over time, resetting the state wherever `resets` is set.

        `grad_from` implements burn-in: steps before it run under `no_grad`, so
        they only warm the recurrent state up and cost no memory.
        """
        outputs = []
        batch_size, horizon = previous_actions.shape
        for t in range(horizon):
            keep = (1.0 - resets[:, t]).unsqueeze(-1)
            state = (state[0] * keep, state[1] * keep)
            if t < grad_from:
                with torch.no_grad():
                    q_values, state = self.step(
                        observations[:, t], previous_actions[:, t], previous_rewards[:, t], state
                    )
                state = (state[0].detach(), state[1].detach())
            else:
                q_values, state = self.step(
                    observations[:, t], previous_actions[:, t], previous_rewards[:, t], state
                )
            outputs.append(q_values)
        return torch.stack(outputs, dim=1), state


class R2D2SequenceBuffer(PrioritizedAtariReplayBuffer):
    """Prioritized replay that serves whole sequences plus their stored state.

    Storage stays frame-efficient (one new 84x84 frame per transition, stacks
    rebuilt at sample time), following `SPRReplayBuffer`; on top of that it keeps
    the LSTM state that produced each step, so a replayed sequence can restart
    from the state the behaviour policy actually had.
    """

    def __init__(self, *pargs, burn_in: int, seq_len: int, lstm_size: int, **kwargs):
        super().__init__(*pargs, **kwargs)
        self.burn_in = int(burn_in)
        self.seq_len = int(seq_len)
        self.span = self.burn_in + self.seq_len
        # A start row becomes sampleable only once the whole sequence *and* the
        # n-step bootstrap window that follows it exist.
        self.sample_horizon = self.span + self.n_step
        self.hidden = np.zeros((self.time_capacity, self.n_envs, lstm_size), dtype=np.float32)
        self.cell = np.zeros_like(self.hidden)

    def add(self, next_observations, actions, rewards, dones, hidden, cell) -> None:
        # The state that produced this action belongs to the row the transition
        # is about to be written into, so record it before advancing `pos`.
        self.hidden[self.pos] = self._numpy(hidden)
        self.cell[self.pos] = self._numpy(cell)
        AtariReplayBuffer.add(self, next_observations, actions, rewards, dones)

        env_indices = np.arange(self.n_envs, dtype=np.int64)
        overwritten = self._flat_indices(np.full(self.n_envs, self.pos), env_indices)
        self.sum_tree.update(overwritten, np.zeros(self.n_envs, dtype=np.float32))

        if self.steps < self.sample_horizon:
            return
        candidate_row = (self.pos - self.sample_horizon) % self.time_capacity
        candidate_id = self.steps - self.sample_horizon
        if self.transition_ids[candidate_row] != candidate_id:
            return
        indices = self._flat_indices(np.full(self.n_envs, candidate_row), env_indices)
        priorities = np.full(self.n_envs, self.max_priority**self.alpha, dtype=np.float32)
        self.sum_tree.update(indices, priorities)

    def sample_sequences(self, batch_size: int):
        indices = self.sum_tree.sample(batch_size)
        rows = indices // self.n_envs
        env_indices = indices % self.n_envs
        start_ids = self.transition_ids[rows]

        horizon = self.span + self.n_step
        observation_stacks = []
        step_rows = np.empty((batch_size, horizon), dtype=np.int64)
        for offset in range(horizon):
            offset_rows = (rows + offset) % self.time_capacity
            step_rows[:, offset] = offset_rows
            observation_stacks.append(self._encode_stack(offset_rows, env_indices, start_ids + offset))
        observations = np.stack(observation_stacks, axis=1)  # (B, T, 4, 84, 84)

        actions = self.actions[step_rows, env_indices[:, None]]
        rewards = self.rewards[step_rows, env_indices[:, None]]
        dones = self.dones[step_rows, env_indices[:, None]].astype(np.float32)

        # `previous_*` at t=0 come from the transition before the sequence; the
        # stored recurrent state was produced with exactly those inputs.
        previous_rows = (rows - 1) % self.time_capacity
        previous_actions = np.concatenate(
            [self.actions[previous_rows, env_indices][:, None], actions[:, :-1]], axis=1
        )
        previous_rewards = np.concatenate(
            [self.rewards[previous_rows, env_indices][:, None], rewards[:, :-1]], axis=1
        )
        previous_dones = np.concatenate(
            [self.dones[previous_rows, env_indices][:, None].astype(np.float32), dones[:, :-1]], axis=1
        )
        # A transition preceding the sequence start may belong to another
        # episode or have been overwritten; treat it as a reset in that case.
        # `start_ids == 0` is checked explicitly because unwritten rows carry the
        # sentinel transition id -1, which would otherwise look like a valid
        # predecessor of the very first transition.
        stale = (
            (self.transition_ids[previous_rows] != start_ids - 1) | (start_ids == 0)
        ).astype(np.float32)
        previous_dones[:, 0] = np.maximum(previous_dones[:, 0], stale)
        previous_actions[:, 0] *= (1 - stale).astype(np.int64)
        previous_rewards[:, 0] *= 1 - stale

        probabilities = self.sum_tree.values(indices) / self.sum_tree.total
        weights = (max(self.num_transitions, 1) * probabilities) ** (-self.beta)
        weights /= weights.max()
        return dict(
            observations=self._to_torch(observations),
            actions=self._to_torch(actions),
            rewards=self._to_torch(rewards),
            dones=self._to_torch(dones),
            previous_actions=self._to_torch(previous_actions),
            previous_rewards=self._to_torch(previous_rewards),
            previous_dones=self._to_torch(previous_dones),
            hidden=self._to_torch(self.hidden[rows, env_indices]),
            cell=self._to_torch(self.cell[rows, env_indices]),
            weights=self._to_torch(weights.astype(np.float32)),
            indices=indices,
        )


def actor_epsilons(num_envs: int, base: float, alpha: float) -> np.ndarray:
    """Ape-X's per-actor epsilon ladder, eps_i = base ** (1 + i/(N-1) * alpha)."""
    if num_envs == 1:
        return np.array([base**(1.0 + alpha)], dtype=np.float32)
    fractions = np.arange(num_envs, dtype=np.float64) / (num_envs - 1)
    return (base ** (1.0 + fractions * alpha)).astype(np.float32)


def r2d2_targets(
    q_online, q_target, actions, rewards, dones, burn_in, seq_len, n_step, gamma, rescale_eps
):
    """Rescaled n-step double-Q targets for the graded part of a sequence.

    Returns targets and predictions of shape (B, seq_len).
    """
    horizon = slice(burn_in, burn_in + seq_len)
    predictions = q_online[:, horizon].gather(2, actions[:, horizon].unsqueeze(-1)).squeeze(-1)

    # Double Q: the online net picks the bootstrap action, the target net scores.
    bootstrap = slice(burn_in + n_step, burn_in + n_step + seq_len)
    best_actions = q_online[:, bootstrap].argmax(dim=2)
    bootstrap_values = q_target[:, bootstrap].gather(2, best_actions.unsqueeze(-1)).squeeze(-1)

    returns = torch.zeros_like(predictions)
    alive = torch.ones_like(predictions)
    discount = torch.ones_like(predictions)
    for offset in range(n_step):
        step = slice(burn_in + offset, burn_in + offset + seq_len)
        returns = returns + discount * rewards[:, step] * alive
        alive = alive * (1.0 - dones[:, step])
        discount = discount * gamma
    # h(sum_k gamma^k r_k + gamma^n h^-1(Q_target)): the inverse undoes the
    # compression before bootstrapping, and h re-applies it to the whole target.
    targets = signed_hyperbolic(
        returns + discount * alive * signed_parabolic(bootstrap_values, rescale_eps),
        rescale_eps,
    )
    return targets, predictions


if __name__ == "__main__":
    process_start = time.perf_counter()
    args = tyro.cli(Args)
    if args.num_envs < 1:
        raise ValueError("num_envs must be positive")
    if args.burn_in < 0:
        raise ValueError("burn_in cannot be negative")
    if args.seq_len < 1:
        raise ValueError("seq_len must be positive")
    if args.n_step < 1:
        raise ValueError("n_step must be positive")
    if not 0.0 <= args.priority_eta <= 1.0:
        raise ValueError("priority_eta must be in [0, 1]")
    if args.max_training_seconds < 0:
        raise ValueError("max_training_seconds must be non-negative")
    if args.benchmark_warmup_iterations < 0:
        raise ValueError("benchmark_warmup_iterations cannot be negative")
    if args.benchmark_measure_iterations < 1:
        raise ValueError("benchmark_measure_iterations must be positive")
    if args.replay_ratio is not None:
        if args.replay_ratio < 0:
            raise ValueError("replay_ratio must be non-negative")
        args.learner_updates_per_vector_step = args.replay_ratio * args.num_envs / args.batch_size
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__compile={args.compile}__cudagraphs={args.cudagraphs}"
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
    if args.cudagraphs and device.type != "cuda":
        raise ValueError("cudagraphs requires CUDA")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    # env setup.  R2D2 rescales values instead of clipping rewards, so reward
    # clipping is off by default on every backend.
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

    q_network = R2D2Network(envs, lstm_size=args.lstm_size).to(device)
    target_network = R2D2Network(envs, lstm_size=args.lstm_size).to(device)
    target_network.load_state_dict(q_network.state_dict())
    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate, eps=args.adam_eps)

    rb = R2D2SequenceBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        n_envs=args.num_envs,
        n_step=args.n_step,
        gamma=args.gamma,
        alpha=args.prioritized_replay_alpha,
        beta=args.prioritized_replay_beta,
        eps=args.prioritized_replay_eps,
        burn_in=args.burn_in,
        seq_len=args.seq_len,
        lstm_size=args.lstm_size,
    )

    # Ape-X epsilon ladder: actor 0 explores least, actor N-1 most.
    epsilons = torch.as_tensor(
        actor_epsilons(args.num_envs, args.actor_epsilon_base, args.actor_epsilon_alpha), device=device
    )

    def policy(observations, previous_actions_in, previous_rewards_in, hidden, cell, epsilon):
        q_values, (new_hidden, new_cell) = q_network.step(
            observations, previous_actions_in, previous_rewards_in, (hidden, cell)
        )
        greedy_actions = torch.argmax(q_values, dim=1)
        random_actions = torch.randint(n_actions, greedy_actions.shape, device=greedy_actions.device)
        explore = torch.rand(greedy_actions.shape, device=greedy_actions.device) < epsilon
        return torch.where(explore, random_actions, greedy_actions), new_hidden, new_cell

    def update(
        observations, actions_in, rewards_in, dones_in,
        previous_actions_in, previous_rewards_in, previous_dones_in,
        hidden, cell, weights,
    ):
        start_state = (hidden, cell)
        # Burn-in: the first `burn_in` steps only warm the state up.
        q_online, _ = q_network.unroll(
            observations, previous_actions_in, previous_rewards_in, previous_dones_in,
            start_state, grad_from=args.burn_in,
        )
        with torch.no_grad():
            q_target_values, _ = target_network.unroll(
                observations, previous_actions_in, previous_rewards_in, previous_dones_in,
                start_state, grad_from=observations.shape[1],
            )
            targets, _ = r2d2_targets(
                q_online.detach(), q_target_values, actions_in, rewards_in, dones_in,
                args.burn_in, args.seq_len, args.n_step, args.gamma, args.rescale_eps,
            )

        horizon = slice(args.burn_in, args.burn_in + args.seq_len)
        predictions = q_online[:, horizon].gather(2, actions_in[:, horizon].unsqueeze(-1)).squeeze(-1)
        td_errors = targets - predictions
        loss = (0.5 * td_errors.pow(2).mean(dim=1) * weights).mean()

        # Mixed priority: neither the max nor the mean alone summarises a
        # sequence well, so R2D2 interpolates between them.
        absolute = td_errors.detach().abs()
        priorities = (
            args.priority_eta * absolute.max(dim=1).values
            + (1.0 - args.priority_eta) * absolute.mean(dim=1)
        )
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(q_network.parameters(), args.max_grad_norm)
        optimizer.step()
        return loss.detach(), priorities, predictions.detach().mean()

    if args.compile:
        # mode=None: CuLE mutates its observation buffer in place after every
        # step, so avoid reduce-overhead's implicit CUDA graphs retaining it.
        policy = torch.compile(policy, mode=None, fullgraph=True)
        update = torch.compile(update, mode=None)

    if args.cudagraphs:
        # CudaGraphModule copies inputs into static buffers and clones outputs;
        # the target-network sync mutates module tensors in place, so replays
        # observe it.  Sequence sampling and priority writes stay outside.
        policy = CudaGraphModule(policy, warmup=20)
        update = CudaGraphModule(update, warmup=20)

    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    reset_result = envs.reset(seed=args.seed) if args.env_backend != "envpool" else envs.reset()
    raw_obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result
    obs = to_tensor(raw_obs, device)
    rb.initialize(obs)
    state = q_network.initial_state(args.num_envs, device)
    previous_actions = torch.zeros(args.num_envs, dtype=torch.long, device=device)
    previous_rewards = torch.zeros(args.num_envs, device=device)

    global_step = 0
    update_budget = 0.0
    learner_updates = 0
    episode_stats = EpisodeStats(args.solve_window, args.solve_reward)
    next_target_update = args.target_network_frequency
    next_log_step = max(10000, args.num_envs)
    num_vector_steps = int(np.ceil(args.total_timesteps / args.num_envs))
    if args.benchmark:
        num_vector_steps = args.benchmark_warmup_iterations + args.benchmark_measure_iterations
    benchmark_start = None
    benchmark_start_step = None
    benchmark_start_updates = None
    last_loss = None
    last_q_value = None
    for vector_step in range(num_vector_steps):
        if args.benchmark and vector_step == args.benchmark_warmup_iterations:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            benchmark_start = time.perf_counter()
            benchmark_start_step = global_step
            benchmark_start_updates = learner_updates
        if args.max_training_seconds and time.time() - start_time >= args.max_training_seconds:
            break

        rb.beta = min(
            1.0,
            args.prioritized_replay_beta
            + global_step * (1.0 - args.prioritized_replay_beta) / args.total_timesteps,
        )

        # ALGO LOGIC: put action logic here.  The state carried into the step is
        # exactly what the replay buffer stores for this row.
        acting_hidden, acting_cell = state
        if args.compile:
            torch.compiler.cudagraph_mark_step_begin()
        with torch.no_grad():
            actions, new_hidden, new_cell = policy(
                obs, previous_actions, previous_rewards, state[0], state[1], epsilons
            )
        state = (new_hidden, new_cell)

        # TRY NOT TO MODIFY: execute the game and log data.
        step_result = step_env(envs, actions)
        if len(step_result) == 5:
            next_obs_raw, rewards, terminations, truncations, infos = step_result
        else:
            next_obs_raw, rewards, terminations, infos = step_result
            truncations = np.zeros_like(np.asarray(terminations), dtype=bool)
        transition_dones = done_tensor(terminations, truncations, device).bool()
        rewards = to_tensor(rewards, device, torch.float32).view(-1)
        global_step += args.num_envs

        solved = False
        if not args.benchmark:
            solved = episode_stats.update(
                completed_episode_infos(infos, transition_dones), global_step, writer
            )

        rb.add(next_obs_raw, actions, rewards, transition_dones, acting_hidden, acting_cell)

        # Carry the recurrent state forward, zeroing it where an episode ended.
        keep = (~transition_dones).float().unsqueeze(-1)
        state = (state[0] * keep, state[1] * keep)
        previous_actions = torch.where(transition_dones, torch.zeros_like(actions), actions)
        previous_rewards = torch.where(transition_dones, torch.zeros_like(rewards), rewards)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = to_tensor(next_obs_raw, device)

        # ALGO LOGIC: training.
        if (
            global_step > args.learning_starts
            and rb.steps >= rb.sample_horizon
            and rb.sum_tree.total > 0
        ):
            update_budget += args.learner_updates_per_vector_step
            num_updates = int(update_budget)
            update_budget -= num_updates
            for _ in range(num_updates):
                data = rb.sample_sequences(args.batch_size)
                if args.compile:
                    torch.compiler.cudagraph_mark_step_begin()
                last_loss, priorities, last_q_value = update(
                    data["observations"],
                    data["actions"],
                    data["rewards"],
                    data["dones"],
                    data["previous_actions"],
                    data["previous_rewards"],
                    data["previous_dones"],
                    data["hidden"],
                    data["cell"],
                    data["weights"],
                )
                # Priority-tree writes stay outside the compiled learner; clone
                # before the next call may reuse the output storage.
                rb.update_priorities(data["indices"], priorities.clone().cpu().numpy())
            learner_updates += num_updates

            if learner_updates >= next_target_update:
                target_network.load_state_dict(q_network.state_dict())
                next_target_update = (
                    learner_updates // args.target_network_frequency + 1
                ) * args.target_network_frequency

            if not args.benchmark and global_step >= next_log_step and num_updates:
                sps = int(global_step / (time.time() - start_time))
                writer.add_scalar("losses/td_loss", last_loss.item(), global_step)
                writer.add_scalar("losses/q_values", last_q_value.item(), global_step)
                writer.add_scalar("charts/SPS", sps, global_step)
                writer.add_scalar("charts/learner_updates", learner_updates, global_step)
                writer.add_scalar("charts/beta", rb.beta, global_step)
                print("SPS:", sps)
                next_log_step = global_step + max(10000, args.num_envs)
        if solved:
            break

    if args.benchmark:
        if benchmark_start is None or benchmark_start_step is None or benchmark_start_updates is None:
            raise RuntimeError("benchmark measurement window did not start")
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        benchmark_end = time.perf_counter()
        measured_steps = global_step - benchmark_start_step
        measured_updates = learner_updates - benchmark_start_updates
        measured_seconds = benchmark_end - benchmark_start
        result = {
            "algorithm": "r2d2",
            "cudagraphs": args.cudagraphs,
            "backend": args.env_backend,
            "batch_size": args.batch_size,
            "benchmark": "full_training_loop",
            "burn_in": args.burn_in,
            "compile": args.compile,
            "env_device": str(getattr(envs, "device", "cpu")),
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
            "replay_backend": "numpy_frame_efficient_sequence_per",
            "replay_ratio": measured_updates * args.batch_size * args.seq_len / max(measured_steps, 1),
            "schema_version": 1,
            "seq_len": args.seq_len,
            "sps": measured_steps / measured_seconds,
            "ups": measured_updates / measured_seconds,
            "warmup_iterations": args.benchmark_warmup_iterations,
        }
        print(f"BENCHMARK_RESULT {json.dumps(result, sort_keys=True)}", flush=True)
    else:
        elapsed = time.time() - start_time
        print("SPS:", int(global_step / max(elapsed, 1e-9)))
        print("learner updates:", learner_updates)
        episode_stats.print_summary()

    envs.close()
    if writer is not None:
        writer.close()
