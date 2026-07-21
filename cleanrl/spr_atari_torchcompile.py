# SPR: Data-Efficient Reinforcement Learning with Self-Predictive
# Representations (Schwarzer et al., 2021, https://arxiv.org/abs/2007.05929).
# Ported from the official author implementation
# (https://github.com/mila-iqia/spr, default scripts/run.py configuration): Rainbow (noisy nets std 0.5, dueling with
# hidden size 256, double DQN, C51, PER alpha 0.5 with beta 0.4 -> 1 over
# 100K) at n-step 10, batch 32, two updates per environment step, target sync
# every update, Adam 1e-4 with eps 1.5e-4, grad-norm clip 10, learning starts
# at 2000 with epsilon 1 -> 0 over the next 2001 steps, plus the SPR
# objective: a convolutional transition model rolled K=5 steps from the
# online latent, a q_l1 projection (the deterministic first dueling layers),
# a linear predictor, an EMA target encoder/projection (tau 0.01), and a
# normalized-L2 latent-matching loss with weight 5, masked past terminals.
# Shift+intensity augmentation is applied to learner inputs and (as in the
# official target_augmentation=1 setting) to behavior-policy observations.
# Structure follows rainbow_atari.py, which is adapted from CleanRL
# (https://github.com/vwxyzjn/cleanrl, MIT); the compile / CUDA-graph structure
# follows LeanRL (https://github.com/meta-pytorch/LeanRL, MIT).  Both licenses
# are reproduced in cleanrl/LICENSE.md.  Supports gymnasium, cule, and envpool.
"""SPR with optional torch.compile and CUDA graphs.

The sequence replay stays on the host (frame-efficient NumPy storage with
PER); each sampled batch is a fixed-shape GPU TensorDict, so the policy and
the full learner update (C51 loss, K-step transition rollout, SPR loss, and
both gradient clips) can be compiled and captured.  Priority-tree writes, the
EMA update, and the target-network sync stay outside the capture; the latter
two mutate module tensors in place, which graph replays observe.
"""
import copy
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
from tensordict import TensorDict
from tensordict.nn import CudaGraphModule
from torch.utils.tensorboard import SummaryWriter

torch.set_float32_matmul_precision("high")

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
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    env_backend: str = "cule"
    """environment backend: `cule`, `envpool`, or `gymnasium`"""
    cule_device: str = "auto"
    """CuLE device; auto uses CUDA for 32+ envs and CPU for smaller batches"""
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""

    env_id: str = "BreakoutNoFrameskip-v4"
    """the id of the environment"""
    total_timesteps: int = 100000
    """total timesteps of the experiments (Atari-100K benchmark budget)"""
    learning_rate: float = 1e-4
    """the learning rate of the optimizer"""
    adam_eps: float = 1.5e-4
    """the epsilon of the Adam optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    buffer_size: int = 1000000
    """the replay memory buffer size (effectively unbounded on 100K steps)"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 1.0
    """the target network update rate"""
    target_network_frequency: int = 1
    """learner updates between target-network updates (SPR syncs every update)"""
    batch_size: int = 32
    """the batch size of sample from the reply memory"""
    end_e: float = 0.0
    """the final epsilon for exploration (noisy nets drive exploration)"""
    epsilon_decay_period: int = 2001
    """steps after learning starts over which epsilon decays from 1 to end-e"""
    learning_starts: int = 2000
    """timestep to start learning"""
    learner_updates_per_vector_step: float = 2.0
    """gradient updates accrued per vector environment step (official replay ratio 64 / batch 32)"""
    replay_ratio: float | None = None
    """sampled replay items per collected transition; overrides learner-updates-per-vector-step"""
    max_training_seconds: float = 0.0
    """wall-clock training limit; zero disables it"""
    solve_reward: float | None = None
    """stop when the moving episodic return reaches this value"""
    solve_window: int = 20
    """number of completed episodes in the solve moving average"""
    n_step: int = 10
    """the number of steps to look ahead for n-step Q learning"""
    jumps: int = 5
    """the number of transition-model prediction steps (K)"""
    spr_weight: float = 5.0
    """weight of the SPR latent-prediction loss (official model-spr-weight)"""
    momentum_tau: float = 0.01
    """EMA rate of the SPR target encoder and projection"""
    max_grad_norm: float = 10.0
    """gradient-norm clip, applied per official parameter group"""
    prioritized_replay_alpha: float = 0.5
    """alpha parameter for prioritized replay"""
    prioritized_replay_beta: float = 0.4
    """initial beta for prioritized replay, annealed to 1 over total-timesteps"""
    prioritized_replay_eps: float = 1e-6
    """epsilon added to priorities"""
    n_atoms: int = 51
    """the number of atoms"""
    v_min: float = -10
    """the return lower bound"""
    v_max: float = 10
    """the return upper bound"""
    hidden_size: int = 256
    """the hidden width of the dueling heads (official dqn-hidden-size)"""
    noisy_std: float = 0.5
    """initial noisy-net sigma"""
    data_augmentation: bool = True
    """apply shift/intensity augmentation to learner inputs"""
    policy_augmentation: bool = True
    """augment behavior-policy observations (official target-augmentation)"""
    compile: bool = False
    """whether to compile the fixed-shape policy and learner regions"""
    cudagraphs: bool = False
    """whether to wrap the policy and learner update in CudaGraphModule"""
    benchmark: bool = False
    """run a fixed warmup/measurement window and print a JSON benchmark result"""
    benchmark_warmup_iterations: int = 10
    """vector-environment steps excluded from benchmark timing"""
    benchmark_measure_iterations: int = 30
    """vector-environment steps included in benchmark timing"""


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
            return {
                "final_info": [
                    {"episode": {"r": float(infos["r"][index]), "l": int(infos["l"][index])}}
                    for index in np.flatnonzero(game_over)
                ]
            }
    return {}


def shift_intensity_augmentation(images: torch.Tensor, pad: int = 4, intensity_scale: float = 0.05) -> torch.Tensor:
    """SPR's shift + intensity augmentation (replication pad, random crop, noise)."""
    batch, _, height, width = images.shape
    padded = F.pad(images, (pad, pad, pad, pad), mode="replicate")
    offsets_h = torch.randint(0, 2 * pad + 1, (batch,), device=images.device)
    offsets_w = torch.randint(0, 2 * pad + 1, (batch,), device=images.device)
    rows = offsets_h.view(batch, 1) + torch.arange(height, device=images.device).view(1, height)
    cols = offsets_w.view(batch, 1) + torch.arange(width, device=images.device).view(1, width)
    batch_indices = torch.arange(batch, device=images.device).view(batch, 1, 1)
    cropped = padded[batch_indices, :, rows.unsqueeze(-1), cols.unsqueeze(1)].permute(0, 3, 1, 2)
    noise = 1.0 + intensity_scale * torch.randn(batch, 1, 1, 1, device=images.device).clamp_(-2.0, 2.0)
    return cropped * noise


def linearly_decaying_epsilon(decay_period: int, step: int, warmup_steps: int, epsilon: float) -> float:
    """Dopamine-style schedule: 1 until warmup, then linear decay to epsilon."""
    steps_left = decay_period + warmup_steps - step
    bonus = (1.0 - epsilon) * steps_left / max(decay_period, 1)
    bonus = min(max(bonus, 0.0), 1.0 - epsilon)
    return epsilon + bonus


def renormalize(latent: torch.Tensor) -> torch.Tensor:
    """Per-sample min-max normalization over all latent dimensions (official)."""
    flat = latent.flatten(1)
    minimum = flat.min(dim=1, keepdim=True).values
    maximum = flat.max(dim=1, keepdim=True).values
    flat = (flat - minimum) / (maximum - minimum + 1e-8)
    return flat.view_as(latent)


class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer("bias_epsilon", torch.empty(out_features))
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    def reset_noise(self):
        self.weight_epsilon.normal_()
        self.bias_epsilon.normal_()

    def forward(self, input):
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(input, weight, bias)

    def deterministic(self, input):
        """Noise-free forward, used by the q_l1 SPR projection."""
        return F.linear(input, self.weight_mu, self.bias_mu)


class TransitionModel(nn.Module):
    """Official SPR convolutional dynamics model (zero residual blocks)."""

    def __init__(self, channels: int, n_actions: int):
        super().__init__()
        self.n_actions = n_actions
        self.network = nn.Sequential(
            nn.Conv2d(channels + n_actions, channels, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(channels),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, latent, action):
        action_planes = torch.zeros(
            latent.shape[0], self.n_actions, latent.shape[-2], latent.shape[-1], device=latent.device
        )
        action_planes[torch.arange(latent.shape[0], device=latent.device), action] = 1.0
        next_latent = F.relu(self.network(torch.cat([latent, action_planes], dim=1)))
        return renormalize(next_latent)


class SPRNetwork(nn.Module):
    def __init__(self, env, n_atoms, v_min, v_max, hidden_size=256, noisy_std=0.5):
        super().__init__()
        self.n_atoms = int(n_atoms)
        self.v_min = v_min
        self.v_max = v_max
        self.delta_z = (v_max - v_min) / (n_atoms - 1)
        self.n_actions = int(env.single_action_space.n)
        self.register_buffer("support", torch.linspace(v_min, v_max, n_atoms))

        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
        )
        conv_output_size = 64 * 7 * 7
        self.value_hidden = NoisyLinear(conv_output_size, hidden_size, noisy_std)
        self.value_out = NoisyLinear(hidden_size, n_atoms, noisy_std)
        self.advantage_hidden = NoisyLinear(conv_output_size, hidden_size, noisy_std)
        self.advantage_out = NoisyLinear(hidden_size, n_atoms * self.n_actions, noisy_std)

        self.transition_model = TransitionModel(64, self.n_actions)
        # q_l1 projection is the deterministic first dueling layers
        # (value 256 + advantage 256); the predictor is a single linear map.
        self.predictor = nn.Linear(2 * hidden_size, 2 * hidden_size)

    def encode(self, x):
        return renormalize(self.conv(x / 255.0))

    def q_dist(self, latent, log=False):
        h = latent.flatten(1)
        value = self.value_out(F.relu(self.value_hidden(h))).view(-1, 1, self.n_atoms)
        advantage = self.advantage_out(F.relu(self.advantage_hidden(h))).view(
            -1, self.n_actions, self.n_atoms
        )
        q_atoms = value + advantage - advantage.mean(dim=1, keepdim=True)
        return F.log_softmax(q_atoms, dim=2) if log else F.softmax(q_atoms, dim=2)

    def project(self, latent):
        h = latent.flatten(1)
        return torch.cat([self.value_hidden.deterministic(h), self.advantage_hidden.deterministic(h)], dim=-1)

    def reset_noise(self):
        for layer in (self.value_hidden, self.value_out, self.advantage_hidden, self.advantage_out):
            layer.reset_noise()


class SPRReplayBuffer(PrioritizedAtariReplayBuffer):
    """Prioritized n-step replay that additionally serves K-step sequences."""

    def __init__(self, *pargs, jumps: int, **kwargs):
        super().__init__(*pargs, **kwargs)
        self.jumps = int(jumps)
        # Candidates become sampleable only once every row of the RL n-step
        # window and the K future observations exist in the buffer.
        self.sample_horizon = max(self.n_step, self.jumps)

    def add(self, next_observations, actions, rewards, dones) -> None:
        # Reimplements the parent's candidate logic with the longer horizon.
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

        valid = np.ones(self.n_envs, dtype=np.bool_)
        for offset in range(self.n_step - 1):
            row = (candidate_row + offset) % self.time_capacity
            valid &= ~self.dones[row]
        indices = self._flat_indices(np.full(self.n_envs, candidate_row), env_indices)
        priorities = np.where(valid, self.max_priority**self.alpha, 0.0).astype(np.float32)
        self.sum_tree.update(indices, priorities)

    def sample_spr(self, batch_size: int):
        indices = self.sum_tree.sample(batch_size)
        rows = indices // self.n_envs
        env_indices = indices % self.n_envs
        samples = self._encode_samples(rows, env_indices, self.n_step, self.gamma)

        start_ids = self.transition_ids[rows]
        future_stacks = []
        for k in range(1, self.jumps + 1):
            future_rows = (rows + k) % self.time_capacity
            future_stacks.append(self._encode_stack(future_rows, env_indices, start_ids + k))
        future_observations = self._to_torch(np.stack(future_stacks, axis=1))  # (B, K, 4, 84, 84)

        action_rows = np.stack(
            [(rows + k) % self.time_capacity for k in range(self.jumps)], axis=1
        )  # (B, K)
        action_sequences = self._to_torch(self.actions[action_rows, env_indices[:, None]])

        # nonterminal[k] == 1 while no terminal occurred in transitions t..t+k-1
        done_rows = action_rows
        step_dones = self.dones[done_rows, env_indices[:, None]]  # (B, K)
        nonterminal = np.concatenate(
            [np.ones((batch_size, 1), dtype=np.float32), 1.0 - np.sign(np.cumsum(step_dones, axis=1))],
            axis=1,
        )  # (B, K+1)

        probabilities = self.sum_tree.values(indices) / self.sum_tree.total
        weights = (max(self.num_transitions, 1) * probabilities) ** (-self.beta)
        weights /= weights.max()
        return (
            samples,
            future_observations,
            action_sequences,
            self._to_torch(nonterminal.astype(np.float32)),
            indices,
            self._to_torch(weights.astype(np.float32).reshape(-1, 1)),
        )


if __name__ == "__main__":
    process_start = time.perf_counter()
    args = tyro.cli(Args)
    if args.learner_updates_per_vector_step < 0:
        raise ValueError("learner_updates_per_vector_step must be non-negative")
    if args.replay_ratio is not None:
        if args.replay_ratio < 0:
            raise ValueError("replay_ratio must be non-negative")
        args.learner_updates_per_vector_step = args.replay_ratio * args.num_envs / args.batch_size
    if args.max_training_seconds < 0:
        raise ValueError("max_training_seconds must be non-negative")
    if args.num_envs < 1:
        raise ValueError("num_envs must be positive")
    if args.jumps < 1:
        raise ValueError("jumps must be positive")
    if args.epsilon_decay_period < 1:
        raise ValueError("epsilon_decay_period must be positive")
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
    n_actions = int(envs.single_action_space.n)

    q_network = SPRNetwork(envs, args.n_atoms, args.v_min, args.v_max, args.hidden_size, args.noisy_std).to(device)
    target_network = SPRNetwork(
        envs, args.n_atoms, args.v_min, args.v_max, args.hidden_size, args.noisy_std
    ).to(device)
    target_network.load_state_dict(q_network.state_dict())
    # EMA target encoder and q_l1 projection for the SPR loss (momentum_tau).
    ema_encoder = copy.deepcopy(q_network.conv).requires_grad_(False)
    ema_value_hidden = copy.deepcopy(q_network.value_hidden).requires_grad_(False)
    ema_advantage_hidden = copy.deepcopy(q_network.advantage_hidden).requires_grad_(False)

    def ema_project(latent):
        h = latent.flatten(1)
        return torch.cat(
            [ema_value_hidden.deterministic(h), ema_advantage_hidden.deterministic(h)], dim=-1
        )

    optimizer = optim.Adam(
        q_network.parameters(),
        lr=args.learning_rate,
        eps=args.adam_eps,
        capturable=args.cudagraphs and not args.compile,
    )
    # The official implementation clips the stem (encoder + heads) and the
    # dynamics model as separate groups.
    stem_parameters = (
        list(q_network.conv.parameters())
        + list(q_network.value_hidden.parameters())
        + list(q_network.value_out.parameters())
        + list(q_network.advantage_hidden.parameters())
        + list(q_network.advantage_out.parameters())
        + list(q_network.predictor.parameters())
    )
    dynamics_parameters = list(q_network.transition_model.parameters())

    rb = SPRReplayBuffer(
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
        jumps=args.jumps,
    )

    def policy(observations: torch.Tensor, epsilon: torch.Tensor) -> torch.Tensor:
        policy_obs = observations.float()
        if args.policy_augmentation:
            policy_obs = shift_intensity_augmentation(policy_obs)
        latent = q_network.encode(policy_obs)
        q_values = torch.sum(q_network.q_dist(latent) * q_network.support, dim=2)
        greedy_actions = torch.argmax(q_values, dim=1)
        random_actions = torch.randint(n_actions, greedy_actions.shape, device=greedy_actions.device)
        explore = torch.rand(greedy_actions.shape, device=greedy_actions.device) < epsilon
        return torch.where(explore, random_actions, greedy_actions)

    def update(data: TensorDict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        observations = data["observations"].float()
        next_observations = data["next_observations"].float()
        if args.data_augmentation:
            observations = shift_intensity_augmentation(observations)
            next_observations = shift_intensity_augmentation(next_observations)
        actions = data["actions"]
        rewards = data["rewards"]
        dones = data["dones"]
        future_observations = data["future_observations"]
        action_sequences = data["action_sequences"]
        nonterminal = data["nonterminal"]
        weights = data["weights"].squeeze(-1)
        batch_indices = torch.arange(observations.shape[0], device=observations.device)

        # --- C51 n-step RL loss (double DQN) ---
        with torch.no_grad():
            next_latent_target = target_network.encode(next_observations)
            next_dist = target_network.q_dist(next_latent_target)
            support = target_network.support
            next_latent_online = q_network.encode(next_observations)
            next_q_online = torch.sum(q_network.q_dist(next_latent_online) * support, dim=2)
            best_actions = torch.argmax(next_q_online, dim=1)
            next_pmfs = next_dist[batch_indices, best_actions]

            gamma_n = args.gamma**args.n_step
            next_atoms = rewards + gamma_n * support * (1 - dones.float())
            tz = next_atoms.clamp(q_network.v_min, q_network.v_max)
            b = (tz - q_network.v_min) / q_network.delta_z
            l = b.floor().clamp(0, args.n_atoms - 1)
            u = b.ceil().clamp(0, args.n_atoms - 1)
            d_m_l = (u.float() + (l == b).float() - b) * next_pmfs
            d_m_u = (b - l) * next_pmfs
            target_pmfs = torch.zeros_like(next_pmfs)
            target_pmfs.scatter_add_(1, l.long(), d_m_l)
            target_pmfs.scatter_add_(1, u.long(), d_m_u)

        latent = q_network.encode(observations)
        log_dist = q_network.q_dist(latent, log=True)
        log_pred = log_dist[batch_indices, actions.flatten()]
        rl_loss_per_sample = -(target_pmfs * log_pred).sum(dim=1)

        # --- SPR loss: roll the transition model K steps ---
        spr_losses = []
        rolled = latent
        for k in range(args.jumps):
            rolled = q_network.transition_model(rolled, action_sequences[:, k])
            projection = q_network.predictor(q_network.project(rolled))
            with torch.no_grad():
                target_obs = future_observations[:, k].float()
                if args.data_augmentation:
                    target_obs = shift_intensity_augmentation(target_obs)
                target_latent = renormalize(ema_encoder(target_obs / 255.0))
                target_projection = ema_project(target_latent)
            f_online = F.normalize(projection, p=2.0, dim=-1, eps=1e-3)
            f_target = F.normalize(target_projection, p=2.0, dim=-1, eps=1e-3)
            spr_losses.append(F.mse_loss(f_online, f_target, reduction="none").sum(-1))
        spr_loss_per_sample = (torch.stack(spr_losses, dim=1) * nonterminal[:, 1:]).mean(dim=1)

        loss = (rl_loss_per_sample * weights).mean() + args.spr_weight * (
            spr_loss_per_sample * weights
        ).mean()

        # priorities: clamped KL divergence, as in the official code
        with torch.no_grad():
            safe_target = target_pmfs.clamp(min=1e-6)
            kl = (safe_target * (safe_target.log() - log_pred)).sum(dim=1)
            kl = kl.clamp(1e-6, 1e6)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(stem_parameters, args.max_grad_norm)
        torch.nn.utils.clip_grad_norm_(dynamics_parameters, args.max_grad_norm)
        optimizer.step()
        q_value = (log_pred.exp() * q_network.support).sum(dim=1).mean()
        return (
            rl_loss_per_sample.detach().mean(),
            spr_loss_per_sample.detach().mean(),
            kl,
            q_value.detach(),
        )

    if args.compile:
        # CuLE observation storage is updated in place, so do not use
        # reduce-overhead's implicit CUDA graphs here.
        policy = torch.compile(policy, mode=None, fullgraph=True)
        update = torch.compile(update, mode=None)

    if args.cudagraphs:
        # CudaGraphModule copies inputs into static buffers and clones outputs.
        # Noisy resets, the EMA update, and the target sync mutate module
        # tensors in place, so graph replays observe them.
        policy = CudaGraphModule(policy, warmup=20)
        update = CudaGraphModule(update, warmup=20)

    epsilon_tensor = torch.zeros((), device=device)
    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    reset_result = envs.reset(seed=args.seed) if args.env_backend != "envpool" else envs.reset()
    obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result
    rb.initialize(obs)
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
    for vector_step in range(num_vector_steps):
        if args.benchmark and vector_step == args.benchmark_warmup_iterations:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            benchmark_start = time.perf_counter()
            benchmark_start_step = global_step
            benchmark_start_updates = learner_updates
        if args.max_training_seconds and time.time() - start_time >= args.max_training_seconds:
            break
        # anneal PER beta to 1 (official pri_beta_steps == the 100K budget)
        rb.beta = min(
            1.0,
            args.prioritized_replay_beta
            + global_step * (1.0 - args.prioritized_replay_beta) / args.total_timesteps,
        )

        # ALGO LOGIC: epsilon-greedy over the noisy-augmented network
        epsilon_tensor.fill_(
            linearly_decaying_epsilon(
                args.epsilon_decay_period, global_step, args.learning_starts, args.end_e
            )
        )
        q_network.reset_noise()
        if args.compile:
            torch.compiler.cudagraph_mark_step_begin()
        with torch.no_grad():
            actions = policy(to_tensor(obs, device), epsilon_tensor)

        # TRY NOT TO MODIFY: execute the game and log data.
        step_result = step_env(envs, actions)
        if len(step_result) == 5:
            next_obs, rewards, terminations, truncations, infos = step_result
        else:
            next_obs, rewards, terminations, infos = step_result
            truncations = np.zeros_like(np.asarray(terminations), dtype=bool)
        transition_dones = done_tensor(terminations, truncations, device).bool()
        global_step += args.num_envs

        solved = False
        if not args.benchmark:
            solved = episode_stats.update(
                completed_episode_infos(infos, transition_dones), global_step, writer
            )

        rb.add(next_obs, actions, rewards, transition_dones)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if (
            global_step > args.learning_starts
            and len(rb) >= args.batch_size
            and rb.sum_tree.total > 0
        ):
            update_budget += args.learner_updates_per_vector_step
            num_updates = int(update_budget)
            update_budget -= num_updates
            for _ in range(num_updates):
                q_network.reset_noise()
                target_network.reset_noise()
                data, future_observations, action_sequences, nonterminal, indices, weights = (
                    rb.sample_spr(args.batch_size)
                )
                batch = TensorDict(
                    {
                        "observations": data.observations,
                        "next_observations": data.next_observations,
                        "actions": data.actions,
                        "rewards": data.rewards,
                        "dones": data.dones,
                        "future_observations": future_observations,
                        "action_sequences": action_sequences,
                        "nonterminal": nonterminal,
                        "weights": weights,
                    },
                    batch_size=[args.batch_size],
                    device=device,
                )
                if args.compile:
                    torch.compiler.cudagraph_mark_step_begin()
                last_rl_loss, last_spr_loss, kl, last_q_value = update(batch)
                rb.update_priorities(indices, kl.cpu().numpy())

                # EMA update of the SPR target encoder and projection
                with torch.no_grad():
                    for ema_module, online_module in (
                        (ema_encoder, q_network.conv),
                        (ema_value_hidden, q_network.value_hidden),
                        (ema_advantage_hidden, q_network.advantage_hidden),
                    ):
                        for ema_param, online_param in zip(
                            ema_module.parameters(), online_module.parameters()
                        ):
                            ema_param.lerp_(online_param, args.momentum_tau)
            learner_updates += num_updates

            # update target network
            if learner_updates >= next_target_update:
                for target_param, param in zip(target_network.parameters(), q_network.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1.0 - args.tau) * target_param.data)
                next_target_update = (
                    learner_updates // args.target_network_frequency + 1
                ) * args.target_network_frequency

            if not args.benchmark and global_step >= next_log_step and num_updates:
                sps = int(global_step / (time.time() - start_time))
                writer.add_scalar("losses/rl_loss", last_rl_loss.item(), global_step)
                writer.add_scalar("losses/spr_loss", last_spr_loss.item(), global_step)
                writer.add_scalar("losses/q_values", last_q_value.item(), global_step)
                writer.add_scalar("charts/SPS", sps, global_step)
                writer.add_scalar("charts/epsilon", epsilon_tensor.item(), global_step)
                writer.add_scalar("charts/learner_updates", learner_updates, global_step)
                writer.add_scalar(
                    "charts/effective_utd",
                    learner_updates / max(global_step - args.learning_starts, 1),
                    global_step,
                )
                writer.add_scalar(
                    "charts/replay_ratio",
                    learner_updates * args.batch_size / max(global_step - args.learning_starts, 1),
                    global_step,
                )
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
            "algorithm": "spr",
            "backend": args.env_backend,
            "batch_size": args.batch_size,
            "benchmark": "full_training_loop",
            "compile": args.compile,
            "cudagraphs": args.cudagraphs,
            "data_augmentation": args.data_augmentation,
            "env_device": str(getattr(envs, "device", "cpu")),
            "env_id": args.env_id,
            "jumps": args.jumps,
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
            "replay_backend": "numpy_frame_efficient_per_sequences",
            "replay_ratio": measured_updates * args.batch_size / max(measured_steps, 1),
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
        print("UPS:", learner_updates / elapsed)
        print("effective UTD:", learner_updates / max(global_step - args.learning_starts, 1))
        print(
            "replay ratio:",
            learner_updates * args.batch_size / max(global_step - args.learning_starts, 1),
        )
        episode_stats.print_summary()

    if args.save_model and not args.benchmark:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        model_data = {
            "model_weights": q_network.state_dict(),
            "args": vars(args),
        }
        torch.save(model_data, model_path)
        print(f"model saved to {model_path}")

    envs.close()
    if writer is not None:
        writer.close()
