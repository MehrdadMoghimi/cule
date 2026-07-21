# BBF: Bigger, Better, Faster — Human-level Atari with human-level efficiency
# (Schwarzer et al., 2023, https://arxiv.org/abs/2305.19452).
# Ported from the official implementation in the google-research monorepo
# (https://github.com/google-research/google-research, subdirectory
# bigger_better_faster/bbf: configs/BBF.gin, agents/spr_agent.py,
# spr_networks.py): an Impala-CNN encoder (width scale 4, two residual blocks
# per stage, min-max renormalized latents), a 2048-unit projection acting as
# the shared hidden layer of dueling C51 heads (no noisy nets), an SPR
# objective (K=5, weight 5, linear predictor) whose targets come from the
# single EMA target network (tau 0.005, updated every step), action selection
# from the target network (target_action_selection), DrQ augmentation,
# prioritized replay, AdamW (lr 1e-4, eps 1.5e-4, weight decay 0.1),
# exponential anneals of the update horizon 10 -> 3 and gamma 0.97 -> 0.997
# over the 10k gradient steps after each reset, and shrink-and-perturb resets
# (0.5 old + 0.5 random for encoder and transition model, hard resets for
# projection/predictor/heads) every 20k gradient steps.
# Structure follows rainbow_atari.py, which is adapted from CleanRL
# (https://github.com/vwxyzjn/cleanrl, MIT); the compile / CUDA-graph structure
# follows LeanRL (https://github.com/meta-pytorch/LeanRL, MIT).  Both licenses
# are reproduced in cleanrl/LICENSE.md.  Supports gymnasium, cule, and envpool.
"""BBF with optional torch.compile and CUDA graphs.

The sequence replay stays on the host; each sampled batch is a fixed-shape
GPU TensorDict (the annealed discount enters as a tensor), so the policy and
the learner update can be compiled and captured.  Shrink-and-perturb resets
mutate parameters in place and zero the optimizer moments in place under
CUDA graphs, keeping captured graphs valid across resets.
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
    """the learning rate of the AdamW optimizer"""
    adam_eps: float = 1.5e-4
    """the epsilon of the AdamW optimizer"""
    weight_decay: float = 0.1
    """the AdamW weight decay"""
    num_envs: int = 1
    """the number of parallel game environments"""
    buffer_size: int = 200000
    """the replay memory buffer size"""
    gamma: float = 0.997
    """the final discount factor gamma"""
    min_gamma: float = 0.97
    """the initial discount factor after each reset"""
    n_step: int = 3
    """the final n-step horizon"""
    max_n_step: int = 10
    """the initial n-step horizon after each reset"""
    cycle_steps: int = 10000
    """gradient steps over which gamma and the horizon anneal after each reset"""
    reset_every: int = 20000
    """gradient steps between shrink-and-perturb resets"""
    no_resets_after: int = 100000
    """gradient step after which no further resets happen"""
    shrink_factor: float = 0.5
    """weight of the old parameters in shrink-and-perturb"""
    perturb_factor: float = 0.5
    """weight of the random parameters in shrink-and-perturb"""
    target_tau: float = 0.005
    """EMA rate of the target network (updated every gradient step)"""
    batch_size: int = 32
    """the batch size of sample from the reply memory"""
    end_e: float = 0.0
    """the final epsilon for exploration"""
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
    jumps: int = 5
    """the number of transition-model prediction steps (K)"""
    spr_weight: float = 5.0
    """weight of the SPR latent-prediction loss"""
    prioritized_replay_alpha: float = 0.5
    """alpha parameter for prioritized replay"""
    prioritized_replay_beta: float = 0.5
    """fixed beta for prioritized replay weights"""
    prioritized_replay_eps: float = 1e-10
    """epsilon added to priorities"""
    n_atoms: int = 51
    """the number of atoms"""
    v_min: float = -10
    """the return lower bound"""
    v_max: float = 10
    """the return upper bound"""
    hidden_size: int = 2048
    """the width of the shared projection / hidden layer"""
    width_scale: int = 4
    """the Impala-CNN width multiplier"""
    data_augmentation: bool = True
    """apply DrQ shift/intensity augmentation to learner inputs"""
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


def drq_image_augmentation(images: torch.Tensor, pad: int = 4, intensity_scale: float = 0.05) -> torch.Tensor:
    """DrQ random-shift and intensity augmentation (Kostrikov et al., 2020)."""
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
    """Dopamine's epsilon schedule: 1 until warmup, then linear decay to epsilon."""
    steps_left = decay_period + warmup_steps - step
    bonus = (1.0 - epsilon) * steps_left / max(decay_period, 1)
    bonus = min(max(bonus, 0.0), 1.0 - epsilon)
    return epsilon + bonus


def exponential_decay_scheduler(decay_period, initial_value, final_value, reverse=False):
    """Official BBF logarithmic annealing schedule (warmup 0)."""
    if reverse:
        initial_value = 1.0 - initial_value
        final_value = 1.0 - final_value
    start = math.log(initial_value)
    end = math.log(final_value)

    def scheduler(step):
        bonus = min(max((decay_period - step) / decay_period, 0.0), 1.0)
        value = math.exp(bonus * (start - end) + end)
        return 1.0 - value if reverse else value

    return scheduler


def renormalize(latent: torch.Tensor) -> torch.Tensor:
    """Per-sample min-max normalization over all latent dimensions."""
    flat = latent.flatten(1)
    minimum = flat.min(dim=1, keepdim=True).values
    maximum = flat.max(dim=1, keepdim=True).values
    flat = (flat - minimum) / (maximum - minimum + 1e-8)
    return flat.view_as(latent)


class ImpalaResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        h = self.conv1(F.relu(x))
        h = self.conv2(F.relu(h))
        return x + h


class ImpalaCNN(nn.Module):
    """Impala encoder with the official BBF scaling (width 4, two blocks)."""

    def __init__(self, in_channels=4, width_scale=4, num_blocks=2, dims=(16, 32, 32)):
        super().__init__()
        stages = []
        channels = in_channels
        for dim in dims:
            out_channels = dim * width_scale
            layers = [
                nn.Conv2d(channels, out_channels, 3, padding=1),
                nn.MaxPool2d(3, stride=2, padding=1),
            ]
            layers.extend(ImpalaResidualBlock(out_channels) for _ in range(num_blocks))
            stages.append(nn.Sequential(*layers))
            channels = out_channels
        self.stages = nn.Sequential(*stages)
        self.out_channels = channels

    def forward(self, x):
        return F.relu(self.stages(x))


class TransitionModel(nn.Module):
    """SPR-style convolutional dynamics model."""

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


class BBFNetwork(nn.Module):
    def __init__(self, env, n_atoms, v_min, v_max, hidden_size=2048, width_scale=4):
        super().__init__()
        self.n_atoms = int(n_atoms)
        self.v_min = v_min
        self.v_max = v_max
        self.delta_z = (v_max - v_min) / (n_atoms - 1)
        self.n_actions = int(env.single_action_space.n)
        self.register_buffer("support", torch.linspace(v_min, v_max, n_atoms))

        self.encoder = ImpalaCNN(width_scale=width_scale)
        latent_channels = self.encoder.out_channels
        with torch.no_grad():
            spatial = self.encoder(torch.zeros(1, 4, 84, 84)).shape
        flat_dim = spatial[1] * spatial[2] * spatial[3]

        self.transition_model = TransitionModel(latent_channels, self.n_actions)
        # The projection is the shared single hidden layer of the dueling head.
        self.projection = nn.Linear(flat_dim, hidden_size)
        self.predictor = nn.Linear(hidden_size, hidden_size)
        self.value_out = nn.Linear(hidden_size, n_atoms)
        self.advantage_out = nn.Linear(hidden_size, n_atoms * self.n_actions)

    def encode(self, x):
        return renormalize(self.encoder(x / 255.0))

    def project(self, latent):
        return self.projection(latent.flatten(1))

    def q_dist(self, latent, log=False):
        h = F.relu(self.project(latent))
        value = self.value_out(h).view(-1, 1, self.n_atoms)
        advantage = self.advantage_out(h).view(-1, self.n_actions, self.n_atoms)
        q_atoms = value + advantage - advantage.mean(dim=1, keepdim=True)
        return F.log_softmax(q_atoms, dim=2) if log else F.softmax(q_atoms, dim=2)


class BBFReplayBuffer(PrioritizedAtariReplayBuffer):
    """PER replay serving K-step sequences and variable-horizon n-step returns.

    Unlike the SPR buffer, candidates only require existence of the maximum
    horizon window; returns are truncated at terminals when sampled, matching
    the official subsequence replay used by BBF's annealed update horizon.
    """

    def __init__(self, *pargs, jumps: int, max_n_step: int, **kwargs):
        super().__init__(*pargs, **kwargs)
        self.jumps = int(jumps)
        self.sample_horizon = max(int(max_n_step), self.jumps)

    def add(self, next_observations, actions, rewards, dones) -> None:
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

    def sample_bbf(self, batch_size: int, n_step: int, gamma: float):
        indices = self.sum_tree.sample(batch_size)
        rows = indices // self.n_envs
        env_indices = indices % self.n_envs
        samples = self._encode_samples(rows, env_indices, n_step, gamma)

        start_ids = self.transition_ids[rows]
        future_stacks = []
        for k in range(1, self.jumps + 1):
            future_rows = (rows + k) % self.time_capacity
            future_stacks.append(self._encode_stack(future_rows, env_indices, start_ids + k))
        future_observations = self._to_torch(np.stack(future_stacks, axis=1))

        action_rows = np.stack(
            [(rows + k) % self.time_capacity for k in range(self.jumps)], axis=1
        )
        action_sequences = self._to_torch(self.actions[action_rows, env_indices[:, None]])

        step_dones = self.dones[action_rows, env_indices[:, None]]
        nonterminal = np.concatenate(
            [np.ones((batch_size, 1), dtype=np.float32), 1.0 - np.sign(np.cumsum(step_dones, axis=1))],
            axis=1,
        )

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


def shrink_perturb_and_reset(q_network, target_network, optimizer, args, envs, device, in_place_state):
    """Official BBF reset: shrink-and-perturb the encoder and transition
    model, hard-reset projection/predictor/heads, and clear the optimizer
    state of every reset parameter (kept parameters keep their moments).

    With ``in_place_state`` the optimizer moments are zeroed in place instead
    of popped, which keeps captured CUDA graphs valid across resets.
    """
    fresh = BBFNetwork(
        envs, args.n_atoms, args.v_min, args.v_max, args.hidden_size, args.width_scale
    ).to(device)
    keep_keys = ("encoder.", "transition_model.")

    def clear_state(param):
        if in_place_state:
            for value in optimizer.state.get(param, {}).values():
                if isinstance(value, torch.Tensor):
                    value.zero_()
        else:
            optimizer.state.pop(param, None)

    with torch.no_grad():
        for (name, param), fresh_param in zip(q_network.named_parameters(), fresh.parameters()):
            if name.startswith(keep_keys):
                param.mul_(args.shrink_factor).add_(fresh_param, alpha=args.perturb_factor)
            else:
                param.copy_(fresh_param)
            clear_state(param)
        # BatchNorm running statistics of the transition model restart too.
        for (name, buf), fresh_buf in zip(q_network.named_buffers(), fresh.buffers()):
            if name.startswith(keep_keys):
                buf.copy_(args.shrink_factor * buf + args.perturb_factor * fresh_buf)
        for target_param, param in zip(target_network.parameters(), q_network.parameters()):
            target_param.copy_(param)
        for target_buf, buf in zip(target_network.buffers(), q_network.buffers()):
            target_buf.copy_(buf)


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
    if args.n_step > args.max_n_step:
        raise ValueError("n_step cannot exceed max_n_step")
    if args.cycle_steps < 1:
        raise ValueError("cycle_steps must be positive")
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

    q_network = BBFNetwork(
        envs, args.n_atoms, args.v_min, args.v_max, args.hidden_size, args.width_scale
    ).to(device)
    target_network = copy.deepcopy(q_network)
    for param in target_network.parameters():
        param.requires_grad_(False)
    optimizer = optim.AdamW(
        q_network.parameters(),
        lr=args.learning_rate,
        eps=args.adam_eps,
        weight_decay=args.weight_decay,
        capturable=args.cudagraphs,
    )

    horizon_fraction = exponential_decay_scheduler(args.cycle_steps, 1.0, args.n_step / args.max_n_step)
    gamma_scheduler = exponential_decay_scheduler(args.cycle_steps, args.min_gamma, args.gamma, reverse=True)

    rb = BBFReplayBuffer(
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
        max_n_step=args.max_n_step,
    )

    def policy(observations: torch.Tensor, epsilon: torch.Tensor) -> torch.Tensor:
        # official target_action_selection=True: act from the target network
        latent = target_network.encode(observations.float())
        q_values = torch.sum(target_network.q_dist(latent) * target_network.support, dim=2)
        greedy_actions = torch.argmax(q_values, dim=1)
        random_actions = torch.randint(n_actions, greedy_actions.shape, device=greedy_actions.device)
        explore = torch.rand(greedy_actions.shape, device=greedy_actions.device) < epsilon
        return torch.where(explore, random_actions, greedy_actions)

    def update(data: TensorDict, gamma_n: torch.Tensor):
        observations = data["observations"].float()
        next_observations = data["next_observations"].float()
        if args.data_augmentation:
            observations = drq_image_augmentation(observations)
            next_observations = drq_image_augmentation(next_observations)
        actions = data["actions"]
        rewards = data["rewards"]
        dones = data["dones"]
        future_observations = data["future_observations"]
        action_sequences = data["action_sequences"]
        nonterminal = data["nonterminal"]
        weights = data["weights"].squeeze(-1)
        batch_indices = torch.arange(observations.shape[0], device=observations.device)

        # --- C51 n-step RL loss (target-network action selection) ---
        with torch.no_grad():
            next_latent = target_network.encode(next_observations)
            next_dist = target_network.q_dist(next_latent)
            support = target_network.support
            next_q = torch.sum(next_dist * support, dim=2)
            best_actions = torch.argmax(next_q, dim=1)
            next_pmfs = next_dist[batch_indices, best_actions]

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
                    target_obs = drq_image_augmentation(target_obs)
                target_projection = target_network.project(target_network.encode(target_obs))
            f_online = F.normalize(projection, p=2.0, dim=-1, eps=1e-3)
            f_target = F.normalize(target_projection, p=2.0, dim=-1, eps=1e-3)
            spr_losses.append(F.mse_loss(f_online, f_target, reduction="none").sum(-1))
        spr_loss_per_sample = (torch.stack(spr_losses, dim=1) * nonterminal[:, 1:]).mean(dim=1)

        loss = (rl_loss_per_sample * weights).mean() + args.spr_weight * (
            spr_loss_per_sample * weights
        ).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # EMA target update every gradient step (tau 0.005)
        with torch.no_grad():
            for target_param, param in zip(target_network.parameters(), q_network.parameters()):
                target_param.lerp_(param, args.target_tau)
            for target_buf, buf in zip(target_network.buffers(), q_network.buffers()):
                target_buf.copy_(buf)
        q_value = (log_pred.exp() * q_network.support).sum(dim=1).mean()
        return (
            rl_loss_per_sample.detach().mean(),
            spr_loss_per_sample.detach().mean(),
            rl_loss_per_sample.detach(),
            q_value.detach(),
        )

    if args.compile:
        # CuLE observation storage is updated in place, so do not use
        # reduce-overhead's implicit CUDA graphs here.
        policy = torch.compile(policy, mode=None, fullgraph=True)
        update = torch.compile(update, mode=None)

    if args.cudagraphs:
        # CudaGraphModule copies inputs into static buffers and clones outputs.
        # The EMA target update and shrink-and-perturb resets mutate module
        # and optimizer tensors in place, so graph replays observe them.
        policy = CudaGraphModule(policy, warmup=20)
        update = CudaGraphModule(update, warmup=20)

    epsilon_tensor = torch.zeros((), device=device)
    gamma_n_tensor = torch.zeros((), device=device)
    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    reset_result = envs.reset(seed=args.seed) if args.env_backend != "envpool" else envs.reset()
    obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result
    rb.initialize(obs)
    global_step = 0
    update_budget = 0.0
    learner_updates = 0
    cycle_grad_steps = 0
    next_reset = args.reset_every
    episode_stats = EpisodeStats(args.solve_window, args.solve_reward)
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

        # ALGO LOGIC: epsilon-greedy over the TARGET network
        # (official target_action_selection=True)
        epsilon_tensor.fill_(
            linearly_decaying_epsilon(
                args.epsilon_decay_period, global_step, args.learning_starts, args.end_e
            )
        )
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
                current_n = max(1, int(args.max_n_step * horizon_fraction(cycle_grad_steps)))
                current_gamma = gamma_scheduler(cycle_grad_steps)
                gamma_n_tensor.fill_(current_gamma**current_n)
                data, future_observations, action_sequences, nonterminal, indices, weights = (
                    rb.sample_bbf(args.batch_size, current_n, current_gamma)
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
                last_rl_loss, last_spr_loss, rl_loss_per_sample, last_q_value = update(
                    batch, gamma_n_tensor
                )
                rb.update_priorities(indices, rl_loss_per_sample.cpu().numpy())

                learner_updates += 1
                cycle_grad_steps += 1
                if learner_updates >= next_reset and learner_updates <= args.no_resets_after:
                    shrink_perturb_and_reset(
                        q_network,
                        target_network,
                        optimizer,
                        args,
                        envs,
                        device,
                        in_place_state=args.cudagraphs,
                    )
                    cycle_grad_steps = 0
                    next_reset += args.reset_every

            if not args.benchmark and global_step >= next_log_step and num_updates:
                sps = int(global_step / (time.time() - start_time))
                writer.add_scalar("losses/rl_loss", last_rl_loss.item(), global_step)
                writer.add_scalar("losses/spr_loss", last_spr_loss.item(), global_step)
                writer.add_scalar("losses/q_values", last_q_value.item(), global_step)
                writer.add_scalar("charts/SPS", sps, global_step)
                writer.add_scalar("charts/epsilon", epsilon_tensor.item(), global_step)
                writer.add_scalar("charts/n_step", current_n, global_step)
                writer.add_scalar("charts/gamma", current_gamma, global_step)
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
            "algorithm": "bbf",
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
