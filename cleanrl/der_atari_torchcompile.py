# DER: Data-Efficient Rainbow (van Hasselt et al., 2019,
# https://arxiv.org/abs/1906.05243) with optional torch.compile, CUDA graphs,
# and GPU-resident TorchRL replay.
# The trainer skeleton derives from CleanRL (https://github.com/vwxyzjn/cleanrl,
# MIT) and the compile / CUDA-graph structure from LeanRL
# (https://github.com/meta-pytorch/LeanRL, MIT); both licenses are reproduced
# in cleanrl/LICENSE.md.
"""DER Atari-100K with optional torch.compile and GPU TorchRL replay.

Ported from the official Dopamine implementation
(https://github.com/google/dopamine, dopamine/labs/atari_100k,
configs/DER.gin).  Defaults reproduce the official single-environment
configuration; see der_atari.py for the eager twin.  Environment stepping,
n-step accumulation, PER sampling, and priority-tree writes stay outside the
compiled regions; the fixed-shape policy and learner update can be compiled
and captured.  Optional DrQ augmentation runs inside the compiled update.
Supports the cule, envpool, and gymnasium backends.
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
from tensordict import TensorDict
from tensordict.nn import CudaGraphModule
from torch.utils.tensorboard import SummaryWriter
from torchrl.data import LazyTensorStorage, ReplayBuffer

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
from cleanrl_utils.episode_stats import EpisodeStats
from cleanrl_utils.torchrl_replay import GpuPrioritizedSampler, NStepTransitionAccumulator


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, torch.backends.cudnn.deterministic=False"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str | None = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances"""
    env_backend: str = "cule"
    """environment backend: cule, envpool, or gymnasium"""
    cule_device: str = "auto"
    """CuLE device; auto uses CUDA for 32+ environments and CPU for smaller batches"""
    save_model: bool = False
    """whether to save the model into the runs/{run_name} folder"""

    env_id: str = "BreakoutNoFrameskip-v4"
    """the id of the environment"""
    total_timesteps: int = 100_000
    """total timesteps of the experiments (Atari-100K benchmark budget)"""
    learning_rate: float = 1e-4
    """the learning rate of the optimizer"""
    adam_eps: float = 1.5e-4
    """the epsilon of the Adam optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    buffer_size: int = 100_000
    """the replay memory size in individual transitions (covers the full 100K budget)"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 1.0
    """the target-network update rate"""
    target_network_frequency: int = 2000
    """learner updates between target-network updates"""
    batch_size: int = 32
    """the replay sample batch size"""
    end_e: float = 0.01
    """the final epsilon for exploration"""
    epsilon_decay_period: int = 2000
    """steps after learning starts over which epsilon decays from 1 to end-e"""
    learning_starts: int = 1600
    """the number of collected transitions before learning starts"""
    learner_updates_per_vector_step: float = 1.0
    """gradient updates accrued per vector environment step; may be fractional"""
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
    prioritized_replay_alpha: float = 0.5
    """alpha parameter for prioritized replay; 0.5 matches Dopamine's sqrt(loss) priorities"""
    prioritized_replay_beta: float = 0.5
    """fixed beta for prioritized replay weights (Dopamine does not anneal it)"""
    prioritized_replay_eps: float = 1e-10
    """epsilon added to priorities"""
    n_atoms: int = 51
    """the number of atoms"""
    v_min: float = -10
    """the return lower bound"""
    v_max: float = 10
    """the return upper bound"""
    data_augmentation: bool = False
    """apply DrQ random-shift/intensity augmentation to replay samples"""

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


class NoisyDuelingDistributionalNetwork(nn.Module):
    def __init__(self, env, n_atoms, v_min, v_max):
        super().__init__()
        self.n_atoms = int(n_atoms)
        self.v_min = v_min
        self.v_max = v_max
        self.delta_z = (v_max - v_min) / (n_atoms - 1)
        self.n_actions = int(env.single_action_space.n)
        self.register_buffer("support", torch.linspace(v_min, v_max, n_atoms))

        self.network = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        conv_output_size = 3136
        self.value_head = nn.Sequential(NoisyLinear(conv_output_size, 512), nn.ReLU(), NoisyLinear(512, n_atoms))
        self.advantage_head = nn.Sequential(
            NoisyLinear(conv_output_size, 512), nn.ReLU(), NoisyLinear(512, n_atoms * self.n_actions)
        )

    def forward(self, x):
        h = self.network(x / 255.0)
        value = self.value_head(h).view(-1, 1, self.n_atoms)
        advantage = self.advantage_head(h).view(-1, self.n_actions, self.n_atoms)
        q_atoms = value + advantage - advantage.mean(dim=1, keepdim=True)
        return F.softmax(q_atoms, dim=2)

    def reset_noise(self):
        for layer in self.value_head:
            if isinstance(layer, NoisyLinear):
                layer.reset_noise()
        for layer in self.advantage_head:
            if isinstance(layer, NoisyLinear):
                layer.reset_noise()


if __name__ == "__main__":
    process_start = time.perf_counter()
    args = tyro.cli(Args)
    if args.num_envs < 1:
        raise ValueError("num_envs must be positive")
    if args.buffer_size < args.num_envs:
        raise ValueError("buffer_size must be at least num_envs")
    if args.batch_size < 1:
        raise ValueError("batch_size must be positive")
    if args.target_network_frequency < 1:
        raise ValueError("target_network_frequency must be positive")
    if args.epsilon_decay_period < 1:
        raise ValueError("epsilon_decay_period must be positive")
    if args.learner_updates_per_vector_step < 0:
        raise ValueError("learner_updates_per_vector_step must be non-negative")
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

    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{args.compile}__{args.cudagraphs}"
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
            "|param|value|\n|-|-|\n%s" % ("\n".join(f"|{key}|{value}|" for key, value in vars(args).items())),
        )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    if args.cudagraphs and device.type != "cuda":
        raise ValueError("cudagraphs requires CUDA")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

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
    if not isinstance(envs.single_action_space, gym.spaces.Discrete):
        raise ValueError("only discrete action spaces are supported")
    n_actions = int(envs.single_action_space.n)

    q_network = NoisyDuelingDistributionalNetwork(envs, args.n_atoms, args.v_min, args.v_max).to(device)
    target_network = NoisyDuelingDistributionalNetwork(envs, args.n_atoms, args.v_min, args.v_max).to(device)
    target_network.load_state_dict(q_network.state_dict())
    optimizer = optim.Adam(
        q_network.parameters(),
        lr=args.learning_rate,
        eps=args.adam_eps,
        capturable=args.cudagraphs and not args.compile,
    )
    priority_sampler = GpuPrioritizedSampler(
        args.buffer_size,
        alpha=args.prioritized_replay_alpha,
        beta=args.prioritized_replay_beta,
        eps=args.prioritized_replay_eps,
        device=device,
    )
    rb = ReplayBuffer(storage=LazyTensorStorage(args.buffer_size, device=device), sampler=priority_sampler)
    n_step_accumulator = NStepTransitionAccumulator(args.n_step, args.gamma)

    def policy(observations: torch.Tensor, epsilon: torch.Tensor) -> torch.Tensor:
        q_dist = q_network(observations)
        q_values = torch.sum(q_dist * q_network.support, dim=2)
        greedy_actions = torch.argmax(q_values, dim=1)
        random_actions = torch.randint(n_actions, greedy_actions.shape, device=greedy_actions.device)
        explore = torch.rand(greedy_actions.shape, device=greedy_actions.device) < epsilon
        return torch.where(explore, random_actions, greedy_actions)

    def update(data: TensorDict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        observations = data["observations"].float()
        next_observations = data["next_observations"].float()
        if args.data_augmentation:
            observations = drq_image_augmentation(observations)
            next_observations = drq_image_augmentation(next_observations)
        actions = data["actions"]
        weights = data["priority_weight"]
        dones = data["dones"].float().unsqueeze(-1)
        rewards = data["rewards"].unsqueeze(-1)
        batch_indices = torch.arange(observations.shape[0], device=observations.device)
        with torch.no_grad():
            next_dist = target_network(next_observations)
            support = target_network.support
            # double q-learning
            next_dist_online = q_network(next_observations)
            next_q_online = torch.sum(next_dist_online * support, dim=2)
            best_actions = torch.argmax(next_q_online, dim=1)
            next_pmfs = next_dist[batch_indices, best_actions]

            gamma_n = args.gamma**args.n_step
            next_atoms = rewards + gamma_n * support * (1 - dones)
            tz = next_atoms.clamp(q_network.v_min, q_network.v_max)
            b = (tz - q_network.v_min) / q_network.delta_z
            lower = b.floor().clamp(0, args.n_atoms - 1)
            upper = b.ceil().clamp(0, args.n_atoms - 1)
            lower_mass = (upper.float() + (lower == b).float() - b) * next_pmfs
            upper_mass = (b - lower) * next_pmfs
            target_pmfs = torch.zeros_like(next_pmfs)
            target_pmfs.scatter_add_(1, lower.long(), lower_mass)
            target_pmfs.scatter_add_(1, upper.long(), upper_mass)

        dist = q_network(observations)
        action_indices = actions.reshape(-1, 1, 1).expand(-1, 1, args.n_atoms)
        pred_dist = dist.gather(1, action_indices).squeeze(1)
        loss_per_sample = -(target_pmfs * torch.log(pred_dist.clamp(min=1e-5, max=1 - 1e-5))).sum(dim=1)
        loss = (loss_per_sample * weights).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        q_value = (pred_dist * q_network.support).sum(dim=1).mean()
        return loss.detach(), loss_per_sample.detach(), q_value.detach()

    if args.compile:
        # CuLE observation storage is updated in place, so do not use
        # reduce-overhead's implicit CUDA graphs here.
        policy = torch.compile(policy, mode=None, fullgraph=True)
        update = torch.compile(update, mode=None)

    if args.cudagraphs:
        # CudaGraphModule copies inputs into static buffers and clones outputs.
        # Noisy-net resets and the target-network sync mutate module tensors in
        # place, so graph replays observe them; PER sampling and priority-tree
        # writes stay outside the capture.
        policy = CudaGraphModule(policy, warmup=20)
        update = CudaGraphModule(update, warmup=20)

    start_time = time.perf_counter()
    reset_result = envs.reset(seed=args.seed) if args.env_backend != "envpool" else envs.reset()
    reset_obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result
    obs = to_tensor(reset_obs, device)
    epsilon_tensor = torch.zeros((), device=device)
    global_step = 0
    learner_updates = 0
    update_budget = 0.0
    next_target_update = args.target_network_frequency
    next_log_step = max(10_000, args.num_envs)
    episode_stats = EpisodeStats(args.solve_window, args.solve_reward)
    last_loss = None
    last_q_value = None
    num_vector_steps = math.ceil(args.total_timesteps / args.num_envs)
    if args.benchmark:
        num_vector_steps = args.benchmark_warmup_iterations + args.benchmark_measure_iterations
    benchmark_start = None
    benchmark_start_step = None
    benchmark_start_updates = None

    for vector_step in range(num_vector_steps):
        if args.max_training_seconds and time.perf_counter() - start_time >= args.max_training_seconds:
            break
        if args.benchmark and vector_step == args.benchmark_warmup_iterations:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            benchmark_start = time.perf_counter()
            benchmark_start_step = global_step
            benchmark_start_updates = learner_updates

        epsilon_tensor.fill_(
            linearly_decaying_epsilon(
                args.epsilon_decay_period, global_step, args.learning_starts, args.end_e
            )
        )
        q_network.reset_noise()
        if args.compile:
            torch.compiler.cudagraph_mark_step_begin()
        with torch.no_grad():
            actions = policy(obs, epsilon_tensor)

        # CuLE mutates and reuses its observation tensor after each step.
        transition_obs = obs.clone()
        step_result = step_env(envs, actions)
        if len(step_result) == 5:
            next_obs_raw, rewards, terminations, truncations, infos = step_result
        else:
            next_obs_raw, rewards, terminations, infos = step_result
            truncations = np.zeros_like(np.asarray(terminations), dtype=bool)
        next_obs = to_tensor(next_obs_raw, device)
        rewards = to_tensor(rewards, device, torch.float32).view(-1)
        transition_dones = done_tensor(terminations, truncations, device).bool()
        transition_next_obs = next_obs.clone()
        if args.env_backend == "gymnasium" and np.asarray(truncations).any():
            for idx, truncated in enumerate(truncations):
                if truncated:
                    transition_next_obs[idx] = to_tensor(infos["final_observation"][idx], device)
        n_step_transition = n_step_accumulator.append(
            TensorDict(
                {
                    "observations": transition_obs,
                    "next_observations": transition_next_obs,
                    "actions": actions,
                    "rewards": rewards,
                    "dones": transition_dones,
                },
                batch_size=[args.num_envs],
                device=device,
            )
        )
        if n_step_transition is not None:
            n_step_data, valid = n_step_transition
            write_indices = rb.extend(n_step_data)
            priority_sampler.set_initial_priorities(write_indices, valid)
        global_step += args.num_envs
        obs = next_obs

        solved = False
        if not args.benchmark:
            solved = episode_stats.update(
                completed_episode_infos(infos, transition_dones), global_step, writer
            )

        if (
            global_step > args.learning_starts
            and global_step >= args.n_step * args.num_envs
            and len(rb) >= args.batch_size
        ):
            update_budget += args.learner_updates_per_vector_step
            num_updates = int(update_budget)
            update_budget -= num_updates
            for _ in range(num_updates):
                q_network.reset_noise()
                target_network.reset_noise()
                data, sample_info = rb.sample(args.batch_size, return_info=True)
                data["priority_weight"] = sample_info["priority_weight"]
                if args.compile:
                    torch.compiler.cudagraph_mark_step_begin()
                last_loss, loss_per_sample, last_q_value = update(data)
                # Priority-tree writes remain outside the compiled learner;
                # with alpha 0.5 the stored priority is sqrt(loss + eps),
                # matching Dopamine's Atari-100K agents.
                rb.update_priority(sample_info["index"], loss_per_sample.detach().clone())
                learner_updates += 1

            if learner_updates >= next_target_update:
                with torch.no_grad():
                    for target_param, param in zip(target_network.parameters(), q_network.parameters()):
                        target_param.copy_(args.tau * param + (1.0 - args.tau) * target_param)
                next_target_update = (
                    learner_updates // args.target_network_frequency + 1
                ) * args.target_network_frequency

            if not args.benchmark and global_step >= next_log_step and num_updates:
                elapsed = time.perf_counter() - start_time
                sps = int(global_step / max(elapsed, 1e-9))
                writer.add_scalar("losses/td_loss", last_loss.item(), global_step)
                writer.add_scalar("losses/q_values", last_q_value.item(), global_step)
                writer.add_scalar("charts/SPS", sps, global_step)
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
                next_log_step = global_step + max(10_000, args.num_envs)
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
            "algorithm": "der",
            "backend": args.env_backend,
            "batch_size": args.batch_size,
            "benchmark": "full_training_loop",
            "compile": args.compile,
            "cudagraphs": args.cudagraphs,
            "data_augmentation": args.data_augmentation,
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
            "replay_backend": "torchrl_lazy_tensor_storage_per",
            "replay_storage": "gpu_full_stacked_transitions",
            "replay_ratio": measured_updates * args.batch_size / max(measured_steps, 1),
            "schema_version": 1,
            "sps": measured_steps / measured_seconds,
            "ups": measured_updates / measured_seconds,
            "warmup_iterations": args.benchmark_warmup_iterations,
        }
        print(f"BENCHMARK_RESULT {json.dumps(result, sort_keys=True)}", flush=True)
    else:
        elapsed = time.perf_counter() - start_time
        print("SPS:", int(global_step / max(elapsed, 1e-9)))
        print("learner updates:", learner_updates)
        print("UPS:", learner_updates / max(elapsed, 1e-9))
        print("effective UTD:", learner_updates / max(global_step - args.learning_starts, 1))
        print("replay ratio:", learner_updates * args.batch_size / max(global_step - args.learning_starts, 1))
        episode_stats.print_summary()

    if args.save_model and not args.benchmark:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        model_data = {"model_weights": q_network.state_dict(), "args": vars(args)}
        torch.save(model_data, model_path)
        print(f"model saved to {model_path}")

    envs.close()
    if writer is not None:
        writer.close()
