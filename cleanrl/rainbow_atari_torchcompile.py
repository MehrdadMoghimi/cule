# torch.compile twin of rainbow_atari.py, which is adapted from CleanRL's
# cleanrl/rainbow_atari.py (https://github.com/vwxyzjn/cleanrl, MIT).  The
# compile / CUDA-graph structure follows LeanRL
# (https://github.com/meta-pytorch/LeanRL, MIT).  Both licenses are
# reproduced in cleanrl/LICENSE.md.
# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/rainbow/#rainbow_ataripy
"""Rainbow Atari with optional torch.compile, CUDA graphs, and GPU TorchRL replay.

Environment interaction, n-step transition construction, sampling, and priority
updates remain eager (the priority segment trees use data-dependent shapes that
CUDA graphs cannot capture).  The learner receives fixed-shape GPU TensorDict
batches, so the policy and learner update can be compiled and captured.
"""

import csv
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
from tqdm import tqdm

torch.set_float32_matmul_precision("high")

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from cule_env import (
    done_tensor,
    frame_stack_observation,
    grayscale_observation,
    make_cule_env,
    resolve_cule_device,
    step_env,
    to_tensor,
)

from cleanrl_utils.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)
from cleanrl_utils.atari_eval import evaluate_cule_policy
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
    """the wandb project name"""
    wandb_entity: str | None = None
    """the WandB entity"""
    capture_video: bool = False
    """whether to capture videos of the agent performances"""
    env_backend: str = "cule"
    """environment backend: cule, envpool, or gymnasium"""
    cule_device: str = "auto"
    """CuLE device; auto uses CUDA for 32+ environments and CPU for smaller batches"""

    env_id: str = "BreakoutNoFrameskip-v4"
    """the id of the environment"""
    total_timesteps: int = 10_000_000
    """total timesteps of the experiments"""
    learning_rate: float = 0.0000625
    """the learning rate of the optimizer"""
    num_envs: int = 256
    """the number of parallel game environments"""
    buffer_size: int = 100_000
    """the replay memory size in individual transitions"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 1.0
    """the target-network update rate"""
    target_network_frequency: int = 2000
    """learner updates between target-network updates"""
    batch_size: int = 512
    """the replay sample batch size"""
    learning_starts: int = 80_000
    """the number of collected transitions before learning starts"""
    learner_updates_per_vector_step: float = 1.0
    """gradient updates accrued per vector environment step; may be fractional"""
    replay_ratio: float | None = 1.0
    """sampled replay items per collected transition; overrides learner-updates-per-vector-step"""
    max_training_seconds: float = 0.0
    """wall-clock training limit; zero disables it"""
    solve_reward: float | None = None
    """stop when the moving episodic return reaches this value"""
    solve_window: int = 20
    """number of completed episodes in the solve moving average"""
    n_step: int = 3
    """the number of steps to look ahead for n-step Q learning"""
    prioritized_replay_alpha: float = 0.5
    """alpha parameter for prioritized replay"""
    prioritized_replay_beta: float = 0.4
    """initial beta parameter for prioritized replay"""
    prioritized_replay_eps: float = 1e-6
    """epsilon added to priorities"""
    n_atoms: int = 51
    """the number of atoms"""
    v_min: float = -10
    """the return lower bound"""
    v_max: float = 10
    """the return upper bound"""

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

    evaluation_interval: int = 1_000_000
    """policy transitions between deterministic full-game evaluations"""
    evaluation_episodes: int = 10
    """complete unclipped games per evaluation"""
    evaluation_seed: int = 10_000
    """first seed in the fixed evaluation seed set"""
    evaluation_max_episode_steps: int = 18_000
    """maximum frame-skipped steps per evaluation game"""
    skip_initial_evaluation: bool = False
    """skip the untrained-policy evaluation when writing a learning curve"""
    learning_curve_path: str | None = None
    """optional CSV path for full-game evaluation results"""
    emit_progress: bool = False
    """emit machine-readable transition progress for an outer launcher"""


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
        self.value_head = nn.Sequential(NoisyLinear(3136, 512), nn.ReLU(), NoisyLinear(512, n_atoms))
        self.advantage_head = nn.Sequential(
            NoisyLinear(3136, 512), nn.ReLU(), NoisyLinear(512, n_atoms * self.n_actions)
        )

    def forward(self, x):
        hidden = self.network(x / 255.0)
        value = self.value_head(hidden).view(-1, 1, self.n_atoms)
        advantage = self.advantage_head(hidden).view(-1, self.n_actions, self.n_atoms)
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
    if args.n_atoms < 2:
        raise ValueError("n_atoms must be at least two")
    if args.n_step < 1:
        raise ValueError("n_step must be positive")
    if args.target_network_frequency < 1:
        raise ValueError("target_network_frequency must be positive")
    if args.learner_updates_per_vector_step < 0:
        raise ValueError("learner_updates_per_vector_step must be non-negative")
    if args.max_training_seconds < 0:
        raise ValueError("max_training_seconds must be non-negative")
    if args.benchmark_warmup_iterations < 0:
        raise ValueError("benchmark_warmup_iterations cannot be negative")
    if args.benchmark_measure_iterations < 1:
        raise ValueError("benchmark_measure_iterations must be positive")
    if args.learning_curve_path and args.evaluation_interval < 1:
        raise ValueError("evaluation_interval must be positive when learning_curve_path is set")
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

    q_network = NoisyDuelingDistributionalNetwork(envs, args.n_atoms, args.v_min, args.v_max).to(device)
    target_network = NoisyDuelingDistributionalNetwork(envs, args.n_atoms, args.v_min, args.v_max).to(device)
    target_network.load_state_dict(q_network.state_dict())
    optimizer = optim.Adam(
        q_network.parameters(),
        lr=args.learning_rate,
        eps=1.5e-4,
        capturable=args.cudagraphs and not args.compile,
    )
    priority_sampler = GpuPrioritizedSampler(
        args.buffer_size,
        alpha=args.prioritized_replay_alpha,
        beta=args.prioritized_replay_beta,
        eps=args.prioritized_replay_eps,
        device=device,
    )
    # Keep full stacked transitions on the learner device, as in
    # dqn_torchcompile.py.  The sampler above supplies Rainbow PER while
    # TorchRL owns storage, cyclic writes, and TensorDict batch retrieval.
    rb = ReplayBuffer(storage=LazyTensorStorage(args.buffer_size, device=device), sampler=priority_sampler)
    n_step_accumulator = NStepTransitionAccumulator(args.n_step, args.gamma)

    curve_file = None
    curve_writer = None
    if args.learning_curve_path:
        curve_path = os.path.abspath(args.learning_curve_path)
        os.makedirs(os.path.dirname(curve_path), exist_ok=True)
        curve_file = open(curve_path, "w", encoding="utf-8", newline="")
        curve_writer = csv.DictWriter(
            curve_file,
            fieldnames=[
                "algorithm",
                "seed",
                "frames",
                "training_seconds",
                "worker_wall_seconds",
                "reward_mean",
                "reward_median",
                "reward_min",
                "reward_max",
                "reward_std",
                "length_mean",
                "length_median",
                "length_min",
                "length_max",
                "length_std",
            ],
        )
        curve_writer.writeheader()

    last_evaluation_step = [-1]

    def evaluate_and_log(frames: int, training_seconds: float) -> float:
        if curve_writer is None:
            return 0.0
        if args.emit_progress:
            print(f"EVALUATION_START {frames}", flush=True)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        evaluation_started = time.perf_counter()
        was_training = q_network.training
        q_network.eval()

        def greedy_actions(states: torch.Tensor) -> torch.Tensor:
            distribution = q_network(states)
            q_values = torch.sum(distribution * q_network.support, dim=2)
            return q_values.argmax(dim=1)

        stats = evaluate_cule_policy(
            args.env_id,
            greedy_actions,
            device,
            num_episodes=args.evaluation_episodes,
            seed=args.evaluation_seed,
            max_episode_steps=args.evaluation_max_episode_steps,
        )
        q_network.train(was_training)
        evaluation_seconds = time.perf_counter() - evaluation_started
        row = {
            "algorithm": "rainbow",
            "seed": args.seed,
            "frames": frames,
            "training_seconds": training_seconds,
            "worker_wall_seconds": time.perf_counter() - process_start,
            **stats,
        }
        curve_writer.writerow(row)
        curve_file.flush()
        last_evaluation_step[0] = frames
        print(f"EVALUATION_RESULT {json.dumps(row, sort_keys=True)}", flush=True)
        return evaluation_seconds

    def policy(observations: torch.Tensor) -> torch.Tensor:
        q_dist = q_network(observations)
        q_values = torch.sum(q_dist * q_network.support, dim=2)
        return torch.argmax(q_values, dim=1)

    def update(data: TensorDict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        observations = data["observations"]
        actions = data["actions"]
        next_observations = data["next_observations"]
        weights = data["priority_weight"]
        dones = data["dones"].float().unsqueeze(-1)
        rewards = data["rewards"].unsqueeze(-1)
        with torch.no_grad():
            next_dist = target_network(next_observations)
            support = target_network.support
            next_dist_online = q_network(next_observations)
            next_q_online = torch.sum(next_dist_online * support, dim=2)
            best_actions = torch.argmax(next_q_online, dim=1)
            batch_indices = torch.arange(next_observations.shape[0], device=next_observations.device)
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
    reset_result = envs.reset(seed=args.seed) if args.env_backend in {"cule", "gymnasium"} else envs.reset()
    reset_obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result
    obs = to_tensor(reset_obs, device)
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
    evaluation_seconds_total = 0.0
    if curve_writer is not None and not args.benchmark and not args.skip_initial_evaluation:
        evaluate_and_log(0, 0.0)
    learning_wall_start = time.perf_counter()
    next_evaluation_step = args.evaluation_interval
    progress_interval = max(args.num_envs, args.total_timesteps // 100)
    next_progress_step = progress_interval

    vector_steps = tqdm(
        range(num_vector_steps),
        desc=f"Rainbow {args.env_backend}",
        unit="step",
        disable=args.benchmark,
    )
    for vector_step in vector_steps:
        if args.max_training_seconds and time.perf_counter() - start_time >= args.max_training_seconds:
            break
        if args.benchmark and vector_step == args.benchmark_warmup_iterations:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            benchmark_start = time.perf_counter()
            benchmark_start_step = global_step
            benchmark_start_updates = learner_updates

        priority_sampler.beta = min(
            1.0,
            args.prioritized_replay_beta
            + global_step * (1.0 - args.prioritized_replay_beta) / args.total_timesteps,
        )
        q_network.reset_noise()
        if args.compile:
            torch.compiler.cudagraph_mark_step_begin()
        with torch.no_grad():
            actions = policy(obs)

        # CuLE mutates and reuses its observation tensor after each step.
        # Replay receives detached lifetime-safe clones before the next step.
        transition_obs = obs.clone()
        step_result = step_env(envs, actions)
        if len(step_result) == 5:
            next_obs_raw, rewards, terminations, truncations, infos = step_result
        else:
            next_obs_raw, rewards, terminations, infos = step_result
            truncations = np.zeros_like(terminations, dtype=bool)
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
        if not args.benchmark:
            vector_steps.set_postfix(frames=global_step, updates=learner_updates, refresh=False)
        if args.emit_progress and not args.benchmark and global_step >= next_progress_step:
            print(f"TRAINING_PROGRESS {global_step}", flush=True)
            while next_progress_step <= global_step:
                next_progress_step += progress_interval

        solved = False
        if not args.benchmark:
            solved = episode_stats.update(infos, global_step, writer)

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
                # Priority-tree writes remain outside the compiled learner.
                # Clone before the next compiled call may reuse output storage.
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
                writer.add_scalar("charts/beta", priority_sampler.beta, global_step)
                print("SPS:", sps)
                next_log_step = global_step + max(10_000, args.num_envs)
        if curve_writer is not None and not args.benchmark and global_step >= next_evaluation_step:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            training_seconds = time.perf_counter() - learning_wall_start - evaluation_seconds_total
            evaluation_seconds_total += evaluate_and_log(global_step, training_seconds)
            while next_evaluation_step <= global_step:
                next_evaluation_step += args.evaluation_interval
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
            "algorithm": "rainbow",
            "backend": args.env_backend,
            "batch_size": args.batch_size,
            "benchmark": "full_training_loop",
            "compile": args.compile,
            "cudagraphs": args.cudagraphs,
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
            "replay_backend": "torchrl_lazy_tensor_storage+gpu_per",
            "replay_storage": "gpu_full_stacked_n_step_transitions",
            "replay_ratio": measured_updates * args.batch_size / max(measured_steps, 1),
            "schema_version": 1,
            "sps": measured_steps / measured_seconds,
            "ups": measured_updates / measured_seconds,
            "warmup_iterations": args.benchmark_warmup_iterations,
        }
        print(f"BENCHMARK_RESULT {json.dumps(result, sort_keys=True)}", flush=True)
    else:
        if curve_writer is not None and last_evaluation_step[0] != global_step:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            training_seconds = time.perf_counter() - learning_wall_start - evaluation_seconds_total
            evaluate_and_log(global_step, training_seconds)
        elapsed = time.perf_counter() - start_time
        print("SPS:", int(global_step / max(elapsed, 1e-9)))
        print("learner updates:", learner_updates)
        print("UPS:", learner_updates / max(elapsed, 1e-9))
        print("effective UTD:", learner_updates / max(global_step - args.learning_starts, 1))
        print("replay ratio:", learner_updates * args.batch_size / max(global_step - args.learning_starts, 1))
        episode_stats.print_summary()

    envs.close()
    if curve_file is not None:
        curve_file.close()
    if writer is not None:
        writer.close()
