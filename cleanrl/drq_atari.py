# DrQ(eps): Data-regularized Q for Atari (Kostrikov et al., 2020,
# https://arxiv.org/abs/2004.13649), the epsilon-greedy variant evaluated in
# Agarwal et al., 2021 (https://arxiv.org/abs/2108.13264).
# Ported from the official Dopamine implementation
# (https://github.com/google/dopamine, dopamine/labs/atari_100k,
# configs/DrQ_eps.gin): Efficient DQN — dueling,
# double DQN, non-distributional, no noisy nets, uniform replay — with n-step
# 10, one batch-32 update per environment step, target sync every update,
# Adam 1e-4 with eps 1.5e-4, first update after 1600 transitions, epsilon
# 1 -> 0.01 over the following 5000 steps, and DrQ random-shift/intensity
# augmentation of every replay sample.
# Structure follows rainbow_atari.py, which is adapted from CleanRL
# (https://github.com/vwxyzjn/cleanrl, MIT; license in cleanrl/LICENSE.md).
# Supports gymnasium, cule, and envpool.
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
import torch.nn.functional as F
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
from cleanrl_utils.buffers import PrioritizedAtariReplayBuffer
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
    """learner updates between target-network updates (DrQ syncs every update)"""
    batch_size: int = 32
    """the batch size of sample from the reply memory"""
    end_e: float = 0.01
    """the final epsilon for exploration"""
    epsilon_decay_period: int = 5000
    """steps after learning starts over which epsilon decays from 1 to end-e"""
    learning_starts: int = 1600
    """timestep to start learning"""
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
    data_augmentation: bool = True
    """apply DrQ random-shift/intensity augmentation to replay samples"""
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
    """DrQ random-shift and intensity augmentation (Kostrikov et al., 2020).

    Matches Dopamine's Atari-100K agents: replication padding, a per-image
    random crop back to the original size, then multiplicative intensity noise
    `1 + scale * clip(N(0, 1), -2, 2)`.
    """
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


# ALGO LOGIC: initialize agent here:
class DuelingQNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.n_actions = int(env.single_action_space.n)
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
        self.value_head = nn.Sequential(nn.Linear(conv_output_size, 512), nn.ReLU(), nn.Linear(512, 1))
        self.advantage_head = nn.Sequential(
            nn.Linear(conv_output_size, 512), nn.ReLU(), nn.Linear(512, self.n_actions)
        )

    def forward(self, x):
        h = self.network(x / 255.0)
        value = self.value_head(h)
        advantage = self.advantage_head(h)
        return value + advantage - advantage.mean(dim=1, keepdim=True)


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
    if args.epsilon_decay_period < 1:
        raise ValueError("epsilon_decay_period must be positive")
    if args.target_network_frequency < 1:
        raise ValueError("target_network_frequency must be positive")
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

    q_network = DuelingQNetwork(envs).to(device)
    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate, eps=args.adam_eps)
    target_network = DuelingQNetwork(envs).to(device)
    target_network.load_state_dict(q_network.state_dict())

    # Uniform n-step replay: alpha 0 gives equal priorities over valid n-step
    # starts (and zero for starts without enough contiguous successors), so
    # the prioritized buffer degenerates to uniform sampling with unit weights.
    rb = PrioritizedAtariReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        n_envs=args.num_envs,
        n_step=args.n_step,
        gamma=args.gamma,
        alpha=0.0,
        beta=1.0,
    )

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

        # ALGO LOGIC: put action logic here
        epsilon = linearly_decaying_epsilon(
            args.epsilon_decay_period, global_step, args.learning_starts, args.end_e
        )
        with torch.no_grad():
            q_values = q_network(to_tensor(obs, device))
            greedy_actions = torch.argmax(q_values, dim=1)
        random_actions = torch.randint(n_actions, (args.num_envs,), device=device)
        explore = torch.rand(args.num_envs, device=device) < epsilon
        actions = torch.where(explore, random_actions, greedy_actions)

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
                data = rb.sample(args.batch_size)

                observations = data.observations.float()
                next_observations = data.next_observations.float()
                if args.data_augmentation:
                    observations = drq_image_augmentation(observations)
                    next_observations = drq_image_augmentation(next_observations)

                with torch.no_grad():
                    # double q-learning with the n-step Bellman update
                    next_q_online = q_network(next_observations)
                    best_actions = torch.argmax(next_q_online, dim=1)
                    next_q_target = target_network(next_observations)
                    next_q = next_q_target.gather(1, best_actions.unsqueeze(-1)).squeeze(-1)
                    gamma_n = args.gamma**args.n_step
                    td_target = data.rewards.squeeze(-1) + gamma_n * next_q * (
                        1 - data.dones.squeeze(-1).float()
                    )

                q_value = q_network(observations).gather(1, data.actions).squeeze(-1)
                loss = F.huber_loss(q_value, td_target)

                # optimize the model
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
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
                writer.add_scalar("losses/td_loss", loss.item(), global_step)
                writer.add_scalar("losses/q_values", q_value.mean().item(), global_step)
                writer.add_scalar("charts/SPS", sps, global_step)
                writer.add_scalar("charts/epsilon", epsilon, global_step)
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
            "algorithm": "drq",
            "backend": args.env_backend,
            "batch_size": args.batch_size,
            "benchmark": "full_training_loop",
            "compile": False,
            "data_augmentation": args.data_augmentation,
            "env_device": str(getattr(envs, "device", "cpu")),
            "env_id": args.env_id,
            "implementation": "original",
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
            "replay_backend": "numpy_frame_efficient_uniform_nstep",
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
