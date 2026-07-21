# Adapted from CleanRL's cleanrl/qdagger_dqn_atari_impalacnn.py
# (https://github.com/vwxyzjn/cleanrl, MIT; license in cleanrl/LICENSE.md).
# The Impala-CNN encoder is from the NeurIPS 2020 Procgen starter kit
# (https://github.com/AIcrowd/neurips2020-procgen-starter-kit, models/impala_cnn_torch.py),
# as in upstream CleanRL.
# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/qdagger/#qdagger_dqn_atari_jax_impalacnnpy
import json
import math
import os
import random
import sys
import time
from collections import deque
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
from dqn_atari import QNetwork as TeacherModel

from cleanrl_utils.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)
from cleanrl_utils.buffers import AtariReplayBuffer
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
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""

    # Algorithm specific arguments
    env_id: str = "BreakoutNoFrameskip-v4"
    """the id of the environment"""
    total_timesteps: int = 10000000
    """total online environment transitions after the offline phase"""
    learning_rate: float = 1e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    buffer_size: int = 1000000
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 1.0
    """the target network update rate"""
    target_network_frequency: int = 1000
    """learner updates between target-network updates"""
    batch_size: int = 32
    """the batch size of sample from the reply memory"""
    start_e: float = 1.0
    """the starting epsilon for exploration"""
    end_e: float = 0.01
    """the ending epsilon for exploration"""
    exploration_fraction: float = 0.10
    """the fraction of `total-timesteps` it takes from start-e to go end-e"""
    learning_starts: int = 80000
    """timestep to start learning"""
    learner_updates_per_vector_step: float = 0.25
    """gradient updates accrued per vector environment step; may be fractional"""
    replay_ratio: float | None = None
    """sampled replay items per collected transition; overrides learner-updates-per-vector-step"""
    max_training_seconds: float = 0.0
    """wall-clock limit for the online phase; zero disables it"""
    solve_reward: float | None = None
    """stop when the moving episodic return reaches this value"""
    solve_window: int = 20
    """number of completed episodes in the solve moving average"""

    # QDagger specific arguments
    teacher_policy_hf_repo: str = None
    """the huggingface repo of the teacher policy"""
    teacher_model_exp_name: str = "dqn_atari"
    """the experiment name of the teacher model"""
    teacher_model_path: str = None
    """local path to a teacher state dict; overrides the huggingface download"""
    teacher_eval_episodes: int = 10
    """the number of episodes to run the teacher policy evaluate"""
    teacher_steps: int = 500000
    """the number of transitions to run the teacher policy to generate the replay buffer"""
    offline_steps: int = 500000
    """the number of gradient updates on the teacher's replay buffer"""
    temperature: float = 1.0
    """the temperature parameter for qdagger"""

    benchmark: bool = False
    """measure the online loop only and print a JSON benchmark result"""
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


# taken from https://github.com/AIcrowd/neurips2020-procgen-starter-kit/blob/142d09586d2272a17f44481a115c4bd817cf6a94/models/impala_cnn_torch.py
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv0 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1)
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1)

    def forward(self, x):
        inputs = x
        x = nn.functional.relu(x)
        x = self.conv0(x)
        x = nn.functional.relu(x)
        x = self.conv1(x)
        return x + inputs


class ConvSequence(nn.Module):
    def __init__(self, input_shape, out_channels):
        super().__init__()
        self._input_shape = input_shape
        self._out_channels = out_channels
        self.conv = nn.Conv2d(in_channels=self._input_shape[0], out_channels=self._out_channels, kernel_size=3, padding=1)
        self.res_block0 = ResidualBlock(self._out_channels)
        self.res_block1 = ResidualBlock(self._out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = nn.functional.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        x = self.res_block0(x)
        x = self.res_block1(x)
        return x

    def get_output_shape(self):
        _c, h, w = self._input_shape
        return (self._out_channels, (h + 1) // 2, (w + 1) // 2)


# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        c, h, w = env.single_observation_space.shape
        shape = (c, h, w)
        conv_seqs = []
        for out_channels in [16, 32, 32]:
            conv_seq = ConvSequence(shape, out_channels)
            shape = conv_seq.get_output_shape()
            conv_seqs.append(conv_seq)
        conv_seqs += [
            nn.Flatten(),
            nn.ReLU(),
            nn.Linear(in_features=shape[0] * shape[1] * shape[2], out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=env.single_action_space.n),
        ]
        self.network = nn.Sequential(*conv_seqs)

    def forward(self, x):
        return self.network(x / 255.0)


def linear_schedule(start_e: float, end_e: float, duration: float, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


def kl_divergence_with_logits(target_logits, prediction_logits):
    """Per-sample on-policy distillation loss."""
    out = -F.softmax(target_logits, dim=-1) * (F.log_softmax(prediction_logits, dim=-1) - F.log_softmax(target_logits, dim=-1))
    return torch.sum(out, dim=-1)


if __name__ == "__main__":
    process_start = time.perf_counter()
    args = tyro.cli(Args)
    if args.num_envs < 1:
        raise ValueError("num_envs must be positive")
    if args.learner_updates_per_vector_step < 0:
        raise ValueError("learner_updates_per_vector_step must be non-negative")
    if args.replay_ratio is not None:
        if args.replay_ratio < 0:
            raise ValueError("replay_ratio must be non-negative")
        args.learner_updates_per_vector_step = args.replay_ratio * args.num_envs / args.batch_size
    if args.max_training_seconds < 0:
        raise ValueError("max_training_seconds must be non-negative")
    if args.benchmark_warmup_iterations < 0:
        raise ValueError("benchmark_warmup_iterations cannot be negative")
    if args.benchmark_measure_iterations < 1:
        raise ValueError("benchmark_measure_iterations must be positive")
    if args.teacher_policy_hf_repo is None:
        args.teacher_policy_hf_repo = f"cleanrl/{args.env_id}-{args.teacher_model_exp_name}-seed1"
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

    def reset_envs():
        reset_result = envs.reset(seed=args.seed) if args.env_backend != "envpool" else envs.reset()
        reset_obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result
        return to_tensor(reset_obs, device)

    def step_vector_env(actions):
        step_result = step_env(envs, actions)
        if len(step_result) == 5:
            next_obs, rewards, terminations, truncations, infos = step_result
            dones = done_tensor(terminations, truncations, device).bool()
        else:
            next_obs, rewards, dones_raw, infos = step_result
            dones = to_tensor(dones_raw, device, torch.bool)
        return to_tensor(next_obs, device), rewards, dones, infos

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

    def epsilon_greedy(model, obs, epsilon):
        with torch.no_grad():
            greedy_actions = torch.argmax(model(obs), dim=1)
        random_actions = torch.randint(n_actions, (args.num_envs,), device=greedy_actions.device)
        explore = torch.rand(args.num_envs, device=greedy_actions.device) < epsilon
        return torch.where(explore, random_actions, greedy_actions)

    def run_policy_episodes(model, num_episodes, epsilon):
        """Collect full-game episodic returns with the vectorized environments."""
        obs = reset_envs()
        returns: list[float] = []
        max_vector_steps = max(1, math.ceil(num_episodes * 27000 / args.num_envs))
        for _ in range(max_vector_steps):
            actions = epsilon_greedy(model, obs, epsilon)
            obs, _, dones, infos = step_vector_env(actions)
            for info in completed_episode_infos(infos, dones).get("final_info", ()):
                if info and "episode" in info:
                    returns.append(float(info["episode"]["r"]))
            if len(returns) >= num_episodes:
                break
        return returns

    q_network = QNetwork(envs).to(device)
    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)
    target_network = QNetwork(envs).to(device)
    target_network.load_state_dict(q_network.state_dict())

    # QDAGGER LOGIC:
    teacher_model = TeacherModel(envs).to(device)
    if not args.benchmark:
        if args.teacher_model_path is not None:
            teacher_model_path = args.teacher_model_path
        else:
            from huggingface_hub import hf_hub_download

            teacher_model_path = hf_hub_download(
                repo_id=args.teacher_policy_hf_repo, filename=f"{args.teacher_model_exp_name}.cleanrl_model"
            )
        teacher_model.load_state_dict(torch.load(teacher_model_path, map_location=device))
    teacher_model.eval()

    # evaluate the teacher model
    teacher_mean_return = None
    if not args.benchmark:
        teacher_episodic_returns = run_policy_episodes(teacher_model, args.teacher_eval_episodes, args.end_e)
        if teacher_episodic_returns:
            teacher_mean_return = float(np.mean(teacher_episodic_returns))
            print(f"teacher avg_episodic_return={teacher_mean_return} over {len(teacher_episodic_returns)} episodes")
            writer.add_scalar("charts/teacher/avg_episodic_return", teacher_mean_return, 0)
        else:
            print("teacher evaluation completed no episodes; distill_coeff stays at 1.0")

    # collect teacher data for args.teacher_steps
    # we assume we don't have access to the teacher's replay buffer
    # see Fig. A.19 in Agarwal et al. 2022 for more detail
    learner_updates = 0
    if not args.benchmark and args.offline_steps > 0:
        teacher_rb = AtariReplayBuffer(
            min(args.buffer_size, args.teacher_steps),
            envs.single_observation_space,
            envs.single_action_space,
            device,
            n_envs=args.num_envs,
        )
        obs = reset_envs()
        teacher_rb.initialize(obs)
        teacher_vector_steps = math.ceil(args.teacher_steps / args.num_envs)
        fill_start = time.time()
        for teacher_step in range(teacher_vector_steps):
            epsilon = linear_schedule(args.start_e, args.end_e, teacher_vector_steps, teacher_step)
            actions = epsilon_greedy(teacher_model, obs, epsilon)
            obs, rewards, dones, infos = step_vector_env(actions)
            teacher_rb.add(obs, actions, rewards, dones)
            if teacher_step % 10000 == 0:
                print(
                    f"filling teacher replay buffer: {teacher_step * args.num_envs}/{args.teacher_steps} "
                    f"({time.time() - fill_start:.0f}s)"
                )

        # offline training phase: train the student model using the qdagger loss
        offline_start = time.time()
        for offline_step in range(args.offline_steps):
            data = teacher_rb.sample(args.batch_size)
            with torch.no_grad():
                target_max, _ = target_network(data.next_observations).max(dim=1)
                td_target = data.rewards.flatten() + args.gamma * target_max * (1 - data.dones.flatten())
                teacher_q_values = teacher_model(data.observations) / args.temperature

            student_q_values = q_network(data.observations)
            old_val = student_q_values.gather(1, data.actions).squeeze()
            q_loss = F.mse_loss(td_target, old_val)

            distill_loss = kl_divergence_with_logits(teacher_q_values, student_q_values / args.temperature).mean()

            loss = q_loss + 1.0 * distill_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            learner_updates += 1

            # update the target network
            if learner_updates % args.target_network_frequency == 0:
                for target_network_param, q_network_param in zip(target_network.parameters(), q_network.parameters()):
                    target_network_param.data.copy_(
                        args.tau * q_network_param.data + (1.0 - args.tau) * target_network_param.data
                    )

            if offline_step % 1000 == 0:
                print(
                    f"offline student training: {offline_step}/{args.offline_steps}, "
                    f"loss={loss.item():.4f} ({time.time() - offline_start:.0f}s)"
                )
                writer.add_scalar("charts/offline/loss", loss, offline_step)
                writer.add_scalar("charts/offline/q_loss", q_loss, offline_step)
                writer.add_scalar("charts/offline/distill_loss", distill_loss, offline_step)
        del teacher_rb

    rb = AtariReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        n_envs=args.num_envs,
    )
    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    obs = reset_envs()
    rb.initialize(obs)
    episodic_returns = deque(maxlen=10)
    episode_stats = EpisodeStats(args.solve_window, args.solve_reward)
    global_step = args.offline_steps if not args.benchmark else 0
    online_steps = 0
    update_budget = 0.0
    next_target_update = learner_updates + args.target_network_frequency
    next_log_step = max(10000, args.num_envs)
    num_vector_steps = math.ceil(args.total_timesteps / args.num_envs)
    if args.benchmark:
        num_vector_steps = args.benchmark_warmup_iterations + args.benchmark_measure_iterations
    benchmark_start = None
    benchmark_start_step = None
    benchmark_start_updates = None
    # online training phase
    for vector_step in range(num_vector_steps):
        if args.benchmark and vector_step == args.benchmark_warmup_iterations:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            benchmark_start = time.perf_counter()
            benchmark_start_step = online_steps
            benchmark_start_updates = learner_updates
        if args.max_training_seconds and time.time() - start_time >= args.max_training_seconds:
            break
        # ALGO LOGIC: put action logic here
        epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step)
        actions = epsilon_greedy(q_network, obs, epsilon)

        # TRY NOT TO MODIFY: execute the game and log data.
        obs, rewards, dones, infos = step_vector_env(actions)
        global_step += args.num_envs
        online_steps += args.num_envs

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        solved = False
        if not args.benchmark:
            episode_infos = completed_episode_infos(infos, dones)
            for info in episode_infos.get("final_info", ()):
                if info and "episode" in info:
                    episodic_returns.append(float(info["episode"]["r"]))
            solved = episode_stats.update(episode_infos, global_step, writer)

        rb.add(obs, actions, rewards, dones)

        # ALGO LOGIC: training.
        if global_step > args.learning_starts and len(rb) >= args.batch_size:
            update_budget += args.learner_updates_per_vector_step
            num_updates = int(update_budget)
            update_budget -= num_updates
            if len(episodic_returns) < 10 or teacher_mean_return is None:
                distill_coeff = 1.0
            else:
                teacher_return = teacher_mean_return if teacher_mean_return != 0 else 1e-8
                distill_coeff = max(1 - np.mean(episodic_returns) / teacher_return, 0)
            for _ in range(num_updates):
                data = rb.sample(args.batch_size)
                # perform a gradient-descent step
                with torch.no_grad():
                    target_max, _ = target_network(data.next_observations).max(dim=1)
                    td_target = data.rewards.flatten() + args.gamma * target_max * (1 - data.dones.flatten())
                    teacher_q_values = teacher_model(data.observations) / args.temperature

                student_q_values = q_network(data.observations)
                old_val = student_q_values.gather(1, data.actions).squeeze()
                q_loss = F.mse_loss(td_target, old_val)

                distill_loss = kl_divergence_with_logits(teacher_q_values, student_q_values / args.temperature).mean()

                loss = q_loss + distill_coeff * distill_loss

                # optimize the model
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                learner_updates += 1

                # update the target network
                if learner_updates >= next_target_update:
                    for target_network_param, q_network_param in zip(target_network.parameters(), q_network.parameters()):
                        target_network_param.data.copy_(
                            args.tau * q_network_param.data + (1.0 - args.tau) * target_network_param.data
                        )
                    next_target_update = (
                        learner_updates // args.target_network_frequency + 1
                    ) * args.target_network_frequency

            if writer is not None and global_step >= next_log_step and num_updates:
                sps = int(online_steps / (time.time() - start_time))
                writer.add_scalar("losses/loss", loss, global_step)
                writer.add_scalar("losses/td_loss", q_loss, global_step)
                writer.add_scalar("losses/distill_loss", distill_loss, global_step)
                writer.add_scalar("losses/q_values", old_val.mean().item(), global_step)
                writer.add_scalar("charts/distill_coeff", distill_coeff, global_step)
                writer.add_scalar("charts/epsilon", epsilon, global_step)
                writer.add_scalar("charts/SPS", sps, global_step)
                writer.add_scalar("charts/learner_updates", learner_updates, global_step)
                print("SPS:", sps)
                next_log_step = global_step + max(10000, args.num_envs)
        if solved:
            break

    if args.benchmark:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        benchmark_end = time.perf_counter()
        measured_steps = online_steps - benchmark_start_step
        measured_updates = learner_updates - benchmark_start_updates
        measured_seconds = benchmark_end - benchmark_start
        result = {
            "algorithm": "qdagger",
            "backend": args.env_backend,
            "batch_size": args.batch_size,
            "benchmark": "online_training_loop",
            "compile": False,
            "env_id": args.env_id,
            "learner_updates": measured_updates,
            "measure_iterations": args.benchmark_measure_iterations,
            "measured_seconds": measured_seconds,
            "measured_steps": measured_steps,
            "num_envs": args.num_envs,
            "peak_cuda_memory_mb": (
                torch.cuda.max_memory_allocated(device) / (1024**2) if device.type == "cuda" else 0.0
            ),
            "process_seconds": benchmark_end - process_start,
            "replay_ratio": measured_updates * args.batch_size / max(measured_steps, 1),
            "schema_version": 1,
            "sps": measured_steps / measured_seconds,
            "ups": measured_updates / measured_seconds,
            "warmup_iterations": args.benchmark_warmup_iterations,
        }
        print(f"BENCHMARK_RESULT {json.dumps(result, sort_keys=True)}", flush=True)
    else:
        elapsed = time.time() - start_time
        print("SPS:", int(online_steps / max(elapsed, 1e-9)))
        print("learner updates:", learner_updates)
        episode_stats.print_summary()

    if args.save_model and not args.benchmark:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(q_network.state_dict(), model_path)
        print(f"model saved to {model_path}")

        if args.env_backend == "gymnasium":
            from cleanrl_utils.evals.dqn_eval import evaluate

            episodic_returns = evaluate(
                model_path,
                make_env,
                args.env_id,
                eval_episodes=10,
                run_name=f"{run_name}-eval",
                Model=QNetwork,
                device=device,
                epsilon=args.end_e,
            )
        else:
            episodic_returns = run_policy_episodes(q_network, 10, args.end_e)
        for idx, episodic_return in enumerate(episodic_returns):
            writer.add_scalar("eval/episodic_return", episodic_return, idx)

        if args.upload_model:
            from cleanrl_utils.huggingface import push_to_hub

            repo_name = f"{args.env_id}-{args.exp_name}-seed{args.seed}"
            repo_id = f"{args.hf_entity}/{repo_name}" if args.hf_entity else repo_name
            push_to_hub(args, episodic_returns, repo_id, "Qdagger", f"runs/{run_name}", f"videos/{run_name}-eval")

    envs.close()
    if writer is not None:
        writer.close()
