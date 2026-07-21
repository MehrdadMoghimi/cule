# Atari network, hyperparameters, and training loop adapted from CleanRL's
# cleanrl/dqn_atari.py (https://github.com/vwxyzjn/cleanrl, MIT).
# The torch.compile / CUDA-graph structure — detached inference params via
# from_module, CudaGraphModule capture, and GPU-resident TorchRL replay — follows
# LeanRL's leanrl/dqn_torchcompile.py (https://github.com/meta-pytorch/LeanRL,
# MIT), which targets classic-control DQN rather than Atari.
# Both licenses are reproduced in cleanrl/LICENSE.md.
# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/dqn/#dqn_ataripy
import csv
import json
import math
import os
import random
import sys
import time
from collections import deque
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm
import tyro
import wandb
from tensordict import TensorDict, from_module
from tensordict.nn import CudaGraphModule
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
    to_tensor,
)
from cleanrl_utils.atari_eval import evaluate_cule_policy

from cleanrl_utils.atari_wrappers import (  # isort:skip
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)


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
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    env_backend: str = "cule"
    """environment backend: `cule` or `gymnasium`"""
    cule_device: str = "auto"
    """CuLE device; auto uses CUDA for 32+ envs and CPU for smaller batches"""
    track: bool = False
    """if toggled, track the experiment with Weights and Biases"""

    # Algorithm specific arguments
    env_id: str = "Breakout-v5"
    """the Atari environment id"""
    total_timesteps: int = 10000000
    """total environment transitions"""
    learning_rate: float = 1e-4
    """the learning rate of the optimizer"""
    num_envs: int = 256
    """the number of parallel Atari environments"""
    buffer_size: int = 100000
    """the replay memory capacity in individual transitions"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 1.0
    """the target network update rate"""
    target_network_frequency: int = 250
    """learner updates between target-network updates"""
    batch_size: int = 512
    """the replay sample batch size"""
    start_e: float = 1
    """the starting epsilon for exploration"""
    end_e: float = 0.01
    """the ending epsilon for exploration"""
    exploration_fraction: float = 0.10
    """the fraction of total transitions used to anneal epsilon"""
    learning_starts: int = 80000
    """the number of collected transitions before learning starts"""
    learner_updates_per_vector_step: float = 1.0
    """gradient updates accrued per vector environment step; may be fractional"""
    replay_ratio: float | None = 1.0
    """sampled replay items per collected transition; overrides learner-updates-per-vector-step"""

    measure_burnin: int = 3
    """number of learner updates to exclude from the speed measurement"""
    compile: bool = False
    """whether to use torch.compile"""
    cudagraphs: bool = False
    """whether to use CUDA graphs on top of compile"""
    benchmark: bool = False
    """run a fixed warmup/measurement window and print a JSON benchmark result"""
    benchmark_warmup_iterations: int = 10
    """vector environment steps excluded from benchmark timing"""
    benchmark_measure_iterations: int = 30
    """vector environment steps included in benchmark timing"""

    evaluation_interval: int = 1000000
    """policy transitions between deterministic full-game evaluations"""
    evaluation_episodes: int = 10
    """complete unclipped games per evaluation"""
    evaluation_seed: int = 10000
    """first seed in the fixed evaluation seed set"""
    evaluation_max_episode_steps: int = 18000
    """maximum frame-skipped steps per evaluation game"""
    learning_curve_path: str | None = None
    """optional CSV path for full-game evaluation results"""


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


class QNetwork(nn.Module):
    def __init__(self, envs, device=None):
        super().__init__()
        if envs.single_observation_space.shape != (4, 84, 84):
            raise ValueError(
                "dqn_torchcompile.py expects four stacked 84x84 Atari frames, "
                f"got {envs.single_observation_space.shape}"
            )
        self.network = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4, device=device),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, device=device),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1, device=device),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512, device=device),
            nn.ReLU(),
            nn.Linear(512, envs.single_action_space.n, device=device),
        )

    def forward(self, x):
        return self.network(x / 255.0)


def linear_schedule(start_e: float, end_e: float, duration: float, t: int) -> float:
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


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
    if args.exploration_fraction <= 0:
        raise ValueError("exploration_fraction must be positive")
    if args.learner_updates_per_vector_step < 0:
        raise ValueError("learner_updates_per_vector_step must be non-negative")
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

    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{args.compile}__{args.cudagraphs}"
    if args.track:
        wandb.init(
            project="dqn_atari",
            name=f"{os.path.splitext(os.path.basename(__file__))[0]}-{run_name}",
            config=vars(args),
            save_code=True,
        )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda:0" if torch.cuda.is_available() and args.cuda else "cpu")
    if args.cudagraphs and device.type != "cuda":
        raise ValueError("cudagraphs requires CUDA")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    if args.env_backend == "cule":
        env_device = resolve_cule_device(args.cule_device, device, args.num_envs)
        envs = make_cule_env(args.env_id, args.num_envs, env_device, args.seed, args.capture_video)
    elif args.env_backend == "gymnasium":
        envs = gym.vector.SyncVectorEnv(
            [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
        )
    else:
        raise ValueError(f"unsupported environment backend: {args.env_backend}")

    if not isinstance(envs.single_action_space, gym.spaces.Discrete):
        raise ValueError("only discrete action spaces are supported")
    n_act = envs.single_action_space.n

    q_network = QNetwork(envs, device=device)
    q_network_detach = QNetwork(envs, device=device)
    params_vals = from_module(q_network).detach()
    params_vals.to_module(q_network_detach, preserve_module_state=True)

    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate, capturable=args.cudagraphs and not args.compile)

    target_network = QNetwork(envs, device=device)
    target_params = params_vals.clone().lock_()
    target_params.to_module(target_network, preserve_module_state=True)

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
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        evaluation_started = time.perf_counter()
        q_network.eval()

        def greedy_actions(states: torch.Tensor) -> torch.Tensor:
            return q_network(states).argmax(dim=1)

        stats = evaluate_cule_policy(
            args.env_id,
            greedy_actions,
            device,
            num_episodes=args.evaluation_episodes,
            seed=args.evaluation_seed,
            max_episode_steps=args.evaluation_max_episode_steps,
        )
        q_network.train()
        evaluation_seconds = time.perf_counter() - evaluation_started
        row = {
            "algorithm": "dqn",
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

    def update(data):
        with torch.no_grad():
            target_max = target_network(data["next_observations"]).max(dim=1).values
            td_target = data["rewards"] + args.gamma * target_max * (~data["dones"]).float()
        old_val = q_network(data["observations"]).gather(1, data["actions"].unsqueeze(-1)).squeeze(-1)
        loss = F.mse_loss(td_target, old_val)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.detach(), old_val.detach().mean()

    def policy(obs, epsilon):
        q_values = q_network_detach(obs)
        greedy_actions = torch.argmax(q_values, dim=1)
        random_actions = torch.randint(n_act, greedy_actions.shape, device=greedy_actions.device)
        explore = torch.rand(greedy_actions.shape, device=greedy_actions.device) < epsilon
        return torch.where(explore, random_actions, greedy_actions)

    rb = ReplayBuffer(storage=LazyTensorStorage(args.buffer_size, device=device))

    if args.compile:
        # reduce-overhead implicitly enables Inductor CUDA graphs, which cannot
        # safely retain CuLE's in-place, reused observation storage.
        mode = None
        update = torch.compile(update, mode=mode)
        policy = torch.compile(policy, mode=mode, fullgraph=True)

    if args.cudagraphs:
        update = CudaGraphModule(update, warmup=20)
        policy = CudaGraphModule(policy, warmup=20)

    reset_result = envs.reset(seed=args.seed)
    reset_obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result
    obs = to_tensor(reset_obs, device)
    epsilon_tensor = torch.zeros((), device=device)
    evaluation_seconds_total = 0.0
    if curve_writer is not None and not args.benchmark:
        evaluate_and_log(0, 0.0)
    learning_wall_start = time.perf_counter()
    next_evaluation_step = args.evaluation_interval
    avg_returns = deque(maxlen=20)
    global_step = 0
    learner_updates = 0
    update_budget = 0.0
    next_target_update = args.target_network_frequency
    loss = None
    q_value = None
    desc = ""
    measure_start_time = None
    measure_start_step = 0

    num_vector_steps = math.ceil(args.total_timesteps / args.num_envs)
    if args.benchmark:
        num_vector_steps = args.benchmark_warmup_iterations + args.benchmark_measure_iterations
    pbar = tqdm.tqdm(range(num_vector_steps), disable=args.benchmark)
    benchmark_start = None
    benchmark_start_step = None
    benchmark_start_updates = None
    for vector_step in pbar:
        if args.benchmark and vector_step == args.benchmark_warmup_iterations:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            benchmark_start = time.perf_counter()
            benchmark_start_step = global_step
            benchmark_start_updates = learner_updates
        epsilon_value = linear_schedule(
            args.start_e,
            args.end_e,
            args.exploration_fraction * args.total_timesteps,
            global_step,
        )
        epsilon_tensor.fill_(epsilon_value)
        with torch.no_grad():
            actions = policy(obs, epsilon_tensor)

        # CuLE reuses and mutates its observation tensor on every step.
        transition_obs = obs.clone()
        next_obs_raw, rewards, terminations, truncations, infos = step_env(envs, actions)
        next_obs = to_tensor(next_obs_raw, device)
        rewards = to_tensor(rewards, device, torch.float32).view(-1)
        dones = done_tensor(terminations, truncations, device).bool()

        transition_next_obs = next_obs.clone()
        if args.env_backend == "gymnasium" and np.asarray(truncations).any():
            for idx, truncated in enumerate(truncations):
                if truncated:
                    transition_next_obs[idx] = to_tensor(infos["final_observation"][idx], device)

        rb.extend(
            TensorDict(
                {
                    "observations": transition_obs,
                    "next_observations": transition_next_obs,
                    "actions": actions,
                    "rewards": rewards,
                    "dones": dones,
                },
                batch_size=[args.num_envs],
                device=device,
            )
        )
        obs = next_obs
        global_step += args.num_envs

        for info in infos.get("final_info", ()):
            if info and "episode" in info:
                avg_returns.append(float(info["episode"]["r"]))
        if avg_returns:
            desc = f", episodic_return={sum(avg_returns) / len(avg_returns):.2f}"

        if global_step > args.learning_starts and len(rb) >= args.batch_size:
            update_budget += args.learner_updates_per_vector_step
            num_updates = int(update_budget)
            update_budget -= num_updates
            for _ in range(num_updates):
                torch.compiler.cudagraph_mark_step_begin()
                loss, q_value = update(rb.sample(args.batch_size))
                learner_updates += 1

                if learner_updates >= next_target_update:
                    target_params.lerp_(params_vals, args.tau)
                    next_target_update = (
                        learner_updates // args.target_network_frequency + 1
                    ) * args.target_network_frequency

        if curve_writer is not None and not args.benchmark and global_step >= next_evaluation_step:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            training_seconds = time.perf_counter() - learning_wall_start - evaluation_seconds_total
            evaluation_seconds_total += evaluate_and_log(global_step, training_seconds)
            while next_evaluation_step <= global_step:
                next_evaluation_step += args.evaluation_interval

                if measure_start_time is None and learner_updates >= args.measure_burnin:
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    measure_start_time = time.time()
                    measure_start_step = global_step

        if not args.benchmark and measure_start_time is not None and global_step % max(args.num_envs * 100, 1) == 0:
            speed = (global_step - measure_start_step) / max(time.time() - measure_start_time, 1e-9)
            pbar.set_description(f"speed: {speed:4.1f} sps, epsilon: {epsilon_value:4.2f}{desc}")
            if args.track and loss is not None:
                wandb.log(
                    {
                        "speed": speed,
                        "episode_return": np.mean(avg_returns) if avg_returns else np.nan,
                        "loss": loss,
                        "q_value": q_value,
                        "epsilon": epsilon_value,
                    },
                    step=global_step,
                )

    if args.benchmark:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        benchmark_end = time.perf_counter()
        measured_steps = global_step - benchmark_start_step
        measured_updates = learner_updates - benchmark_start_updates
        measured_seconds = benchmark_end - benchmark_start
        result = {
            "algorithm": "dqn",
            "backend": args.env_backend,
            "batch_size": args.batch_size,
            "benchmark": "full_training_loop",
            "compile": args.compile,
            "cudagraphs": args.cudagraphs,
            "env_id": args.env_id,
            "learner_updates": measured_updates,
            "measure_iterations": args.benchmark_measure_iterations,
            "measured_seconds": measured_seconds,
            "measured_steps": measured_steps,
            "num_envs": args.num_envs,
            "peak_cuda_memory_mb": (
                torch.cuda.max_memory_allocated() / (1024**2) if device.type == "cuda" else 0.0
            ),
            "process_seconds": benchmark_end - process_start,
            "replay_ratio": measured_updates * args.batch_size / measured_steps,
            "schema_version": 1,
            "sps": measured_steps / measured_seconds,
            "ups": measured_updates / measured_seconds,
            "warmup_iterations": args.benchmark_warmup_iterations,
        }
        print(f"BENCHMARK_RESULT {json.dumps(result, sort_keys=True)}", flush=True)

    if curve_writer is not None and not args.benchmark and last_evaluation_step[0] != global_step:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        training_seconds = time.perf_counter() - learning_wall_start - evaluation_seconds_total
        evaluate_and_log(global_step, training_seconds)

    envs.close()
    if curve_file is not None:
        curve_file.close()
