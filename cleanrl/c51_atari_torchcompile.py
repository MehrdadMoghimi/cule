# torch.compile twin of c51_atari.py, which is adapted from CleanRL's
# cleanrl/c51_atari.py (https://github.com/vwxyzjn/cleanrl, MIT).  The
# compile / CUDA-graph structure follows LeanRL
# (https://github.com/meta-pytorch/LeanRL, MIT).  Both licenses are
# reproduced in cleanrl/LICENSE.md.
# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/c51/#c51_ataripy
"""C51 Atari with optional torch.compile and GPU-resident TorchRL replay.

Environment stepping and replay-buffer operations deliberately remain outside
the compiled regions.  CuLE reuses its observation tensor after every step, so
each transition stores explicit clones before the next environment step.
"""

import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from tensordict import TensorDict
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
    """environment backend: cule or gymnasium"""
    cule_device: str = "auto"
    """CuLE device; auto uses CUDA for 32+ environments and CPU for smaller batches"""
    save_model: bool = False
    """whether to save the model into the runs/{run_name} folder"""
    upload_model: bool = False
    """whether to upload the saved model to Hugging Face"""
    hf_entity: str = ""
    """the user or org name of the Hugging Face model repository"""

    # Algorithm specific arguments
    env_id: str = "BreakoutNoFrameskip-v4"
    """the id of the environment"""
    total_timesteps: int = 10_000_000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 256
    """the number of parallel game environments"""
    n_atoms: int = 51
    """the number of atoms"""
    v_min: float = -10
    """the return lower bound"""
    v_max: float = 10
    """the return upper bound"""
    buffer_size: int = 100_000
    """the replay memory size in individual transitions"""
    gamma: float = 0.99
    """the discount factor gamma"""
    target_network_frequency: int = 2500
    """learner updates between target-network updates"""
    batch_size: int = 512
    """the replay sample batch size"""
    start_e: float = 1
    """the starting epsilon for exploration"""
    end_e: float = 0.01
    """the ending epsilon for exploration"""
    exploration_fraction: float = 0.10
    """the fraction of total timesteps used to anneal epsilon"""
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

    compile: bool = False
    """whether to compile the fixed-shape policy and learner regions"""
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


class QNetwork(nn.Module):
    def __init__(self, env, n_atoms=51, v_min=-10, v_max=10):
        super().__init__()
        self.n_atoms = int(n_atoms)
        self.register_buffer("atoms", torch.linspace(v_min, v_max, steps=n_atoms))
        self.n_actions = int(env.single_action_space.n)
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
            nn.Linear(512, self.n_actions * n_atoms),
        )

    def get_action(self, x: torch.Tensor, action: torch.Tensor | None = None):
        logits = self.network(x / 255.0)
        # Keep the batch dimension inferred.  Passing x.shape[0] through view
        # triggers an Inductor scalar extraction on this CUDA stack.
        pmfs = torch.softmax(logits.reshape(-1, self.n_actions, self.n_atoms), dim=2)
        q_values = (pmfs * self.atoms).sum(2)
        if action is None:
            action = torch.argmax(q_values, dim=1)
        batch_indices = torch.arange(pmfs.shape[0], device=pmfs.device)
        return action, pmfs[batch_indices, action]


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
    if args.n_atoms < 2:
        raise ValueError("n_atoms must be at least two")
    if args.target_network_frequency < 1:
        raise ValueError("target_network_frequency must be positive")
    if args.exploration_fraction <= 0:
        raise ValueError("exploration_fraction must be positive")
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

    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__compile={args.compile}"
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
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

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

    q_network = QNetwork(envs, args.n_atoms, args.v_min, args.v_max).to(device)
    target_network = QNetwork(envs, args.n_atoms, args.v_min, args.v_max).to(device)
    target_network.load_state_dict(q_network.state_dict())
    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate, eps=0.01 / args.batch_size)
    # Match dqn_torchcompile.py: keep complete stacked transitions on the
    # learner device, eliminating the NumPy replay and CPU-to-GPU sample copy.
    rb = ReplayBuffer(storage=LazyTensorStorage(args.buffer_size, device=device))
    n_actions = int(envs.single_action_space.n)

    def policy(observations: torch.Tensor, epsilon: torch.Tensor) -> torch.Tensor:
        greedy_actions, _ = q_network.get_action(observations)
        random_actions = torch.randint(n_actions, greedy_actions.shape, device=greedy_actions.device)
        explore = torch.rand(greedy_actions.shape, device=greedy_actions.device) < epsilon
        return torch.where(explore, random_actions, greedy_actions)

    def update(data: TensorDict) -> tuple[torch.Tensor, torch.Tensor]:
        observations = data["observations"]
        actions = data["actions"]
        next_observations = data["next_observations"]
        dones = data["dones"].float().unsqueeze(-1)
        rewards = data["rewards"].unsqueeze(-1)
        with torch.no_grad():
            _, next_pmfs = target_network.get_action(next_observations)
            next_atoms = rewards + args.gamma * target_network.atoms * (1 - dones)
            delta_z = target_network.atoms[1] - target_network.atoms[0]
            tz = next_atoms.clamp(args.v_min, args.v_max)
            b = (tz - args.v_min) / delta_z
            lower = b.floor().clamp(0, args.n_atoms - 1)
            upper = b.ceil().clamp(0, args.n_atoms - 1)
            lower_mass = (upper + (lower == upper).float() - b) * next_pmfs
            upper_mass = (b - lower) * next_pmfs
            target_pmfs = torch.zeros_like(next_pmfs)
            target_pmfs.scatter_add_(1, lower.long(), lower_mass)
            target_pmfs.scatter_add_(1, upper.long(), upper_mass)

        _, old_pmfs = q_network.get_action(observations, actions.flatten())
        loss = -(target_pmfs * old_pmfs.clamp(min=1e-5, max=1 - 1e-5).log()).sum(dim=-1).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        q_value = (old_pmfs * q_network.atoms).sum(dim=1).mean()
        return loss.detach(), q_value.detach()

    if args.compile:
        # CuLE reuses its observation buffer after every environment step.
        # Avoid reduce-overhead's implicit CUDA graphs, which can retain/reuse
        # graph-owned tensors across those in-place environment updates.
        policy = torch.compile(policy, mode=None, fullgraph=True)
        update = torch.compile(update, mode=None)

    start_time = time.perf_counter()
    reset_result = envs.reset(seed=args.seed)
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

    for vector_step in range(num_vector_steps):
        if args.max_training_seconds and time.perf_counter() - start_time >= args.max_training_seconds:
            break
        if args.benchmark and vector_step == args.benchmark_warmup_iterations:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            benchmark_start = time.perf_counter()
            benchmark_start_step = global_step
            benchmark_start_updates = learner_updates

        epsilon = torch.tensor(
            linear_schedule(
                args.start_e,
                args.end_e,
                args.exploration_fraction * args.total_timesteps,
                global_step,
            ),
            device=device,
        )
        with torch.no_grad():
            actions = policy(obs, epsilon)

        # CuLE mutates and reuses the observation tensor on every step.  The
        # replay buffer must therefore receive clones, not CuLE-owned views.
        transition_obs = obs.clone()
        next_obs_raw, rewards, terminations, truncations, infos = step_env(envs, actions)
        next_obs = to_tensor(next_obs_raw, device)
        rewards = to_tensor(rewards, device, torch.float32).view(-1)
        transition_dones = done_tensor(terminations, truncations, device).bool()
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
                    "dones": transition_dones,
                },
                batch_size=[args.num_envs],
                device=device,
            )
        )
        global_step += args.num_envs
        obs = next_obs

        solved = False
        if not args.benchmark:
            solved = episode_stats.update(infos, global_step, writer)

        if global_step > args.learning_starts and len(rb) >= args.batch_size:
            update_budget += args.learner_updates_per_vector_step
            num_updates = int(update_budget)
            update_budget -= num_updates
            for _ in range(num_updates):
                torch.compiler.cudagraph_mark_step_begin()
                last_loss, last_q_value = update(rb.sample(args.batch_size))
                learner_updates += 1

            if learner_updates >= next_target_update:
                target_network.load_state_dict(q_network.state_dict())
                next_target_update = (
                    learner_updates // args.target_network_frequency + 1
                ) * args.target_network_frequency

            if not args.benchmark and global_step >= next_log_step and num_updates:
                elapsed = time.perf_counter() - start_time
                sps = int(global_step / max(elapsed, 1e-9))
                writer.add_scalar("losses/loss", last_loss.item(), global_step)
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
            "algorithm": "c51",
            "backend": args.env_backend,
            "batch_size": args.batch_size,
            "benchmark": "full_training_loop",
            "compile": args.compile,
            "env_device": str(getattr(envs, "device", "cpu")),
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
            "replay_backend": "torchrl_lazy_tensor_storage",
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
        from cleanrl_utils.evals.c51_eval import evaluate

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
        for idx, episodic_return in enumerate(episodic_returns):
            writer.add_scalar("eval/episodic_return", episodic_return, idx)
        if args.upload_model:
            from cleanrl_utils.huggingface import push_to_hub

            repo_name = f"{args.env_id}-{args.exp_name}-seed{args.seed}"
            repo_id = f"{args.hf_entity}/{repo_name}" if args.hf_entity else repo_name
            push_to_hub(args, episodic_returns, repo_id, "C51", f"runs/{run_name}", f"videos/{run_name}-eval")

    envs.close()
    if writer is not None:
        writer.close()
