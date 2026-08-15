# torch.compile twin of hadamax_pqn_atari_envpool.py.
#
# Hadamax (Kooi et al., NeurIPS 2025, https://arxiv.org/abs/2505.15345) ported
# from https://github.com/jacobkooi/hadamax (purejaxql/networks.py).  Relative to
# pqn_atari_envpool_torchcompile.py the only change is `QNetwork`; the compiled
# policy / target / learner regions and the CUDA-graph structure are inherited
# unchanged.  The Hadamax encoder is fixed-shape and capture-safe.
#
# The trainer body is adapted from CleanRL's
# cleanrl/pqn_atari_envpool.py (https://github.com/vwxyzjn/cleanrl, MIT).  The
# compile / CUDA-graph structure follows LeanRL
# (https://github.com/meta-pytorch/LeanRL, MIT).  Both licenses are
# reproduced in cleanrl/LICENSE.md.
# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/pqn/#pqn_atari_envpoolpy#
# CONFIRMED against the official implementation:
# `tests/crosscheck/check_hadamax.py` runs jacobkooi/hadamax's real Flax
# QNetwork, loads its parameters into this file, and diffs every stage. 12/12
# components match on CPU and CUDA (<= 1e-5): each Hadamard block, the 7744-wide
# features, and the Q logits. One deliberate difference: Flax is NHWC and
# flattens (H, W, C) while PyTorch is NCHW and flattens (C, H, W), so the
# projection's input columns are a permutation of upstream's -- the same model,
# relabelled, which the cross-check undoes before comparing.
"""PQN Atari with optional torch.compile and CUDA graphs on fixed-shape regions.

Rollout collection, environment interaction, episode bookkeeping, and minibatch
orchestration stay eager.  Policy inference (including epsilon-greedy sampling),
Q(lambda) target construction, and the fixed-size learner update can be compiled
independently, and the policy and learner can additionally be captured with
CudaGraphModule, without retaining CuLE's reused observation buffers.
"""

import csv
import json
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
from tensordict import from_module
from tensordict.nn import CudaGraphModule
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

torch.set_float32_matmul_precision("high")

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from cule_env import done_tensor, make_cule_env, resolve_cule_device, step_env, to_numpy, to_tensor
from cleanrl_utils.atari_eval import evaluate_cule_policy


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
    """the WandB project name"""
    wandb_entity: str | None = None
    """the WandB entity"""
    capture_video: bool = False
    """whether to capture videos of the agent performances"""
    env_backend: str = "cule"
    """environment backend: cule or envpool"""
    cule_device: str = "auto"
    """CuLE device; auto uses CUDA for 32+ environments and CPU for smaller batches"""

    # Algorithm specific arguments
    env_id: str = "BreakoutNoFrameskip-v4"
    """the id of the environment"""
    total_timesteps: int = 10_000_000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 256
    """the number of parallel game environments"""
    num_steps: int = 32
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """toggle learning-rate annealing"""
    gamma: float = 0.99
    """the discount factor gamma"""
    num_minibatches: int = 4
    """the number of minibatches"""
    update_epochs: int = 4
    """the number of update epochs"""
    max_grad_norm: float = 10.0
    """the maximum gradient norm"""
    start_e: float = 1
    """the starting epsilon for exploration"""
    end_e: float = 0.01
    """the ending epsilon for exploration"""
    exploration_fraction: float = 0.10
    """the fraction of total timesteps used to anneal epsilon"""
    q_lambda: float = 0.65
    """the lambda for the Q-learning algorithm"""

    compile: bool = False
    """whether to compile policy, target, and learner tensor regions"""
    cudagraphs: bool = False
    """whether to wrap the policy and learner update in CudaGraphModule"""
    benchmark: bool = False
    """run a fixed warmup/measurement window and print a JSON benchmark result"""
    benchmark_warmup_iterations: int = 3
    """full training iterations excluded from benchmark timing"""
    benchmark_measure_iterations: int = 10
    """full training iterations included in benchmark timing"""

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

    # Filled in at runtime
    batch_size: int = 0
    """the computed rollout batch size"""
    minibatch_size: int = 0
    """the computed minibatch size"""
    num_iterations: int = 0
    """the computed number of iterations"""


class RecordEpisodeStatistics(gym.Wrapper):
    """Small EnvPool-compatible episode-statistics adapter."""

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
        self.episode_returns *= 1 - infos["terminated"]
        self.episode_lengths *= 1 - infos["terminated"]
        infos["r"] = self.returned_episode_returns
        infos["l"] = self.returned_episode_lengths
        if len(result) == 5:
            return observations, rewards, terminations, truncations, infos
        return observations, rewards, dones, infos


# jax.nn.gelu defaults to approximate=True (the tanh form) and flax.linen's
# LayerNorm defaults to epsilon=1e-6; PyTorch defaults to the exact erf GELU and
# eps=1e-5.  Both defaults are matched to Flax below so the encoder is numerically
# the official one.
FLAX_LAYER_NORM_EPS = 1e-6


class ChannelLayerNorm(nn.Module):
    """LayerNorm over the channel axis of an NCHW tensor.

    Flax's `nn.LayerNorm()` reduces the trailing axis, which is the channel axis
    in the official NHWC encoder, so statistics are per spatial position.  The
    PQN parent instead normalizes jointly over [C, H, W]; Hadamax keeps the
    official per-position convention.
    """

    def __init__(self, channels: int):
        super().__init__()
        # Flax's nn.LayerNorm defaults to epsilon=1e-6, PyTorch's to 1e-5.
        self.norm = nn.LayerNorm(channels, eps=FLAX_LAYER_NORM_EPS)

    def forward(self, x):
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class HadamaxBlock(nn.Module):
    """Max-pooled Hadamard product of two GELU-activated parallel convolutions.

    Both branches see the same input, are normalized *before* the activation,
    and are combined multiplicatively; the max-pool then does the spatial
    downsampling that strided convolutions do in the Nature CNN.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, pool: nn.Module):
        super().__init__()
        # padding="same" with stride 1 reproduces Flax's SAME padding exactly,
        # including the asymmetric split for even kernels.
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding="same")
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding="same")
        for conv in (self.conv1, self.conv2):
            nn.init.xavier_normal_(conv.weight)
            nn.init.zeros_(conv.bias)
        self.norm1 = ChannelLayerNorm(out_channels)
        self.norm2 = ChannelLayerNorm(out_channels)
        self.pool = pool

    def forward(self, x):
        branch1 = F.gelu(self.norm1(self.conv1(x)), approximate="tanh")
        branch2 = F.gelu(self.norm2(self.conv2(x)), approximate="tanh")
        return self.pool(branch1 * branch2)


class QNetwork(nn.Module):
    """PQN's Q-network with the Nature-CNN trunk replaced by the Hadamax encoder."""

    def __init__(self, env):
        super().__init__()
        self.encoder = nn.Sequential(
            HadamaxBlock(4, 32, 8, nn.MaxPool2d(4, stride=4)),  # 84 -> 21
            # ceil_mode reproduces Flax's SAME max-pool padding for odd inputs.
            HadamaxBlock(32, 64, 4, nn.MaxPool2d(2, stride=2, ceil_mode=True)),  # 21 -> 11
            HadamaxBlock(64, 64, 3, nn.MaxPool2d(3, stride=1, padding=1)),  # 11 -> 11
            nn.Flatten(),
        )
        self.projection = nn.Linear(64 * 11 * 11, 512)
        nn.init.kaiming_normal_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)
        self.projection_norm = nn.LayerNorm(512, eps=FLAX_LAYER_NORM_EPS)
        self.head = nn.Linear(512, env.single_action_space.n)

    def forward(self, x):
        hidden = self.encoder(x / 255.0)
        hidden = F.gelu(self.projection_norm(self.projection(hidden)), approximate="tanh")
        return self.head(hidden)


def linear_schedule(start_e: float, end_e: float, duration: float, t: int) -> float:
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


if __name__ == "__main__":
    process_start = time.perf_counter()
    args = tyro.cli(Args)
    if args.num_envs < 1:
        raise ValueError("num_envs must be positive")
    if args.num_steps < 1:
        raise ValueError("num_steps must be positive")
    if args.num_minibatches < 1:
        raise ValueError("num_minibatches must be positive")
    if args.update_epochs < 1:
        raise ValueError("update_epochs must be positive")
    if args.exploration_fraction <= 0:
        raise ValueError("exploration_fraction must be positive")
    if args.benchmark_warmup_iterations < 0:
        raise ValueError("benchmark_warmup_iterations cannot be negative")
    if args.benchmark_measure_iterations < 1:
        raise ValueError("benchmark_measure_iterations must be positive")
    if args.learning_curve_path and args.evaluation_interval < 1:
        raise ValueError("evaluation_interval must be positive when learning_curve_path is set")
    args.batch_size = args.num_envs * args.num_steps
    if args.batch_size % args.num_minibatches:
        raise ValueError("num_envs * num_steps must be divisible by num_minibatches")
    args.minibatch_size = args.batch_size // args.num_minibatches
    args.num_iterations = args.total_timesteps // args.batch_size
    if args.benchmark:
        args.num_iterations = args.benchmark_warmup_iterations + args.benchmark_measure_iterations

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
    else:
        raise ValueError(f"unsupported environment backend: {args.env_backend}")
    if not hasattr(envs.single_action_space, "n"):
        raise ValueError("only discrete action spaces are supported")

    q_network = QNetwork(envs).to(device)
    # Detached parameter view for rollout inference; shares storage with the
    # training parameters, so optimizer steps are visible without copies.
    q_network_inference = QNetwork(envs).to(device)
    inference_params = from_module(q_network).detach()
    inference_params.to_module(q_network_inference, preserve_module_state=True)
    # A tensor learning rate lets annealing update the compiled or captured
    # optimizer step in place instead of forcing a recompile per iteration.
    optimizer = optim.RAdam(
        q_network.parameters(),
        lr=torch.tensor(args.learning_rate, device=device),
        capturable=args.cudagraphs and not args.compile,
    )
    n_actions = int(envs.single_action_space.n)
    observation_shape = envs.single_observation_space.shape
    action_shape = envs.single_action_space.shape

    # Keep the rollout storage compact.  The model normalizes uint8 frames in
    # its forward pass, and assigning CuLE's reused observations copies them.
    obs = torch.zeros((args.num_steps, args.num_envs) + observation_shape, device=device, dtype=torch.uint8)
    actions = torch.zeros((args.num_steps, args.num_envs) + action_shape, device=device, dtype=torch.long)
    rewards = torch.zeros((args.num_steps, args.num_envs), device=device)
    dones = torch.zeros((args.num_steps, args.num_envs), device=device)
    values = torch.zeros((args.num_steps, args.num_envs), device=device)
    avg_returns = deque(maxlen=20)

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
            return q_network(states).argmax(dim=1)

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
            "algorithm": "hadamax_pqn",
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

    def policy(observations: torch.Tensor, epsilon: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        q_values = q_network_inference(observations)
        max_actions = torch.argmax(q_values, dim=1)
        selected_values = q_values.gather(1, max_actions.unsqueeze(1)).squeeze(1)
        random_actions = torch.randint(n_actions, max_actions.shape, device=max_actions.device)
        explore = torch.rand(max_actions.shape, device=max_actions.device) < epsilon
        actions = torch.where(explore, random_actions, max_actions)
        return actions, selected_values

    def q_lambda_targets(
        rollout_rewards: torch.Tensor,
        rollout_dones: torch.Tensor,
        rollout_values: torch.Tensor,
        next_observations: torch.Tensor,
        next_done: torch.Tensor,
    ) -> torch.Tensor:
        returns = torch.zeros_like(rollout_rewards)
        next_value = q_network(next_observations).amax(dim=-1)
        next_nonterminal = 1.0 - next_done
        for step in range(args.num_steps - 1, -1, -1):
            if step == args.num_steps - 1:
                returns[step] = rollout_rewards[step] + args.gamma * next_value * next_nonterminal
            else:
                next_nonterminal = 1.0 - rollout_dones[step + 1]
                returns[step] = rollout_rewards[step] + args.gamma * (
                    args.q_lambda * returns[step + 1] + (1.0 - args.q_lambda) * rollout_values[step + 1]
                ) * next_nonterminal
        return returns

    def update(
        minibatch_obs: torch.Tensor,
        minibatch_actions: torch.Tensor,
        minibatch_returns: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        optimizer.zero_grad()
        old_values = q_network(minibatch_obs).gather(1, minibatch_actions.unsqueeze(-1).long()).squeeze(-1)
        loss = F.mse_loss(minibatch_returns, old_values)
        loss.backward()
        nn.utils.clip_grad_norm_(q_network.parameters(), args.max_grad_norm)
        optimizer.step()
        return loss.detach(), old_values.detach().mean()

    if args.compile:
        # mode=None avoids implicit CUDA graphs retaining CuLE's mutable
        # observation storage or rollout values across calls.
        policy = torch.compile(policy, mode=None, fullgraph=True)
        q_lambda_targets = torch.compile(q_lambda_targets, mode=None, fullgraph=True)
        update = torch.compile(update, mode=None)

    if args.cudagraphs:
        # CudaGraphModule copies inputs into static buffers, so CuLE's reused
        # observation tensor is safe to pass, and clones outputs before
        # returning them. The once-per-iteration target computation stays
        # uncaptured.
        policy = CudaGraphModule(policy, warmup=20)
        update = CudaGraphModule(update, warmup=20)

    global_step = 0
    learner_updates = 0
    start_time = time.perf_counter()
    benchmark_start = None
    benchmark_start_step = None
    benchmark_start_updates = None
    reset_result = envs.reset(seed=args.seed) if args.env_backend == "cule" else envs.reset()
    reset_obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result
    next_obs = to_tensor(reset_obs, device, torch.uint8)
    next_done = torch.zeros(args.num_envs, device=device)
    epsilon_tensor = torch.zeros((), device=device)
    last_loss = None
    last_q_value = None
    evaluation_seconds_total = 0.0
    if curve_writer is not None and not args.benchmark and not args.skip_initial_evaluation:
        evaluate_and_log(0, 0.0)
    learning_wall_start = time.perf_counter()
    next_evaluation_step = args.evaluation_interval
    progress_interval = max(args.batch_size, args.total_timesteps // 100)
    next_progress_step = progress_interval

    iterations = tqdm(
        range(1, args.num_iterations + 1),
        desc=f"Hadamax-PQN {args.env_backend}",
        unit="update",
        disable=args.benchmark,
    )
    for iteration in iterations:
        if args.benchmark and iteration == args.benchmark_warmup_iterations + 1:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            benchmark_start = time.perf_counter()
            benchmark_start_step = global_step
            benchmark_start_updates = learner_updates

        if args.anneal_lr:
            fraction = 1.0 - (iteration - 1.0) / max(args.num_iterations, 1)
            optimizer.param_groups[0]["lr"].copy_(fraction * args.learning_rate)

        for step in range(args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done
            epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step)
            epsilon_tensor.fill_(epsilon)

            if args.compile:
                torch.compiler.cudagraph_mark_step_begin()
            with torch.no_grad():
                action, selected_values = policy(next_obs, epsilon_tensor)
            actions[step] = action
            values[step] = selected_values

            step_result = step_env(envs, action)
            logging_dones = None
            if len(step_result) == 5:
                next_obs_raw, reward, terminations, truncations, info = step_result
                if not isinstance(terminations, torch.Tensor):
                    logging_dones = np.logical_or(terminations, truncations)
                next_done = done_tensor(terminations, truncations, device)
            else:
                next_obs_raw, reward, next_done_raw, info = step_result
                logging_dones = np.asarray(next_done_raw, dtype=bool)
                next_done = to_tensor(next_done_raw, device, torch.float32)
            rewards[step] = to_tensor(reward, device, torch.float32).view(-1)
            next_obs = to_tensor(next_obs_raw, device, torch.uint8)

            if not args.benchmark:
                if "final_info" in info:
                    for final_info in info["final_info"]:
                        if final_info and "episode" in final_info:
                            episode_return = float(final_info["episode"]["r"])
                            print(f"global_step={global_step}, episodic_return={episode_return}")
                            avg_returns.append(episode_return)
                            writer.add_scalar("charts/avg_episodic_return", np.average(avg_returns), global_step)
                            writer.add_scalar("charts/episodic_return", episode_return, global_step)
                            writer.add_scalar("charts/episodic_length", final_info["episode"]["l"], global_step)
                elif "r" in info:
                    if logging_dones is None:
                        logging_dones = to_numpy(next_done).astype(bool)
                    game_overs = logging_dones & (to_numpy(info["lives"]) == 0)
                    for index in np.flatnonzero(game_overs):
                        episode_return = float(to_numpy(info["r"])[index])
                        print(f"global_step={global_step}, episodic_return={episode_return}")
                        avg_returns.append(episode_return)
                        writer.add_scalar("charts/avg_episodic_return", np.average(avg_returns), global_step)
                        writer.add_scalar("charts/episodic_return", episode_return, global_step)
                        writer.add_scalar("charts/episodic_length", to_numpy(info["l"])[index], global_step)

        if args.compile:
            torch.compiler.cudagraph_mark_step_begin()
        with torch.no_grad():
            returns = q_lambda_targets(rewards, dones, values, next_obs, next_done)
        flat_obs = obs.reshape((-1,) + observation_shape)
        flat_actions = actions.reshape(-1)
        flat_returns = returns.reshape(-1)

        for _ in range(args.update_epochs):
            for minibatch_indices in torch.randperm(args.batch_size, device=device).split(args.minibatch_size):
                if args.compile:
                    torch.compiler.cudagraph_mark_step_begin()
                last_loss, last_q_value = update(
                    flat_obs[minibatch_indices],
                    flat_actions[minibatch_indices],
                    flat_returns[minibatch_indices],
                )
                learner_updates += 1

        if not args.benchmark:
            writer.add_scalar("losses/td_loss", last_loss.item(), global_step)
            writer.add_scalar("losses/q_values", last_q_value.item(), global_step)
            writer.add_scalar("charts/SPS", int(global_step / max(time.perf_counter() - start_time, 1e-9)), global_step)
            iterations.set_postfix(frames=global_step, refresh=False)

        if args.emit_progress and not args.benchmark and global_step >= next_progress_step:
            print(f"TRAINING_PROGRESS {global_step}", flush=True)
            while next_progress_step <= global_step:
                next_progress_step += progress_interval

        if curve_writer is not None and not args.benchmark and global_step >= next_evaluation_step:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            training_seconds = time.perf_counter() - learning_wall_start - evaluation_seconds_total
            evaluation_seconds_total += evaluate_and_log(global_step, training_seconds)
            while next_evaluation_step <= global_step:
                next_evaluation_step += args.evaluation_interval

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
            "algorithm": "hadamax_pqn",
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
            "num_envs": args.num_envs,
            "num_minibatches": args.num_minibatches,
            "num_steps": args.num_steps,
            "peak_cuda_memory_mb": (
                torch.cuda.max_memory_allocated(device) / (1024**2) if device.type == "cuda" else 0.0
            ),
            "process_seconds": benchmark_end - process_start,
            "schema_version": 1,
            "sps": measured_steps / measured_seconds,
            "update_epochs": args.update_epochs,
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

    envs.close()
    if curve_file is not None:
        curve_file.close()
    if writer is not None:
        writer.close()
