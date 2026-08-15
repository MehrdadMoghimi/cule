# PQN + SEM (Simplicial Embeddings) for Atari.
#
# Obando-Ceron, Mayor, Lavoie, Fujimoto, Courville and Castro, 2026,
# "Simplicial Embeddings Improve Sample Efficiency in Actor-Critic Agents"
# (https://arxiv.org/abs/2510.13704). Ported from the authors' release,
# waltermayor/FastTD3_SEm (`fast_td3/fast_td3.py`, the `SimNorm` and
# `SimNormLinear` modules).
#
# SEM is a representation constraint, not a learning rule -- the same shape of
# contribution as Hadamax, and deliberately a diff off the same parent so the
# two are directly comparable. Where Hadamax changes the *convolutional* stack,
# SEM changes the single dense layer that ends it:
#
#     PQN:  Linear(3136, 512) -> LayerNorm(512) -> ReLU
#     SEM:  Linear(3136, L*V) -> LayerNorm(L*V) -> group-wise softmax over V
#
# The trunk is split into `L` groups of `V` dimensions and each group is
# softmaxed independently, so the representation lives on a product of `L`
# simplices. Every feature is in [0, 1], each group sums to exactly 1, and the
# activation is sparse and discrete-*like* without any straight-through
# estimator or quantisation -- the gradient is the ordinary softmax Jacobian.
#
# Why it should matter here specifically: the paper's argument is that
# large-scale environment parallelism buys wall-clock but not sample
# efficiency, and that the bottleneck is representation collapse under
# non-stationary targets. Bounded, normalised, group-sparse features cannot
# collapse to a constant direction or blow up in scale. The paper reports gains
# on PPO and PQN across 28 ALE games, which is why PQN is the parent here.
#
# Two things a careless port gets wrong, both pinned by
# `tests/test_sem_equivalence.py`:
#
#   * the softmax runs over the **within-group** axis after a reshape to
#     `(..., L, V)` -- not over the flat 512-vector, and not over the group axis;
#   * the LayerNorm sits **before** the softmax and spans the full `L*V` width,
#     not each group separately.
#
# `--sem-groups 64 --sem-dim 8` keeps the trunk at PQN's 512 units, so the
# parameter count is unchanged from the parent and the comparison is clean. The
# reference's FastTD3 defaults are 8 x 8, i.e. a 64-wide trunk -- far too narrow
# for a pixel encoder, so it is not the default here.
#
# Everything else -- Q(lambda) targets, RAdam, the epsilon schedule, LR
# annealing, every hyperparameter -- is inherited from `pqn_atari_envpool.py`
# unchanged.
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
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from cule_env import done_tensor, make_cule_env, step_env, to_tensor


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
    env_backend: str = "envpool"
    """environment backend: `envpool` or `cule`"""

    # Algorithm specific arguments
    env_id: str = "Breakout-v5"
    """the id of the environment"""
    total_timesteps: int = 10000000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 8
    """the number of parallel game environments"""
    num_steps: int = 128
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    num_minibatches: int = 4
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    max_grad_norm: float = 10.0
    """the maximum norm for the gradient clipping"""
    start_e: float = 1
    """the starting epsilon for exploration"""
    end_e: float = 0.01
    """the ending epsilon for exploration"""
    exploration_fraction: float = 0.10
    """the fraction of `total_timesteps` it takes from start_e to end_e"""
    q_lambda: float = 0.65
    """the lambda for the Q-Learning algorithm"""

    # SEM arguments
    sem_groups: int = 64
    """number of simplices L; `sem_groups * sem_dim` is the trunk width"""
    sem_dim: int = 8
    """dimensions V per simplex, i.e. the width each softmax runs over"""

    benchmark: bool = False
    """run a fixed warmup/measurement window and print a JSON benchmark result"""
    benchmark_warmup_iterations: int = 3
    """full training iterations excluded from benchmark timing"""
    benchmark_measure_iterations: int = 10
    """full training iterations included in benchmark timing"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


class RecordEpisodeStatistics(gym.Wrapper):
    def __init__(self, env, deque_size=100):
        super().__init__(env)
        self.num_envs = getattr(env, "num_envs", 1)
        self.single_action_space = getattr(env, "single_action_space", env.action_space)
        self.single_observation_space = getattr(env, "single_observation_space", env.observation_space)
        self.episode_returns = None
        self.episode_lengths = None

    def reset(self, **kwargs):
        observations = super().reset(**kwargs)
        self.episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        self.lives = np.zeros(self.num_envs, dtype=np.int32)
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


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class SimNorm(nn.Module):
    """Group-wise softmax: `fast_td3/fast_td3.py::SimNorm`.

    Reshape the trailing axis to `(L, V)`, softmax over `V`, flatten back. The
    result lies on a product of `L` simplices, hence "simplicial embedding".
    """

    def __init__(self, seq_len=8, simnorm_dim=8):
        super().__init__()
        self.L = seq_len
        self.dim = simnorm_dim

    def forward(self, x):
        shape = x.shape
        x = x.view(*shape[:-1], self.L, self.dim)
        x = F.softmax(x, dim=-1)
        return x.view(*shape)

    def __repr__(self):
        return f"SimNorm(seq_len={self.L}, simnorm_dim={self.dim})"


class SimNormLinear(nn.Module):
    """`Linear -> LayerNorm -> SimNorm`, the reference's composite block.

    The LayerNorm spans the whole `L * V` output, *not* each group separately:
    normalising per group would erase the between-group scale differences that
    set each softmax's effective temperature.
    """

    def __init__(self, in_features: int, seq_len: int, simnorm_dim: int):
        super().__init__()
        out_features = seq_len * simnorm_dim
        self.linear = nn.Linear(in_features, out_features)
        self.norm = nn.LayerNorm(out_features)
        self.simnorm = SimNorm(seq_len, simnorm_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.simnorm(self.norm(self.linear(x)))


class QNetwork(nn.Module):
    """PQN's encoder with the dense trunk replaced by a simplicial embedding."""

    def __init__(self, env, sem_groups: int = 64, sem_dim: int = 8):
        super().__init__()
        if sem_groups < 1 or sem_dim < 2:
            raise ValueError("sem_groups must be >= 1 and sem_dim >= 2")
        self.sem_groups = sem_groups
        self.sem_dim = sem_dim
        self.encoder = nn.Sequential(
            layer_init(nn.Conv2d(4, 32, 8, stride=4)),
            nn.LayerNorm([32, 20, 20]),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.LayerNorm([64, 9, 9]),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.LayerNorm([64, 7, 7]),
            nn.ReLU(),
            nn.Flatten(),
        )
        # The whole of SEM: this block replaces PQN's
        # `Linear(3136, 512) -> LayerNorm(512) -> ReLU`.
        self.trunk = SimNormLinear(3136, sem_groups, sem_dim)
        self.head = layer_init(nn.Linear(sem_groups * sem_dim, env.single_action_space.n))

    def forward(self, x):
        return self.head(self.trunk(self.encoder(x / 255.0)))


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
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
    if args.benchmark_warmup_iterations < 0:
        raise ValueError("benchmark_warmup_iterations cannot be negative")
    if args.benchmark_measure_iterations < 1:
        raise ValueError("benchmark_measure_iterations must be positive")
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    if args.minibatch_size < 1:
        raise ValueError("num_minibatches cannot exceed num_envs * num_steps")
    args.num_iterations = args.total_timesteps // args.batch_size
    if args.benchmark:
        args.num_iterations = args.benchmark_warmup_iterations + args.benchmark_measure_iterations
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
        envs = make_cule_env(args.env_id, args.num_envs, device, args.seed, args.capture_video)
    elif args.env_backend == "envpool":
        if envpool is None:
            raise ImportError("EnvPool backend requested; install envpool or pass --env-backend cule")
        envs = envpool.make(
            args.env_id,
            env_type="gym",
            num_envs=args.num_envs,
            episodic_life=True,
            reward_clip=True,
            seed=args.seed,
        )
        envs = RecordEpisodeStatistics(envs)
    else:
        raise ValueError(f"unsupported environment backend: {args.env_backend}")
    assert hasattr(envs.single_action_space, "n"), "only discrete action space is supported"

    q_network = QNetwork(envs, args.sem_groups, args.sem_dim).to(device)
    optimizer = optim.RAdam(q_network.parameters(), lr=args.learning_rate)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    avg_returns = deque(maxlen=20)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    benchmark_start = None
    benchmark_start_step = None
    reset_result = envs.reset()
    reset_obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result
    next_obs = to_tensor(reset_obs, device)
    next_done = torch.zeros(args.num_envs).to(device)

    for iteration in range(1, args.num_iterations + 1):
        if args.benchmark and iteration == args.benchmark_warmup_iterations + 1:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            benchmark_start = time.perf_counter()
            benchmark_start_step = global_step

        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step)

            random_actions = torch.randint(0, envs.single_action_space.n, (args.num_envs,)).to(device)
            with torch.no_grad():
                q_values = q_network(next_obs)
                max_actions = torch.argmax(q_values, dim=1)
                values[step] = q_values[torch.arange(args.num_envs), max_actions].flatten()

            explore = torch.rand((args.num_envs,)).to(device) < epsilon
            action = torch.where(explore, random_actions, max_actions)
            actions[step] = action

            # TRY NOT TO MODIFY: execute the game and log data.
            step_result = step_env(envs, action)
            logging_dones = None
            if len(step_result) == 5:
                next_obs, reward, terminations, truncations, info = step_result
                if not isinstance(terminations, torch.Tensor):
                    logging_dones = np.logical_or(terminations, truncations)
                next_done = done_tensor(terminations, truncations, device)
            else:
                next_obs, reward, next_done, info = step_result
                logging_dones = np.asarray(next_done, dtype=bool)
                next_done = to_tensor(next_done, device, torch.float32)
            rewards[step] = to_tensor(reward, device).view(-1)
            next_obs = to_tensor(next_obs, device)

            if "final_info" in info:
                for final_info in info["final_info"]:
                    if final_info and "episode" in final_info:
                        episode_return = final_info["episode"]["r"]
                        if not args.benchmark:
                            print(f"global_step={global_step}, episodic_return={episode_return}")
                        avg_returns.append(episode_return)
                        if writer is not None:
                            writer.add_scalar("charts/avg_episodic_return", np.average(avg_returns), global_step)
                            writer.add_scalar("charts/episodic_return", episode_return, global_step)
                            writer.add_scalar("charts/episodic_length", final_info["episode"]["l"], global_step)
            elif "r" in info:
                game_overs = logging_dones & (info["lives"] == 0)
                for idx in np.flatnonzero(game_overs):
                    if not args.benchmark:
                        print(f"global_step={global_step}, episodic_return={info['r'][idx]}")
                    avg_returns.append(info["r"][idx])
                    if writer is not None:
                        writer.add_scalar("charts/avg_episodic_return", np.average(avg_returns), global_step)
                        writer.add_scalar("charts/episodic_return", info["r"][idx], global_step)
                        writer.add_scalar("charts/episodic_length", info["l"][idx], global_step)

        # Compute Q(lambda) targets
        with torch.no_grad():
            returns = torch.zeros_like(rewards).to(device)
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    next_value, _ = torch.max(q_network(next_obs), dim=-1)
                    nextnonterminal = 1.0 - next_done
                    returns[t] = rewards[t] + args.gamma * next_value * nextnonterminal
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    next_value = values[t + 1]
                    returns[t] = (
                        rewards[t]
                        + args.gamma * (args.q_lambda * returns[t + 1] + (1 - args.q_lambda) * next_value) * nextnonterminal
                    )

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_returns = returns.reshape(-1)

        # Optimizing the Q-network
        b_inds = np.arange(args.batch_size)
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                old_val = q_network(b_obs[mb_inds]).gather(1, b_actions[mb_inds].unsqueeze(-1).long()).squeeze()
                loss = F.mse_loss(b_returns[mb_inds], old_val)

                # optimize the model
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(q_network.parameters(), args.max_grad_norm)
                optimizer.step()

        if writer is not None:
            writer.add_scalar("losses/td_loss", loss, global_step)
            writer.add_scalar("losses/q_values", old_val.mean().item(), global_step)
            print("SPS:", int(global_step / (time.time() - start_time)))
            writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    if args.benchmark:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        benchmark_end = time.perf_counter()
        measured_steps = global_step - benchmark_start_step
        measured_seconds = benchmark_end - benchmark_start
        result = {
            "algorithm": "sem_pqn",
            "backend": args.env_backend,
            "batch_size": args.batch_size,
            "benchmark": "full_training_loop",
            "compile": False,
            "env_device": str(getattr(envs, "device", "cpu")),
            "env_id": args.env_id,
            "measure_iterations": args.benchmark_measure_iterations,
            "measured_seconds": measured_seconds,
            "measured_steps": measured_steps,
            "num_envs": args.num_envs,
            "num_minibatches": args.num_minibatches,
            "num_steps": args.num_steps,
            "sem_dim": args.sem_dim,
            "sem_groups": args.sem_groups,
            "peak_cuda_memory_mb": (
                torch.cuda.max_memory_allocated(device) / (1024**2) if device.type == "cuda" else 0.0
            ),
            "process_seconds": benchmark_end - process_start,
            "schema_version": 1,
            "sps": measured_steps / measured_seconds,
            "update_epochs": args.update_epochs,
            "warmup_iterations": args.benchmark_warmup_iterations,
        }
        print(f"BENCHMARK_RESULT {json.dumps(result, sort_keys=True)}", flush=True)

    envs.close()
    if writer is not None:
        writer.close()
