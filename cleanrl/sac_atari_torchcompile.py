# torch.compile twin of sac_atari.py, which is adapted from CleanRL's
# cleanrl/sac_atari.py (https://github.com/vwxyzjn/cleanrl, MIT); the networks
# are imported from it directly.  The compile / CUDA-graph structure follows
# LeanRL (https://github.com/meta-pytorch/LeanRL, MIT), whose
# leanrl/sac_continuous_action_torchcompile.py is the continuous-action
# counterpart.  Both licenses are reproduced in cleanrl/LICENSE.md.
# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/sac/#sac_ataripy
"""Discrete soft actor-critic with optional torch.compile and CUDA graphs.

Environment interaction stays eager; the fixed-shape policy and the combined
critic/actor/temperature update can be compiled and captured.  Replay keeps
full stacked transitions on the training device via TorchRL's
``LazyTensorStorage``, matching dqn_torchcompile.py.
"""

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
import tqdm
import tyro
from tensordict import TensorDict, from_module
from tensordict.nn import CudaGraphModule
from torch.distributions.categorical import Categorical, Distribution
from torchrl.data import LazyTensorStorage, ReplayBuffer

Distribution.set_default_validate_args(False)

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
from sac_atari import Actor, SoftQNetwork, make_env

from cleanrl_utils.episode_stats import EpisodeStats


class _NullWriter:
    """Satisfy EpisodeStats' writer interface without TensorBoard."""

    def add_scalar(self, *args, **kwargs):
        return None


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
    """environment backend: `cule`, `envpool`, or `gymnasium`"""
    cule_device: str = "auto"
    """CuLE device; auto uses CUDA for 32+ envs and CPU for smaller batches"""
    track: bool = False
    """if toggled, track the experiment with Weights and Biases"""

    # Algorithm specific arguments
    env_id: str = "BeamRiderNoFrameskip-v4"
    """the id of the environment"""
    total_timesteps: int = 5000000
    """total timesteps of the experiments"""
    num_envs: int = 256
    """the number of parallel game environments"""
    buffer_size: int = 100000
    """the replay memory capacity in individual transitions (kept on the GPU)"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 1.0
    """target smoothing coefficient (default: 1)"""
    batch_size: int = 512
    """the replay sample batch size"""
    learning_starts: int = 20000
    """timestep to start learning"""
    policy_lr: float = 3e-4
    """the learning rate of the policy network optimizer"""
    q_lr: float = 3e-4
    """the learning rate of the Q network network optimizer"""
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
    target_network_frequency: int = 2000
    """learner updates between target-network updates"""
    alpha: float = 0.2
    """Entropy regularization coefficient."""
    autotune: bool = True
    """automatic tuning of the entropy coefficient"""
    target_entropy_scale: float = 0.89
    """coefficient for scaling the autotune entropy target"""

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
            project="sac_atari",
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
    n_act = int(envs.single_action_space.n)

    def step_vector_env(actions):
        step_result = step_env(envs, actions)
        if len(step_result) == 5:
            next_obs, rewards, terminations, truncations, infos = step_result
            dones = done_tensor(terminations, truncations, device).bool()
        else:
            next_obs, rewards, dones_raw, infos = step_result
            truncations = None
            dones = to_tensor(dones_raw, device, torch.bool)
        return to_tensor(next_obs, device), rewards, dones, truncations, infos

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

    actor = Actor(envs).to(device)
    actor_detach = Actor(envs).to(device)
    actor_params = from_module(actor).detach()
    actor_params.to_module(actor_detach, preserve_module_state=True)

    qf1 = SoftQNetwork(envs).to(device)
    qf2 = SoftQNetwork(envs).to(device)
    qf1_params = from_module(qf1).detach()
    qf2_params = from_module(qf2).detach()

    qf1_target = SoftQNetwork(envs).to(device)
    qf1_target_params = qf1_params.clone().lock_()
    qf1_target_params.to_module(qf1_target, preserve_module_state=True)
    qf2_target = SoftQNetwork(envs).to(device)
    qf2_target_params = qf2_params.clone().lock_()
    qf2_target_params.to_module(qf2_target, preserve_module_state=True)

    capturable = args.cudagraphs and not args.compile
    # TRY NOT TO MODIFY: eps=1e-4 increases numerical stability
    q_optimizer = optim.Adam(
        list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr, eps=1e-4, capturable=capturable
    )
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr, eps=1e-4, capturable=capturable)

    # Automatic entropy tuning
    if args.autotune:
        target_entropy = -args.target_entropy_scale * torch.log(1 / torch.tensor(n_act, device=device))
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        a_optimizer = optim.Adam([log_alpha], lr=args.q_lr, eps=1e-4, capturable=capturable)
        alpha_const = None
    else:
        alpha_const = torch.tensor(args.alpha, device=device)

    def policy(obs):
        _, _, action_probs = actor_detach.get_action(obs)
        return torch.multinomial(action_probs, 1).squeeze(-1)

    def update(data):
        alpha = log_alpha.detach().exp() if args.autotune else alpha_const
        observations = data["observations"]
        # CRITIC training
        with torch.no_grad():
            _, next_state_log_pi, next_state_action_probs = actor.get_action(data["next_observations"])
            qf1_next_target = qf1_target(data["next_observations"])
            qf2_next_target = qf2_target(data["next_observations"])
            # we can use the action probabilities instead of MC sampling to estimate the expectation
            min_qf_next_target = next_state_action_probs * (
                torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
            )
            # adapt Q-target for discrete Q-function
            min_qf_next_target = min_qf_next_target.sum(dim=1)
            next_q_value = data["rewards"] + (~data["dones"]).float() * args.gamma * min_qf_next_target

        # use Q-values only for the taken actions
        actions = data["actions"].unsqueeze(-1)
        qf1_a_values = qf1(observations).gather(1, actions).view(-1)
        qf2_a_values = qf2(observations).gather(1, actions).view(-1)
        qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
        qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
        qf_loss = qf1_loss + qf2_loss

        q_optimizer.zero_grad()
        qf_loss.backward()
        q_optimizer.step()

        # ACTOR training
        _, log_pi, action_probs = actor.get_action(observations)
        with torch.no_grad():
            qf1_values = qf1(observations)
            qf2_values = qf2(observations)
            min_qf_values = torch.min(qf1_values, qf2_values)
        # no need for reparameterization, the expectation can be calculated for discrete actions
        actor_loss = (action_probs * ((alpha * log_pi) - min_qf_values)).mean()

        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()

        if args.autotune:
            # reuse action probabilities for temperature loss
            alpha_loss = (action_probs.detach() * (-log_alpha.exp() * (log_pi + target_entropy).detach())).mean()

            a_optimizer.zero_grad()
            alpha_loss.backward()
            a_optimizer.step()
        else:
            alpha_loss = torch.zeros_like(actor_loss)

        return qf_loss.detach(), actor_loss.detach(), alpha_loss.detach(), alpha.detach(), qf1_a_values.detach().mean()

    if args.compile:
        # reduce-overhead implicitly enables Inductor CUDA graphs, which cannot
        # safely retain CuLE's in-place, reused observation storage.
        mode = None
        update = torch.compile(update, mode=mode)
        policy = torch.compile(policy, mode=mode, fullgraph=True)

    if args.cudagraphs:
        update = CudaGraphModule(update, warmup=20)
        policy = CudaGraphModule(policy, warmup=20)

    rb = ReplayBuffer(storage=LazyTensorStorage(args.buffer_size, device=device))

    reset_result = envs.reset(seed=args.seed) if args.env_backend != "envpool" else envs.reset()
    reset_obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result
    obs = to_tensor(reset_obs, device)
    avg_returns = deque(maxlen=20)
    episode_stats = EpisodeStats(args.solve_window, args.solve_reward)
    null_writer = _NullWriter()
    global_step = 0
    learner_updates = 0
    update_budget = 0.0
    next_target_update = args.target_network_frequency
    qf_loss = None
    actor_loss = None
    alpha_loss = None
    alpha_value = None
    q_value = None
    desc = ""
    start_time = time.perf_counter()

    num_vector_steps = math.ceil(args.total_timesteps / args.num_envs)
    if args.benchmark:
        num_vector_steps = args.benchmark_warmup_iterations + args.benchmark_measure_iterations
    pbar = tqdm.tqdm(range(num_vector_steps), desc=f"SAC {args.env_backend}", unit="step", disable=args.benchmark)
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
        if args.max_training_seconds and time.perf_counter() - start_time >= args.max_training_seconds:
            break

        # ALGO LOGIC: put action logic here
        if global_step < args.learning_starts:
            actions = torch.randint(n_act, (args.num_envs,), device=device)
        else:
            torch.compiler.cudagraph_mark_step_begin()
            with torch.no_grad():
                actions = policy(obs)

        # CuLE reuses and mutates its observation tensor on every step.
        transition_obs = obs.clone()
        next_obs, rewards, dones, truncations, infos = step_vector_env(actions)
        transition_next_obs = next_obs.clone()
        if args.env_backend == "gymnasium" and truncations is not None and np.asarray(truncations).any():
            for idx, truncated in enumerate(truncations):
                if truncated:
                    transition_next_obs[idx] = to_tensor(infos["final_observation"][idx], device)

        rb.extend(
            TensorDict(
                {
                    "observations": transition_obs,
                    "next_observations": transition_next_obs,
                    "actions": actions,
                    "rewards": to_tensor(rewards, device, torch.float32).view(-1),
                    "dones": dones,
                },
                batch_size=[args.num_envs],
                device=device,
            )
        )
        obs = next_obs
        global_step += args.num_envs

        solved = False
        if not args.benchmark:
            episode_infos = completed_episode_infos(infos, dones)
            for info in episode_infos.get("final_info", ()):
                if info and "episode" in info:
                    avg_returns.append(float(info["episode"]["r"]))
            solved = episode_stats.update(episode_infos, global_step, null_writer)
            if avg_returns:
                desc = f", episodic_return={sum(avg_returns) / len(avg_returns):.2f}"

        # ALGO LOGIC: training.
        if global_step > args.learning_starts and len(rb) >= args.batch_size:
            update_budget += args.learner_updates_per_vector_step
            num_updates = int(update_budget)
            update_budget -= num_updates
            for _ in range(num_updates):
                torch.compiler.cudagraph_mark_step_begin()
                qf_loss, actor_loss, alpha_loss, alpha_value, q_value = update(rb.sample(args.batch_size))
                learner_updates += 1

                # update the target networks
                if learner_updates >= next_target_update:
                    qf1_target_params.lerp_(qf1_params, args.tau)
                    qf2_target_params.lerp_(qf2_params, args.tau)
                    next_target_update = (
                        learner_updates // args.target_network_frequency + 1
                    ) * args.target_network_frequency

        if not args.benchmark and global_step % max(args.num_envs * 100, 1) == 0:
            speed = global_step / max(time.perf_counter() - start_time, 1e-9)
            pbar.set_description(f"speed: {speed:4.1f} sps{desc}")
            if args.track and qf_loss is not None:
                wandb.log(
                    {
                        "speed": speed,
                        "episode_return": np.mean(avg_returns) if avg_returns else np.nan,
                        "qf_loss": qf_loss,
                        "actor_loss": actor_loss,
                        "alpha_loss": alpha_loss,
                        "alpha": alpha_value,
                        "q_value": q_value,
                    },
                    step=global_step,
                )
        if solved:
            break

    if args.benchmark:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        benchmark_end = time.perf_counter()
        measured_steps = global_step - benchmark_start_step
        measured_updates = learner_updates - benchmark_start_updates
        measured_seconds = benchmark_end - benchmark_start
        result = {
            "algorithm": "sac",
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
        print(
            "replay ratio:",
            learner_updates * args.batch_size / max(global_step - args.learning_starts, 1),
        )
        episode_stats.print_summary()

    envs.close()
