# torch.compile twin of qdagger_dqn_atari_impalacnn.py, which is adapted from
# CleanRL's cleanrl/qdagger_dqn_atari_impalacnn.py
# (https://github.com/vwxyzjn/cleanrl, MIT); the Impala-CNN student and the
# distillation loss are imported from it directly.  QDagger is from
# Reincarnating RL (Agarwal et al., 2022, https://arxiv.org/abs/2206.01626).
# The compile / CUDA-graph structure follows LeanRL
# (https://github.com/meta-pytorch/LeanRL, MIT).  Both licenses are
# reproduced in cleanrl/LICENSE.md.
# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/qdagger/#qdagger_dqn_atari_jax_impalacnnpy
"""QDagger (Impala-CNN student) with optional torch.compile and CUDA graphs.

Environment interaction stays eager; the fixed-shape policy and learner update
(TD loss plus teacher distillation) can be compiled and captured.  Both the
teacher replay buffer and the online replay buffer keep full stacked
transitions on the training device via TorchRL's ``LazyTensorStorage``, so the
default buffer and phase sizes are smaller than the eager trainer's.
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
from dqn_atari import QNetwork as TeacherModel
from qdagger_dqn_atari_impalacnn import (
    QNetwork,
    RecordEpisodeStatistics,
    kl_divergence_with_logits,
    linear_schedule,
    make_env,
)

from cleanrl_utils.episode_stats import EpisodeStats


class _NullWriter:
    """Satisfy EpisodeStats' writer interface without TensorBoard."""

    def add_scalar(self, *args, **kwargs):
        return None


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
    env_id: str = "BreakoutNoFrameskip-v4"
    """the id of the environment"""
    total_timesteps: int = 10000000
    """total online environment transitions after the offline phase"""
    learning_rate: float = 1e-4
    """the learning rate of the optimizer"""
    num_envs: int = 256
    """the number of parallel game environments"""
    buffer_size: int = 100000
    """the replay memory capacity in individual transitions (kept on the GPU)"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 1.0
    """the target network update rate"""
    target_network_frequency: int = 1000
    """learner updates between target-network updates"""
    batch_size: int = 512
    """the replay sample batch size"""
    start_e: float = 1.0
    """the starting epsilon for exploration"""
    end_e: float = 0.01
    """the ending epsilon for exploration"""
    exploration_fraction: float = 0.10
    """the fraction of `total-timesteps` it takes from start-e to go end-e"""
    learning_starts: int = 80000
    """timestep to start learning"""
    learner_updates_per_vector_step: float = 1.0
    """gradient updates accrued per vector environment step; may be fractional"""
    replay_ratio: float | None = 1.0
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
    teacher_steps: int = 100000
    """the number of transitions to run the teacher policy to generate the replay buffer"""
    offline_steps: int = 100000
    """the number of gradient updates on the teacher's replay buffer"""
    temperature: float = 1.0
    """the temperature parameter for qdagger"""

    compile: bool = False
    """whether to use torch.compile"""
    cudagraphs: bool = False
    """whether to use CUDA graphs on top of compile"""
    benchmark: bool = False
    """measure the online loop only and print a JSON benchmark result"""
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
    if args.teacher_policy_hf_repo is None:
        args.teacher_policy_hf_repo = f"cleanrl/{args.env_id}-{args.teacher_model_exp_name}-seed1"

    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{args.compile}__{args.cudagraphs}"
    if args.track:
        import wandb

        wandb.init(
            project="qdagger_atari",
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

    def transition_tensordict(transition_obs, next_obs, actions, rewards, dones, truncations, infos):
        transition_next_obs = next_obs.clone()
        if args.env_backend == "gymnasium" and truncations is not None and np.asarray(truncations).any():
            for idx, truncated in enumerate(truncations):
                if truncated:
                    transition_next_obs[idx] = to_tensor(infos["final_observation"][idx], device)
        return TensorDict(
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

    q_network = QNetwork(envs).to(device)
    q_network_detach = QNetwork(envs).to(device)
    params_vals = from_module(q_network).detach()
    params_vals.to_module(q_network_detach, preserve_module_state=True)

    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate, capturable=args.cudagraphs and not args.compile)

    target_network = QNetwork(envs).to(device)
    target_params = params_vals.clone().lock_()
    target_params.to_module(target_network, preserve_module_state=True)

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

    distill_coeff_tensor = torch.ones((), device=device)

    def policy(obs, epsilon):
        q_values = q_network_detach(obs)
        greedy_actions = torch.argmax(q_values, dim=1)
        random_actions = torch.randint(n_act, greedy_actions.shape, device=greedy_actions.device)
        explore = torch.rand(greedy_actions.shape, device=greedy_actions.device) < epsilon
        return torch.where(explore, random_actions, greedy_actions)

    def teacher_policy(obs, epsilon):
        with torch.no_grad():
            q_values = teacher_model(obs)
        greedy_actions = torch.argmax(q_values, dim=1)
        random_actions = torch.randint(n_act, greedy_actions.shape, device=greedy_actions.device)
        explore = torch.rand(greedy_actions.shape, device=greedy_actions.device) < epsilon
        return torch.where(explore, random_actions, greedy_actions)

    def update(data):
        with torch.no_grad():
            target_max = target_network(data["next_observations"]).max(dim=1).values
            td_target = data["rewards"] + args.gamma * target_max * (~data["dones"]).float()
            teacher_q_values = teacher_model(data["observations"]) / args.temperature
        student_q_values = q_network(data["observations"])
        old_val = student_q_values.gather(1, data["actions"].unsqueeze(-1)).squeeze(-1)
        q_loss = F.mse_loss(td_target, old_val)

        distill_loss = kl_divergence_with_logits(teacher_q_values, student_q_values / args.temperature).mean()
        loss = q_loss + distill_coeff_tensor * distill_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return q_loss.detach(), distill_loss.detach(), old_val.detach().mean()

    if args.compile:
        # reduce-overhead implicitly enables Inductor CUDA graphs, which cannot
        # safely retain CuLE's in-place, reused observation storage.
        mode = None
        update = torch.compile(update, mode=mode)
        policy = torch.compile(policy, mode=mode, fullgraph=True)
        teacher_policy = torch.compile(teacher_policy, mode=mode, fullgraph=True)

    if args.cudagraphs:
        # distill_coeff_tensor is read from a stable address inside the capture
        # and updated in place outside it, like the target-network parameters.
        update = CudaGraphModule(update, warmup=20)
        policy = CudaGraphModule(policy, warmup=20)
        teacher_policy = CudaGraphModule(teacher_policy, warmup=20)

    epsilon_tensor = torch.zeros((), device=device)

    def run_teacher_episodes(num_episodes):
        """Collect full-game episodic returns for the teacher policy."""
        obs = reset_envs()
        epsilon_tensor.fill_(args.end_e)
        returns: list[float] = []
        max_vector_steps = max(1, math.ceil(num_episodes * 27000 / args.num_envs))
        for _ in range(max_vector_steps):
            torch.compiler.cudagraph_mark_step_begin()
            with torch.no_grad():
                actions = teacher_policy(obs, epsilon_tensor)
            obs, _, dones, _, infos = step_vector_env(actions)
            for info in completed_episode_infos(infos, dones).get("final_info", ()):
                if info and "episode" in info:
                    returns.append(float(info["episode"]["r"]))
            if len(returns) >= num_episodes:
                break
        return returns

    # evaluate the teacher model
    teacher_mean_return = None
    if not args.benchmark:
        teacher_episodic_returns = run_teacher_episodes(args.teacher_eval_episodes)
        if teacher_episodic_returns:
            teacher_mean_return = float(np.mean(teacher_episodic_returns))
            print(f"teacher avg_episodic_return={teacher_mean_return} over {len(teacher_episodic_returns)} episodes")
        else:
            print("teacher evaluation completed no episodes; distill_coeff stays at 1.0")

    learner_updates = 0
    if not args.benchmark and args.offline_steps > 0:
        # collect teacher data; we assume no access to the teacher's own buffer
        teacher_rb = ReplayBuffer(
            storage=LazyTensorStorage(min(args.buffer_size, args.teacher_steps), device=device)
        )
        obs = reset_envs()
        teacher_vector_steps = math.ceil(args.teacher_steps / args.num_envs)
        for teacher_step in tqdm.tqdm(range(teacher_vector_steps), desc="teacher replay fill", unit="step"):
            epsilon_tensor.fill_(linear_schedule(args.start_e, args.end_e, teacher_vector_steps, teacher_step))
            torch.compiler.cudagraph_mark_step_begin()
            with torch.no_grad():
                actions = teacher_policy(obs, epsilon_tensor)
            transition_obs = obs.clone()
            next_obs, rewards, dones, truncations, infos = step_vector_env(actions)
            teacher_rb.extend(
                transition_tensordict(transition_obs, next_obs, actions, rewards, dones, truncations, infos)
            )
            obs = next_obs

        # offline training phase: train the student model using the qdagger loss
        distill_coeff_tensor.fill_(1.0)
        for _ in tqdm.tqdm(range(args.offline_steps), desc="offline student training", unit="update"):
            torch.compiler.cudagraph_mark_step_begin()
            q_loss, distill_loss, q_value = update(teacher_rb.sample(args.batch_size))
            learner_updates += 1
            if learner_updates % args.target_network_frequency == 0:
                target_params.lerp_(params_vals, args.tau)
        del teacher_rb
        if device.type == "cuda":
            torch.cuda.empty_cache()

    rb = ReplayBuffer(storage=LazyTensorStorage(args.buffer_size, device=device))

    obs = reset_envs()
    episodic_returns = deque(maxlen=10)
    episode_stats = EpisodeStats(args.solve_window, args.solve_reward)
    null_writer = _NullWriter()
    global_step = args.offline_steps if not args.benchmark else 0
    online_steps = 0
    update_budget = 0.0
    next_target_update = learner_updates + args.target_network_frequency
    q_loss = None
    distill_loss = None
    q_value = None
    start_time = time.perf_counter()
    num_vector_steps = math.ceil(args.total_timesteps / args.num_envs)
    if args.benchmark:
        num_vector_steps = args.benchmark_warmup_iterations + args.benchmark_measure_iterations
    benchmark_start = None
    benchmark_start_step = None
    benchmark_start_updates = None
    desc = ""
    pbar = tqdm.tqdm(range(num_vector_steps), desc="online student training", unit="step", disable=args.benchmark)
    # online training phase
    for vector_step in pbar:
        if args.benchmark and vector_step == args.benchmark_warmup_iterations:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            benchmark_start = time.perf_counter()
            benchmark_start_step = online_steps
            benchmark_start_updates = learner_updates
        if args.max_training_seconds and time.perf_counter() - start_time >= args.max_training_seconds:
            break
        epsilon_tensor.fill_(
            linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step)
        )
        torch.compiler.cudagraph_mark_step_begin()
        with torch.no_grad():
            actions = policy(obs, epsilon_tensor)

        # CuLE reuses and mutates its observation tensor on every step.
        transition_obs = obs.clone()
        next_obs, rewards, dones, truncations, infos = step_vector_env(actions)
        rb.extend(transition_tensordict(transition_obs, next_obs, actions, rewards, dones, truncations, infos))
        obs = next_obs
        global_step += args.num_envs
        online_steps += args.num_envs

        solved = False
        if not args.benchmark:
            episode_infos = completed_episode_infos(infos, dones)
            for info in episode_infos.get("final_info", ()):
                if info and "episode" in info:
                    episodic_returns.append(float(info["episode"]["r"]))
            solved = episode_stats.update(episode_infos, global_step, null_writer)
            if episodic_returns:
                desc = f", episodic_return={sum(episodic_returns) / len(episodic_returns):.2f}"

        if global_step > args.learning_starts and len(rb) >= args.batch_size:
            update_budget += args.learner_updates_per_vector_step
            num_updates = int(update_budget)
            update_budget -= num_updates
            if num_updates:
                if len(episodic_returns) < 10 or teacher_mean_return is None:
                    distill_coeff = 1.0
                else:
                    teacher_return = teacher_mean_return if teacher_mean_return != 0 else 1e-8
                    distill_coeff = max(1 - np.mean(episodic_returns) / teacher_return, 0)
                distill_coeff_tensor.fill_(distill_coeff)
            for _ in range(num_updates):
                torch.compiler.cudagraph_mark_step_begin()
                q_loss, distill_loss, q_value = update(rb.sample(args.batch_size))
                learner_updates += 1

                if learner_updates >= next_target_update:
                    target_params.lerp_(params_vals, args.tau)
                    next_target_update = (
                        learner_updates // args.target_network_frequency + 1
                    ) * args.target_network_frequency

        if not args.benchmark and online_steps % max(args.num_envs * 100, 1) == 0:
            speed = online_steps / max(time.perf_counter() - start_time, 1e-9)
            pbar.set_description(f"speed: {speed:4.1f} sps{desc}")
            if args.track and q_loss is not None:
                wandb.log(
                    {
                        "speed": speed,
                        "episode_return": np.mean(episodic_returns) if episodic_returns else np.nan,
                        "q_loss": q_loss,
                        "distill_loss": distill_loss,
                        "q_value": q_value,
                        "distill_coeff": distill_coeff_tensor.item(),
                        "epsilon": epsilon_tensor.item(),
                    },
                    step=global_step,
                )
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
        print("SPS:", int(online_steps / max(elapsed, 1e-9)))
        print("learner updates:", learner_updates)
        episode_stats.print_summary()

    envs.close()
