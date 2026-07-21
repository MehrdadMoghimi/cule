# torch.compile twin of ppo_atari_lstm.py, which is adapted from CleanRL's
# cleanrl/ppo_atari_lstm.py (https://github.com/vwxyzjn/cleanrl, MIT); the
# recurrent Agent is imported from it directly.  The compile / CUDA-graph
# structure follows LeanRL (https://github.com/meta-pytorch/LeanRL, MIT).
# Both licenses are reproduced in cleanrl/LICENSE.md.
# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_atari_lstmpy
"""Recurrent PPO with optional torch.compile and CUDA graphs.

Environment interaction stays eager; the single-step recurrent policy, GAE,
and the sequence-minibatch learner update can be compiled and captured.  The
LSTM consumes one 84x84 frame per step (no frame stacking), so the CuLE and
EnvPool backends are configured with a stack of one.
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
import torch.optim as optim
import tqdm
import tyro
from tensordict import from_module
from tensordict.nn import CudaGraphModule
from torch.distributions.categorical import Distribution

Distribution.set_default_validate_args(False)

torch.set_float32_matmul_precision("high")

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from cule_env import (
    done_tensor,
    make_cule_env,
    resolve_cule_device,
    step_env,
    to_numpy,
    to_tensor,
)
from ppo_atari_lstm import Agent, make_env


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
    env_id: str = "BreakoutNoFrameskip-v4"
    """the id of the environment"""
    total_timesteps: int = 10000000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 256
    """the number of parallel game environments"""
    num_steps: int = 32
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 4
    """the number of mini-batches (over environments, keeping sequences intact)"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.1
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.01
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float | None = None
    """the target KL divergence threshold"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""

    compile: bool = False
    """whether to use torch.compile"""
    cudagraphs: bool = False
    """whether to use CUDA graphs on top of compile"""
    benchmark: bool = False
    """run a fixed warmup/measurement window and print a JSON benchmark result"""
    benchmark_warmup_iterations: int = 3
    """full training iterations excluded from benchmark timing"""
    benchmark_measure_iterations: int = 10
    """full training iterations included in benchmark timing"""


if __name__ == "__main__":
    process_start = time.perf_counter()
    args = tyro.cli(Args)
    if args.num_envs < 1:
        raise ValueError("num_envs must be positive")
    if args.num_steps < 1:
        raise ValueError("num_steps must be positive")
    if args.num_minibatches < 1:
        raise ValueError("num_minibatches must be positive")
    if args.num_envs % args.num_minibatches != 0:
        raise ValueError("num_envs must be divisible by num_minibatches (sequence minibatching)")
    if args.benchmark_warmup_iterations < 0:
        raise ValueError("benchmark_warmup_iterations cannot be negative")
    if args.benchmark_measure_iterations < 1:
        raise ValueError("benchmark_measure_iterations must be positive")
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    if args.benchmark:
        args.num_iterations = args.benchmark_warmup_iterations + args.benchmark_measure_iterations

    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{args.compile}__{args.cudagraphs}"
    if args.track:
        import wandb

        wandb.init(
            project="ppo_atari_lstm",
            name=f"{os.path.splitext(os.path.basename(__file__))[0]}-{run_name}",
            config=vars(args),
            save_code=True,
        )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    if args.cudagraphs and device.type != "cuda":
        raise ValueError("cudagraphs requires CUDA")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    # env setup
    if args.env_backend == "cule":
        env_device = resolve_cule_device(args.cule_device, device, args.num_envs)
        envs = make_cule_env(
            args.env_id, args.num_envs, env_device, args.seed, args.capture_video, frame_stack=1
        )
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
                stack_num=1,
                seed=args.seed,
            )
        )
    elif args.env_backend == "gymnasium":
        envs = gym.vector.SyncVectorEnv(
            [make_env(args.env_id, i, args.capture_video, run_name) for i in range(args.num_envs)],
        )
    else:
        raise ValueError(f"unsupported environment backend: {args.env_backend}")
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

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

    agent = Agent(envs).to(device)
    agent_inference = Agent(envs).to(device)
    agent_inference_p = from_module(agent).detach()
    agent_inference_p.to_module(agent_inference, preserve_module_state=True)

    optimizer = optim.Adam(
        agent.parameters(),
        lr=torch.tensor(args.learning_rate, device=device),
        eps=1e-5,
        capturable=args.cudagraphs and not args.compile,
    )

    def policy(obs, done, lstm_h, lstm_c):
        action, logprob, _, value, (next_h, next_c) = agent_inference.get_action_and_value(
            obs, (lstm_h, lstm_c), done
        )
        return action, logprob, value.flatten(), next_h, next_c

    def gae(next_obs, next_done, next_h, next_c, dones, values, rewards):
        next_value = agent_inference.get_value(next_obs, (next_h, next_c), next_done).reshape(1, -1)
        advantages = torch.zeros_like(rewards)
        lastgaelam = torch.zeros(args.num_envs, device=rewards.device)
        for t in range(args.num_steps - 1, -1, -1):
            if t == args.num_steps - 1:
                nextnonterminal = 1.0 - next_done
                nextvalues = next_value
            else:
                nextnonterminal = 1.0 - dones[t + 1]
                nextvalues = values[t + 1]
            delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
            lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            advantages[t] = lastgaelam
        returns = advantages + values
        return advantages, returns

    def update(mb_obs, mb_actions, mb_logprobs, mb_advantages, mb_returns, mb_values, mb_dones, lstm_h, lstm_c):
        optimizer.zero_grad()
        _, newlogprob, entropy, newvalue, _ = agent.get_action_and_value(
            mb_obs, (lstm_h, lstm_c), mb_dones, mb_actions.long()
        )
        logratio = newlogprob - mb_logprobs
        ratio = logratio.exp()

        with torch.no_grad():
            # calculate approx_kl http://joschu.net/blog/kl-approx.html
            old_approx_kl = (-logratio).mean()
            approx_kl = ((ratio - 1) - logratio).mean()
            clipfrac = ((ratio - 1.0).abs() > args.clip_coef).float().mean()

        if args.norm_adv:
            mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

        # Policy loss
        pg_loss1 = -mb_advantages * ratio
        pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
        pg_loss = torch.max(pg_loss1, pg_loss2).mean()

        # Value loss
        newvalue = newvalue.view(-1)
        if args.clip_vloss:
            v_loss_unclipped = (newvalue - mb_returns) ** 2
            v_clipped = mb_values + torch.clamp(
                newvalue - mb_values,
                -args.clip_coef,
                args.clip_coef,
            )
            v_loss_clipped = (v_clipped - mb_returns) ** 2
            v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
            v_loss = 0.5 * v_loss_max.mean()
        else:
            v_loss = 0.5 * ((newvalue - mb_returns) ** 2).mean()

        entropy_loss = entropy.mean()
        loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

        loss.backward()
        gn = nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
        optimizer.step()

        return (
            approx_kl,
            v_loss.detach(),
            pg_loss.detach(),
            entropy_loss.detach(),
            old_approx_kl,
            clipfrac,
            gn,
        )

    if args.compile:
        # Dynamo rejects nn.LSTM unless experimental RNN support is enabled.
        torch._dynamo.config.allow_rnn = True
        # reduce-overhead implicitly enables Inductor CUDA graphs, which cannot
        # safely retain CuLE's in-place, reused observation storage.
        mode = None
        policy = torch.compile(policy, mode=mode)
        gae = torch.compile(gae, mode=mode, fullgraph=True)
        update = torch.compile(update, mode=mode)

    if args.cudagraphs:
        policy = CudaGraphModule(policy, warmup=20)
        update = CudaGraphModule(update, warmup=20)

    # ALGO Logic: Storage setup
    obs = torch.zeros(
        (args.num_steps, args.num_envs) + envs.single_observation_space.shape, device=device
    )
    actions = torch.zeros((args.num_steps, args.num_envs), device=device, dtype=torch.long)
    logprobs = torch.zeros((args.num_steps, args.num_envs), device=device)
    rewards = torch.zeros((args.num_steps, args.num_envs), device=device)
    dones = torch.zeros((args.num_steps, args.num_envs), device=device)
    values = torch.zeros((args.num_steps, args.num_envs), device=device)

    avg_returns = deque(maxlen=20)
    global_step = 0
    start_time = time.perf_counter()
    reset_result = envs.reset(seed=args.seed) if args.env_backend != "envpool" else envs.reset()
    reset_obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result
    next_obs = to_tensor(reset_obs, device)
    next_done = torch.zeros(args.num_envs, device=device)
    next_lstm_state = (
        torch.zeros(agent.lstm.num_layers, args.num_envs, agent.lstm.hidden_size, device=device),
        torch.zeros(agent.lstm.num_layers, args.num_envs, agent.lstm.hidden_size, device=device),
    )  # hidden and cell states (see https://youtu.be/8HyCNIVRbSU)

    envsperbatch = args.num_envs // args.num_minibatches
    out = None
    pbar = tqdm.tqdm(
        range(1, args.num_iterations + 1),
        desc=f"PPO-LSTM {args.env_backend}",
        unit="update",
        disable=args.benchmark,
    )
    benchmark_start = None
    benchmark_start_step = None
    for iteration in pbar:
        if args.benchmark and iteration == args.benchmark_warmup_iterations + 1:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            benchmark_start = time.perf_counter()
            benchmark_start_step = global_step

        initial_lstm_state = (next_lstm_state[0].clone(), next_lstm_state[1].clone())
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"].copy_(lrnow)

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            torch.compiler.cudagraph_mark_step_begin()
            with torch.no_grad():
                action, logprob, value, next_h, next_c = policy(
                    next_obs, next_done, next_lstm_state[0], next_lstm_state[1]
                )
                next_lstm_state = (next_h.clone(), next_c.clone())
                values[step] = value
                actions[step] = action
                logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            step_result = step_env(envs, action)
            if len(step_result) == 5:
                next_obs_raw, reward, terminations, truncations, infos = step_result
                step_dones = done_tensor(terminations, truncations, device)
            else:
                next_obs_raw, reward, dones_raw, infos = step_result
                step_dones = to_tensor(dones_raw, device, torch.float32)
            next_done = step_dones
            rewards[step] = to_tensor(reward, device).view(-1)
            next_obs = to_tensor(next_obs_raw, device)

            if not args.benchmark:
                for info in completed_episode_infos(infos, next_done).get("final_info", ()):
                    if info and "episode" in info:
                        avg_returns.append(float(info["episode"]["r"]))

        # bootstrap value if not done
        torch.compiler.cudagraph_mark_step_begin()
        with torch.no_grad():
            advantages, returns = gae(
                next_obs, next_done, next_lstm_state[0], next_lstm_state[1], dones, values, rewards
            )

        # flatten the batch, keeping each environment's sequence contiguous in time
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape(-1)
        b_dones = dones.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        envinds = np.arange(args.num_envs)
        flatinds = np.arange(args.batch_size).reshape(args.num_steps, args.num_envs)
        for epoch in range(args.update_epochs):
            np.random.shuffle(envinds)
            for start in range(0, args.num_envs, envsperbatch):
                end = start + envsperbatch
                mbenvinds = envinds[start:end]
                mb_inds = flatinds[:, mbenvinds].ravel()  # be really careful about the index

                torch.compiler.cudagraph_mark_step_begin()
                out = update(
                    b_obs[mb_inds],
                    b_actions[mb_inds],
                    b_logprobs[mb_inds],
                    b_advantages[mb_inds],
                    b_returns[mb_inds],
                    b_values[mb_inds],
                    b_dones[mb_inds],
                    initial_lstm_state[0][:, mbenvinds].contiguous(),
                    initial_lstm_state[1][:, mbenvinds].contiguous(),
                )
                if args.target_kl is not None and out[0] > args.target_kl:
                    break
            else:
                continue
            break

        if not args.benchmark and iteration % 10 == 0:
            speed = global_step / max(time.perf_counter() - start_time, 1e-9)
            desc = f", episodic_return={sum(avg_returns) / len(avg_returns):.2f}" if avg_returns else ""
            pbar.set_description(f"speed: {speed:4.1f} sps{desc}")
            if args.track and out is not None:
                approx_kl, v_loss, pg_loss, entropy_loss, old_approx_kl, clipfrac, gn = out
                wandb.log(
                    {
                        "speed": speed,
                        "episode_return": np.mean(avg_returns) if avg_returns else np.nan,
                        "v_loss": v_loss,
                        "pg_loss": pg_loss,
                        "entropy_loss": entropy_loss,
                        "approx_kl": approx_kl,
                        "clipfrac": clipfrac,
                        "gn": gn,
                        "lr": optimizer.param_groups[0]["lr"],
                    },
                    step=global_step,
                )

    if args.benchmark:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        benchmark_end = time.perf_counter()
        measured_steps = global_step - benchmark_start_step
        measured_seconds = benchmark_end - benchmark_start
        result = {
            "algorithm": "ppo_lstm",
            "backend": args.env_backend,
            "batch_size": args.batch_size,
            "benchmark": "full_training_loop",
            "compile": args.compile,
            "cudagraphs": args.cudagraphs,
            "env_device": str(getattr(envs, "device", "cpu")),
            "env_id": args.env_id,
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
            "warmup_iterations": args.benchmark_warmup_iterations,
        }
        print(f"BENCHMARK_RESULT {json.dumps(result, sort_keys=True)}", flush=True)
    else:
        elapsed = time.perf_counter() - start_time
        print("SPS:", int(global_step / max(elapsed, 1e-9)))
        if avg_returns:
            print("recent mean return:", sum(avg_returns) / len(avg_returns))

    envs.close()
