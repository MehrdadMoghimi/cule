# GPG (Group Policy Gradient) for Atari.
#
# Chen, Zhang, Zhong and Antonova, 2025, "Group Policy Gradient"
# (https://arxiv.org/abs/2510.03679). Implemented from the paper — Algorithms 1
# and 2 and Equations 5-6 — because no code had been released at the time of
# writing. The tests in `tests/test_gpg_equivalence.py` are therefore
# transcriptions of the paper's algorithm boxes plus the estimator properties it
# proves, not a diff against upstream source.
#
# GPG deletes the critic. GRPO does this for RLHF by sampling a *group* of
# completions per prompt and using the group's own return statistics as the
# baseline; GPG generalises that to arbitrary MDPs. Gone with the critic: the
# value head, the value loss, `--vf-coef`, the GAE recursion and `--gae-lambda`.
# PPO's clipped surrogate is kept exactly as it was.
#
# ## The estimator (Algorithm 2)
#
# Given a group of `N` trajectories and a **binning function** `f(s, t)`:
#
#   1. compute Monte-Carlo discounted returns `R^n_t` for every step;
#   2. for each trajectory, insert `R^n_t` into bin `B[f(s^n_t, t)]` — but only
#      on the **first visit** that trajectory makes to that bin;
#   3. `A^n_t = R^n_t - mean(B[f(s^n_t, t)])`.
#
# The first-visit rule is load-bearing and easy to drop. Without it a trajectory
# that loiters in one bin contributes many correlated returns and dominates its
# own baseline — which is precisely the term the baseline exists to cancel.
#
# `f` controls the bias/variance trade-off, and the paper is explicit that both
# too coarse and too fine a bin hurt:
#
#   --binning zero       `f(s, t) = 0`, one bin for everything. This is GRPO
#                        with outcome supervision, and — with advantage
#                        normalisation — REINFORCE++.
#   --binning timestep   `f(s, t) = t`, a separate baseline per timestep. The
#                        first-visit rule is vacuous here, since each timestep
#                        occurs exactly once per trajectory.
#
# The paper's other two choices — `f(s, t) = s` for discrete states and
# `eps * Round(s / eps)` for continuous ones — do not transfer to 84x84x4 pixel
# observations, where every state is distinct, every bin holds exactly one
# return, and every advantage is therefore identically zero. That degeneracy is
# pinned by a test rather than left as a surprise.
#
# ## What is honestly different here
#
# The paper's group is `N` *independent trajectories from the same start state*.
# A synchronous vectorised rollout does not provide that: the `num_envs`
# environments sit at different points of different episodes. Two options, both
# available, neither pretending to be the other:
#
#   --group-mode batch  (default) the group is all environments. With
#                       `--binning timestep` the baseline is the
#                       cross-environment mean return at each rollout offset.
#                       The baseline does not depend on the action, so the
#                       gradient stays unbiased (Prop. 1), but the variance
#                       reduction is weaker than the paper's because the states
#                       being averaged over differ.
#   --group-mode env    the group is one environment's own rollout, so the
#                       baseline is that environment's mean return over the
#                       segment. Useful when episodes are long relative to the
#                       rollout.
#
# ## The truncation caveat
#
# `R^n_t` is a Monte-Carlo return, and with no critic there is nothing to
# bootstrap from at the end of a truncated rollout. Returns for steps near the
# end of a rollout that has not terminated are therefore short, and biased
# downwards. `--drop-truncated` excludes those steps from the loss entirely —
# unbiased, at the cost of discarding data. It is off by default, matching the
# paper's fixed-duration-episode setting where the issue does not arise, and
# both branches are tested.
import json
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
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

try:
    import envpool
except ImportError:
    envpool = None

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

from cleanrl_utils.atari_wrappers import (  # isort:skip
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)
from cleanrl_utils.episode_stats import EpisodeStats  # isort:skip


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
    """device for the CuLE backend: `auto`, `cpu`, or a CUDA device string"""

    # Algorithm specific arguments
    env_id: str = "BreakoutNoFrameskip-v4"
    """the id of the environment"""
    total_timesteps: int = 10000000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 32
    """parallel environments; with `--group-mode batch` this is the group size N"""
    num_steps: int = 128
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing"""
    gamma: float = 0.99
    """the discount factor gamma"""
    num_minibatches: int = 4
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.1
    """the surrogate clipping coefficient"""
    ent_coef: float = 0.01
    """coefficient of the entropy"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

    # GPG arguments
    binning: str = "timestep"
    """`timestep` (f(s,t) = t) or `zero` (f(s,t) = 0, GRPO outcome supervision)"""
    group_mode: str = "batch"
    """`batch` (group = all environments) or `env` (group = one environment's rollout)"""
    drop_truncated: bool = False
    """exclude steps whose Monte-Carlo return was cut short by the rollout boundary"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""

    solve_window: int = 100
    """episodes averaged when reporting the running return"""

    benchmark: bool = False
    """run a fixed warmup/measurement window and print a JSON benchmark result"""
    benchmark_warmup_iterations: int = 3
    """full training iterations excluded from benchmark timing"""
    benchmark_measure_iterations: int = 10
    """full training iterations included in benchmark timing"""


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


# ---------------------------------------------------------------------------
# the group advantage estimator (Algorithms 1 and 2)
# ---------------------------------------------------------------------------

def monte_carlo_returns(rewards, next_dones, gamma):
    """`R^n_t`, the discounted return from `t` to the end of the episode.

    No bootstrap — GPG has no critic to bootstrap from. The recursion is cut at
    every episode boundary, so a return never runs across a `done`.

    Also returns `complete`: 1 where the return ran all the way to a genuine
    terminal, 0 where it was cut short by the end of the rollout. Those latter
    returns are systematically too small, which is what `--drop-truncated`
    exists to handle.
    """
    num_steps = rewards.shape[0]
    returns = torch.zeros_like(rewards)
    complete = torch.zeros_like(rewards)
    running = torch.zeros_like(rewards[0])
    running_complete = torch.zeros_like(rewards[0])
    for t in reversed(range(num_steps)):
        not_done = 1.0 - next_dones[t]
        running = rewards[t] + gamma * running * not_done
        # A step is "complete" if it terminated here, or if the step after it
        # was itself complete.
        running_complete = next_dones[t] + not_done * running_complete
        returns[t] = running
        complete[t] = running_complete
    return returns, complete


def bin_indices(binning, num_steps, num_envs, device):
    """The binning function `f(s, t)`, evaluated on a whole rollout.

    Returns a `[T, N]` integer tensor of bin ids.

    `zero`     -> every step in bin 0 (GRPO outcome supervision).
    `timestep` -> bin `t`, one baseline per rollout offset.

    State-based binning (`f(s, t) = s`, or an epsilon-discretisation of a
    continuous state) is not offered: on 84x84x4 pixel observations every state
    is unique, so every bin would hold exactly one return and every advantage
    would be exactly zero.
    """
    if binning == "zero":
        return torch.zeros((num_steps, num_envs), dtype=torch.long, device=device)
    if binning == "timestep":
        return torch.arange(num_steps, device=device).unsqueeze(1).expand(num_steps, num_envs)
    raise ValueError(f"unsupported binning: {binning}")


def first_visit_mask(bins):
    """1 at each trajectory's *first* visit to a bin, 0 on later visits.

    Algorithm 2, line 5: `if f(s^n_t) != f(s^n_i) for i = 1..t-1`. Column `n` of
    `bins` is one trajectory, so this is a per-column running "seen" test.

    Without it, a trajectory that dwells in one bin contributes many correlated
    returns to its own baseline and cancels its own advantage.
    """
    num_steps, num_envs = bins.shape
    mask = torch.zeros_like(bins, dtype=torch.bool)
    for env in range(num_envs):
        seen = set()
        column = bins[:, env].tolist()
        for t in range(num_steps):
            if column[t] not in seen:
                seen.add(column[t])
                mask[t, env] = True
    return mask


def group_advantages(returns, bins, group_mode="batch", eligible=None):
    """`A^n_t = R^n_t - mean(B[f(s^n_t, t)])`, Equation 5.

    `eligible` is the first-visit mask: only those entries contribute to the
    bin means, but *every* step gets an advantage.

    In `batch` mode the bins are pooled across the whole group of environments;
    in `env` mode each environment is its own group, so a bin is scoped to one
    trajectory. A bin with no eligible entries falls back to a baseline of zero,
    which is what an unvisited bin's mean has to be.
    """
    num_steps, num_envs = returns.shape
    if eligible is None:
        eligible = torch.ones_like(returns, dtype=torch.bool)
    weights = eligible.to(returns.dtype)
    num_bins = int(bins.max().item()) + 1

    if group_mode == "batch":
        flat_bins = bins.reshape(-1)
        totals = torch.zeros(num_bins, dtype=returns.dtype, device=returns.device)
        counts = torch.zeros(num_bins, dtype=returns.dtype, device=returns.device)
        totals.scatter_add_(0, flat_bins, (returns * weights).reshape(-1))
        counts.scatter_add_(0, flat_bins, weights.reshape(-1))
        means = totals / counts.clamp(min=1.0)
        baseline = means[flat_bins].reshape(num_steps, num_envs)
    elif group_mode == "env":
        offsets = torch.arange(num_envs, device=returns.device) * num_bins
        flat_bins = (bins + offsets.unsqueeze(0)).reshape(-1)
        size = num_bins * num_envs
        totals = torch.zeros(size, dtype=returns.dtype, device=returns.device)
        counts = torch.zeros(size, dtype=returns.dtype, device=returns.device)
        totals.scatter_add_(0, flat_bins, (returns * weights).reshape(-1))
        counts.scatter_add_(0, flat_bins, weights.reshape(-1))
        means = totals / counts.clamp(min=1.0)
        baseline = means[flat_bins].reshape(num_steps, num_envs)
    else:
        raise ValueError(f"unsupported group_mode: {group_mode}")

    return returns - baseline


def gpg_advantages(rewards, next_dones, gamma, binning="timestep",
                   group_mode="batch", drop_truncated=False):
    """The full Algorithm 2. Returns `(advantages, keep_mask)`."""
    num_steps, num_envs = rewards.shape
    returns, complete = monte_carlo_returns(rewards, next_dones, gamma)
    bins = bin_indices(binning, num_steps, num_envs, rewards.device)
    eligible = first_visit_mask(bins)
    if drop_truncated:
        eligible = eligible & complete.bool()
    advantages = group_advantages(returns, bins, group_mode, eligible)
    keep = complete.bool() if drop_truncated else torch.ones_like(complete, dtype=torch.bool)
    return advantages, keep


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    """PPO's agent with the critic head removed.

    That removal is the point: no value head means no value loss, no target to
    regress, and one fewer network's worth of memory and compute.
    """

    def __init__(self, envs):
        super().__init__()
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(4, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 7 * 7, 512)),
            nn.ReLU(),
        )
        self.actor = layer_init(nn.Linear(512, envs.single_action_space.n), std=0.01)

    def get_action(self, x, action=None):
        logits = self.actor(self.network(x / 255.0))
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy()


if __name__ == "__main__":
    process_start = time.perf_counter()
    args = tyro.cli(Args)
    if args.num_envs < 1 or args.num_steps < 1 or args.num_minibatches < 1:
        raise ValueError("num_envs, num_steps and num_minibatches must be positive")
    if args.binning not in ("zero", "timestep"):
        raise ValueError("binning must be `zero` or `timestep`")
    if args.group_mode not in ("batch", "env"):
        raise ValueError("group_mode must be `batch` or `env`")
    if args.group_mode == "env" and args.binning == "timestep":
        # One bin per (timestep, environment) holds exactly one return, so the
        # baseline equals the return and every advantage is identically zero.
        # Silently training on a zero gradient is worse than refusing to start.
        raise ValueError(
            "`--group-mode env --binning timestep` is degenerate: each bin would "
            "hold a single return and every advantage would be zero. Use "
            "`--group-mode batch` with `--binning timestep`, or "
            "`--group-mode env` with `--binning zero`."
        )
    if args.benchmark_warmup_iterations < 0 or args.benchmark_measure_iterations < 1:
        raise ValueError("invalid benchmark window")
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

        wandb.init(project=args.wandb_project_name, entity=args.wandb_entity, sync_tensorboard=True,
                   config=vars(args), name=run_name, monitor_gym=True, save_code=True)
    writer = None if args.benchmark else SummaryWriter(f"runs/{run_name}")
    if writer is not None:
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )

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
            envpool.make(args.env_id, env_type="gym", num_envs=args.num_envs,
                         episodic_life=True, reward_clip=True, seed=args.seed))
    elif args.env_backend == "gymnasium":
        envs = gym.vector.SyncVectorEnv(
            [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)])
    else:
        raise ValueError(f"unsupported environment backend: {args.env_backend}")
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup. No `values` buffer -- there is no critic.
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs), dtype=torch.long).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    next_dones = torch.zeros((args.num_steps, args.num_envs)).to(device)

    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = to_tensor(next_obs, device)
    next_done = torch.zeros(args.num_envs).to(device)
    stats = EpisodeStats(args.solve_window)
    benchmark_start = None
    benchmark_start_step = None

    for iteration in range(1, args.num_iterations + 1):
        if args.benchmark and iteration == args.benchmark_warmup_iterations + 1:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            benchmark_start = time.perf_counter()
            benchmark_start_step = global_step
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            optimizer.param_groups[0]["lr"] = frac * args.learning_rate

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs

            with torch.no_grad():
                action, logprob, _ = agent.get_action(next_obs)
            actions[step] = action
            logprobs[step] = logprob

            next_obs, reward, terminations, truncations, infos = step_env(envs, action)
            next_done = done_tensor(terminations, truncations, device)
            rewards[step] = to_tensor(reward, device).view(-1)
            next_dones[step] = next_done
            next_obs = to_tensor(next_obs, device)

            stats.update(completed_episode_infos(infos, next_done), global_step, writer)

        # ALGO LOGIC: the whole of GPG. No bootstrap, no GAE, no critic.
        with torch.no_grad():
            advantages, keep = gpg_advantages(
                rewards, next_dones, args.gamma,
                binning=args.binning, group_mode=args.group_mode,
                drop_truncated=args.drop_truncated)

        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_keep = keep.reshape(-1)

        # Steps whose Monte-Carlo return was truncated are dropped here, not
        # zero-weighted, so they do not dilute the minibatch mean.
        usable = torch.nonzero(b_keep, as_tuple=False).squeeze(-1).cpu().numpy()
        if len(usable) < args.num_minibatches:
            continue
        minibatch_size = max(1, len(usable) // args.num_minibatches)

        for epoch in range(args.update_epochs):
            np.random.shuffle(usable)
            for start in range(0, len(usable), minibatch_size):
                mb_inds = usable[start : start + minibatch_size]
                if len(mb_inds) < 2:
                    continue

                _, newlogprob, entropy = agent.get_action(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean()

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                pg_loss = torch.max(
                    -mb_advantages * ratio,
                    -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef),
                ).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        if writer is not None:
            writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
            writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
            writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
            writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
            writer.add_scalar("gpg/advantage_std", advantages.std().item(), global_step)
            writer.add_scalar("gpg/usable_fraction", len(usable) / args.batch_size, global_step)
            print("SPS:", int(global_step / (time.time() - start_time)))
            writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    if args.benchmark:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        benchmark_end = time.perf_counter()
        measured_steps = global_step - benchmark_start_step
        measured_seconds = benchmark_end - benchmark_start
        result = {
            "algorithm": "gpg",
            "backend": args.env_backend,
            "batch_size": args.batch_size,
            "benchmark": "full_training_loop",
            "binning": args.binning,
            "compile": False,
            "env_id": args.env_id,
            "group_mode": args.group_mode,
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

    envs.close()
    if writer is not None:
        writer.close()
