# PPO + Relative Value Learning (RV): the critic learns an antisymmetric
# function Delta(s_i, s_j) ~ V(s_i) - V(s_j) instead of absolute values, and
# advantages are rebuilt from those differences (R-GAE).
#
# "Relative Value Learning", Hoeftmann, Robine & Harmeling, ICLR 2026
# (arXiv:2607.21120). Official code: github.com/Hauf3n/relative-value-learning.
# Ported here rather than vendored; tests/crosscheck/check_rv.py diffs this
# file against that repository component by component.
#
# The PPO scaffolding is CleanRL's cleanrl/ppo_atari.py
# (https://github.com/vwxyzjn/cleanrl, MIT; license in cleanrl/LICENSE.md).
# Supports gymnasium, cule, and envpool backends.
#
# Deliberate divergences from the official code, all asserted in the tests:
#   * Trajectory ranking builds the N x N pairwise matrix from N encodings
#     instead of encoding 2*N^2 observations. For the linear head this is an
#     algebraic identity (see `start_state_offsets`), not an approximation.
#   * The official loop caps each epoch at 8 minibatches; at its own default
#     num_minibatches=8 that cap never binds, so it is not reproduced.
#   * Episode-boundary convention. The official code is written against
#     EnvPool's gym API, where the observation following a done is a leftover
#     terminal frame; it therefore marks start states one index late and drops
#     the last step of every terminal episode. This repository's backends
#     (gymnasium, cule, envpool) all auto-reset so that the stored `next_obs`
#     already belongs to the new episode, so the convention here is:
#         episode_start[t]  obs[t] begins a new (sub-)episode
#         next_done[t]      the transition at t ended an episode
#     and no step is dropped. `tests/crosscheck/check_rv.py` builds the same
#     episode structure under both conventions and checks the outputs agree.
#   * The episode lookup in the pair sampler uses `bucketize(..., right=True)`.
#     Upstream's default `right=False` puts the first index of each episode in
#     the previous episode's range, so that one anchor per boundary draws its
#     "same-episode" partner from the wrong episode.
#
# DOES NOT LEARN -- do not use this file for results yet. On Pong with CuLE,
# 2M steps, plain PPO under matched settings reaches +9.7 while this reaches
# -20.2, with the policy entropy pinned at ln(6) and the critic loss flat. The
# same failure appears at the paper's exact configuration (8 envs, T=128) with
# the zero anchor that upstream forces for Pong, so it is not the batch size and
# not trajectory ranking. Every component below matches the official
# implementation numerically (see the CONFIRMED note), which places the defect
# in the training loop in this file rather than in the ported functions.
#
# CONFIRMED against the official implementation: `tests/crosscheck/check_rv.py`
# transplants Hauf3n/relative-value-learning's agent weights into this file and
# runs the two side by side. 28/28 components match on CPU and CUDA (<= 1.4e-6,
# float32 reduction order; the trajectory-ranking offsets differ by 3e-7
# because the O(N) identity replaces their O(N^2) matrix): the encoder, the
# relative-value head, the start-state mask, the ranking offsets, R-GAE under
# both anchors, every n-step target helper, the pairwise target with its
# terminal cases, the partner sampler draw for draw, the clipped critic loss,
# and the hyperparameters read off upstream's own args.py.
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
import torch.optim as optim
import tyro
from torch.distributions.categorical import Categorical
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

from cleanrl_utils.atari_wrappers import (  # isort:skip
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

    # Algorithm specific arguments
    env_id: str = "BreakoutNoFrameskip-v4"
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
    lr_minimum: float = 1e-5
    """floor for the annealed learning rate"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for R-GAE"""
    num_minibatches: int = 8
    """the number of mini-batches"""
    update_epochs: int = 5
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """normalize R-GAE advantages over the whole batch, as the official code does"""
    clip_coef: float = 0.1
    """the surrogate clipping coefficient"""
    anneal_clip: bool = True
    """anneal the surrogate clipping coefficient"""
    clip_minimum: float = 0.001
    """floor for the annealed surrogate clipping coefficient"""
    ent_coef: float = 0.00875
    """coefficient of the entropy (official code's default; the paper's table says 0.01)"""
    rv_coef: float = 1.25
    """coefficient of the relative-value loss"""
    clip_rv_loss: bool = True
    """toggle the PPO-style clipped relative-value loss"""
    clip_rv: float = 0.15
    """the relative-value clipping coefficient"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = 0.0225
    """stop updating the policy (but keep training the critic) above this approx KL"""
    n_step_cutoff: int = 6
    """the n-step relative-value target horizon"""
    anneal_n_step_cutoff: bool = True
    """anneal the n-step horizon down to `n_step_cutoff_minimum`"""
    n_step_cutoff_minimum: int = 5
    """floor for the annealed n-step horizon (the paper's table reports 5)"""
    anneal_n_step_cutoff_frac: float = 0.1
    """fraction of training over which the n-step horizon anneals"""
    p_same_episode: float = 0.33
    """probability that the partner state is drawn from the same episode"""
    trajectory_ranking: bool = True
    """rank trajectories to drive E[B_t] to zero; disabling it uses the zero anchor"""
    value_head: str = "linear"
    """relative-value head: `linear` (paper default) or `mlp_symmetric`"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""

    benchmark: bool = False
    """run a fixed warmup/measurement window and print a JSON benchmark result"""
    benchmark_warmup_iterations: int = 3
    """full training iterations excluded from benchmark timing"""
    benchmark_measure_iterations: int = 10
    """full training iterations included in benchmark timing"""


def make_env(env_id, idx, capture_video, run_name):
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
            returns = np.asarray(infos["r"])
            lengths = np.asarray(infos["l"])
            return {
                "final_info": [
                    {"episode": {"r": float(returns[i]), "l": int(lengths[i])}} if game_over[i] else None
                    for i in range(len(game_over))
                ]
            }
    return infos


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    if layer.bias is not None:
        torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    """Shared PPO encoder, a categorical policy head, and a relative-value head.

    Section 5.1: "the policy and relative value function share the CNN encoder
    f_enc(s) and then split their computation by using a single linear layer
    for their respective outputs". The relative value is
    Delta(s_i, s_j) = Phi(f_enc(s_i) - f_enc(s_j)) with Phi carrying no bias, so
    Delta(s_i, s_j) = -Delta(s_j, s_i) and Delta(s, s) = 0 hold by construction.
    """

    def __init__(self, envs, value_head="linear"):
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
        if value_head == "linear":
            self.rv_head = layer_init(nn.Linear(512, 1, bias=False), std=1.0)
        elif value_head == "mlp_symmetric":
            # Odd by construction: tanh is odd and no layer carries a bias, so
            # negating the input negates the output. Antisymmetry survives.
            self.rv_head = nn.Sequential(
                layer_init(nn.Linear(512, 96, bias=False), std=1.0),
                nn.Tanh(),
                layer_init(nn.Linear(96, 64, bias=False), std=1.0),
                nn.Tanh(),
                layer_init(nn.Linear(64, 64, bias=False), std=1.0),
                nn.Tanh(),
                layer_init(nn.Linear(64, 1, bias=False), std=1.0),
            )
        else:
            raise ValueError(f"unknown value head: {value_head}")
        self.linear_head = value_head == "linear"

    def encode(self, x):
        return self.network(x / 255.0)

    def encoding_to_rv(self, encoded_i, encoded_j):
        return self.rv_head(encoded_i - encoded_j).squeeze(-1)

    def get_rv(self, x_i, x_j):
        """Delta(s_i, s_j); both states go through the encoder in one batch."""
        encoded = self.encode(torch.cat((x_i, x_j), dim=0))
        encoded_i, encoded_j = encoded.chunk(2, dim=0)
        return self.encoding_to_rv(encoded_i, encoded_j)

    def get_action(self, x, action=None):
        logits = self.actor(self.encode(x))
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy()


# ---------------------------------------------------------------------------
# Trajectory ranking (Section 4.1)
# ---------------------------------------------------------------------------


def find_start_states_in_batch(episode_start):
    """Mask of states that begin a (sub-)episode.

    K(m) = {0} union {t : obs[t] begins a new episode} (Equation 23). Row 0 is
    always a sub-episode start: the rollout buffer boundary cuts an episode,
    and R-GAE anchors each such fragment independently.
    """
    state_mask = episode_start.to(torch.bool).clone()
    state_mask[0] = True
    return state_mask


def start_state_offsets(encoded_start_states, agent):
    """Non-negative offsets V_hat for the start states (Equation 25).

    The paper forms the full N x N matrix Delta_ij and takes row means. For the
    linear head Phi(z) = w . z that row mean collapses:

        O_i = (1/N) sum_j w . (f_i - f_j) = w . f_i - mean_j (w . f_j),

    so the N x N matrix never has to be built, and the encoder runs N times
    instead of 2N^2. `tests/test_rv_equivalence.py` checks this against the
    explicit quadratic form. A non-linear head has no such identity, so it
    falls back to the quadratic form.
    """
    n = encoded_start_states.shape[0]
    if agent.linear_head:
        head_values = agent.rv_head(encoded_start_states).squeeze(-1)
        offsets = head_values - head_values.mean()
    else:
        expanded_i = encoded_start_states.unsqueeze(1).expand(n, n, -1).reshape(n * n, -1)
        expanded_j = encoded_start_states.unsqueeze(0).expand(n, n, -1).reshape(n * n, -1)
        offsets = agent.encoding_to_rv(expanded_i, expanded_j).view(n, n).mean(dim=1)
    return offsets - offsets.min()


def init_values_optimal(encoded_obs, episode_start, agent):
    """Per-start-state offsets laid out over the rollout, shape (T + 1, actors)."""
    steps, actors = episode_start.shape
    state_mask = find_start_states_in_batch(episode_start)
    start_positions = torch.nonzero(state_mask, as_tuple=False)
    value_offset = torch.zeros(steps + 1, actors, device=episode_start.device)
    if start_positions.numel() == 0:
        return value_offset
    start_t, start_a = start_positions[:, 0], start_positions[:, 1]
    offsets = start_state_offsets(encoded_obs[start_t, start_a], agent)
    value_offset[start_t, start_a] = offsets
    return value_offset


# ---------------------------------------------------------------------------
# R-GAE (Section 3.3)
# ---------------------------------------------------------------------------


def relative_values(delta, next_done, value_offset=None):
    """Telescope the differences into relative values (Equation 9).

    V~(s_0) = 0 and V~(s_t) = sum_{k<t} Delta(s_{k+1}, s_k), re-anchored at
    every episode boundary. With `value_offset` supplied, each (sub-)episode
    starts from its ranked offset (Equation 26) instead of from zero.
    """
    steps, actors = delta.shape
    values = torch.zeros(steps + 1, actors, device=delta.device, dtype=delta.dtype)
    running = torch.zeros(actors, device=delta.device, dtype=delta.dtype)
    if value_offset is not None:
        running = running + value_offset[0]
    for t in range(steps):
        values[t] = running
        running = running + delta[t]
        running = running * (1 - next_done[t])
        if value_offset is not None:
            running = running + value_offset[t + 1]
    values[steps] = running
    return values


def relative_gae(values, rewards, next_done, gamma, gae_lambda):
    """R-GAE: the usual GAE recursion driven by relative TD residuals (Eq. 10-11)."""
    td_residuals = rewards + gamma * values[1:] * (1 - next_done) - values[:-1]
    advantages = torch.zeros_like(td_residuals)
    running = torch.zeros(td_residuals.shape[1], device=td_residuals.device, dtype=td_residuals.dtype)
    for t in reversed(range(td_residuals.shape[0])):
        running = running * (1 - next_done[t])
        running = td_residuals[t] + gae_lambda * gamma * running
        advantages[t] = running
    return advantages


# ---------------------------------------------------------------------------
# Relative value targets (Section 3.4)
# ---------------------------------------------------------------------------


def steps_to_next_done(next_done):
    """For each t, how many transitions remain up to and including the next done.

    Rows with no done ahead of them get the distance to the end of the buffer,
    which is what caps the n-step horizon at the rollout boundary.
    """
    mask = next_done.bool()
    n = mask.shape[0]
    reversed_mask = mask.flip(0)
    index = torch.arange(n, device=next_done.device)
    last_done = torch.where(reversed_mask, index, torch.full_like(index, -1))
    last_seen, _ = last_done.cummax(dim=0)
    # The "+ 1" is where this parts company with the official helper. Upstream's
    # flag marks a terminal *observation*, so the count stops before it; ours
    # marks the terminal *transition*, which is itself a valid step and carries
    # the terminal reward. Without it the n-step window always stops one step
    # short of the boundary, the terminal reward never enters a target, and the
    # Equation 20 case distinction can never fire.
    distance = torch.where(last_seen >= 0, index - last_seen + 1, index + 1)
    return distance.flip(0).to(torch.long)


def shifted_reward_rows(rewards, n):
    """Row i is `rewards` shifted left by i, zero-filled past the end."""
    steps = rewards.shape[0]
    rows = torch.arange(n, device=rewards.device).unsqueeze(1)
    cols = torch.arange(steps, device=rewards.device).unsqueeze(0)
    index = rows + cols
    gathered = rewards[index.clamp(max=steps - 1)]
    return gathered * (index < steps).to(rewards.dtype)


def compute_discounted_reward_sums(rewards, next_done, gamma):
    """All (1..T)-step discounted reward sums per timestep, truncated at the next done."""
    steps = rewards.shape[0]
    mask = torch.triu(torch.ones((steps, steps), dtype=torch.bool, device=rewards.device))
    powers = torch.arange(steps, device=rewards.device).unsqueeze(1).repeat(1, steps) * mask
    # Built in the rewards' dtype: `gamma ** int_tensor` would otherwise land in
    # the default float32 and break the matmul under float64.
    discounts = (gamma ** powers.to(rewards.dtype)) * mask
    all_step_targets = shifted_reward_rows(rewards, steps) @ discounts
    distance = steps_to_next_done(next_done)
    valid = torch.arange(steps, device=rewards.device).unsqueeze(0) < distance.unsqueeze(1)
    return all_step_targets * valid.to(all_step_targets.dtype), distance


def prepare_data(data, gamma):
    """Attach `discounted_reward_sums` and `max_n_step` to a (T, actors) rollout."""
    steps, actors = data["rewards"].shape[:2]
    sums = torch.zeros((steps, actors, steps), device=data["rewards"].device, dtype=data["rewards"].dtype)
    max_n_step = torch.zeros((steps, actors), device=data["rewards"].device, dtype=torch.long)
    for actor in range(actors):
        sums[:, actor, :], max_n_step[:, actor] = compute_discounted_reward_sums(
            data["rewards"][:, actor], data["next_done"][:, actor], gamma
        )
    data["discounted_reward_sums"] = sums
    data["max_n_step"] = max_n_step
    return data


def rv_n_step_target(flat_data, idx_i, idx_j, agent, gamma, n_cutoff):
    """The pairwise n-step target of Equation 21 with the Equation 20 terminal cases.

    Absolute values are unavailable, so a terminal successor cannot simply be
    bootstrapped as 0 - V(s'). Each terminal case is rewritten in terms of
    observable rewards and a non-terminal pairwise difference:

        neither terminal   Delta(s_{i+n}, s_{j+n})
        s_i terminal       Delta(s_i, s_{j+n}) - r_i
        s_j terminal       Delta(s_{i+n}, s_j) + r_j
        both terminal      0                     (variance-reducing default)
    """
    horizon = torch.minimum(
        torch.minimum(flat_data["max_n_step"][idx_i], flat_data["max_n_step"][idx_j]),
        torch.as_tensor(n_cutoff, device=idx_i.device),
    ) - 1

    end_i, end_j = idx_i + horizon, idx_j + horizon
    obs_i, obs_j = flat_data["encoded_obs"][end_i], flat_data["encoded_obs"][end_j]
    next_i, next_j = flat_data["encoded_next_obs"][end_i], flat_data["encoded_next_obs"][end_j]
    rewards_i, rewards_j = flat_data["rewards"][end_i], flat_data["rewards"][end_j]
    next_done_i, next_done_j = flat_data["next_done"][end_i], flat_data["next_done"][end_j]
    sum_i = flat_data["discounted_reward_sums"][idx_i, horizon]
    sum_j = flat_data["discounted_reward_sums"][idx_j, horizon]

    with torch.no_grad():
        # One batched call for all four bootstrap variants.
        first = torch.cat([next_i, next_i, obs_i, obs_i], dim=0)
        second = torch.cat([next_j, obs_j, next_j, obs_j], dim=0)
        raw = agent.encoding_to_rv(first, second).view(4, obs_i.shape[0])
        delta_next, delta_raw_j, delta_raw_i, _ = raw

        reward_difference = sum_i - sum_j
        # Same trap as in compute_discounted_reward_sums: an integer exponent
        # would put the discount in the default float32 regardless of dtype.
        discount = gamma ** (horizon + 1).to(reward_difference.dtype)
        target = reward_difference + discount * delta_next
        target_i = reward_difference + discount * (delta_raw_i - rewards_i)
        target_j = reward_difference + discount * (delta_raw_j + rewards_j)
        target_ij = torch.zeros_like(target)

        mask_i, mask_j = next_done_i.bool(), next_done_j.bool()
        target = torch.where(
            mask_i & mask_j,
            target_ij,
            torch.where(mask_i, target_i, torch.where(mask_j, target_j, target)),
        )
        old_delta = agent.encoding_to_rv(
            flat_data["encoded_obs"][idx_i], flat_data["encoded_obs"][idx_j]
        )
    return target, old_delta


@torch.no_grad()
def get_target_indices(anchors, episode_offsets, p_same=0.33, generator=None):
    """Pick a partner index for each anchor (Appendix D, "Additional Hyperparameters").

    With probability `p_same` the partner comes from the anchor's own episode,
    otherwise from anywhere in the batch. Self-pairs are excluded either way,
    since Delta(s, s) = 0 carries no signal.
    """
    anchors = anchors.long()
    device = anchors.device
    total = int(episode_offsets[-1].item())
    batch = anchors.numel()

    # `right=True` matters. The official code calls bucketize with the default
    # right=False, which sends the first index of each episode into the previous
    # episode's range: with offsets [0, 20, 45], anchor 20 is looked up as
    # episode 0. That is one anchor per episode boundary drawing its
    # "same-episode" partner from the wrong episode. The paper describes
    # sampling from the anchor's own episode, so this port takes right=True and
    # tests/crosscheck/check_rv.py pins the difference to exactly those rows.
    episode_index = torch.bucketize(anchors, episode_offsets[1:], right=True)
    start = episode_offsets[episode_index].to(device)
    end = episode_offsets[episode_index + 1].to(device)
    length = end - start

    can_same = length > 1
    draw = torch.rand(batch, device=device, generator=generator) < p_same
    use_same = draw & can_same

    # Uniform over the episode minus the anchor itself, via a cyclic shift.
    shift = torch.floor(
        torch.rand(batch, device=device, generator=generator) * torch.clamp_min(length - 1, 1)
    ).long()
    same_episode = start + ((anchors - start + 1 + shift) % torch.clamp_min(length, 1))

    # Uniform over the whole batch minus the anchor itself.
    drawn = torch.randint(0, max(total - 1, 1), (batch,), device=device, generator=generator)
    anywhere = drawn + (drawn >= anchors).long()

    return torch.where(use_same, same_episode, anywhere)


def rv_loss(predicted, target, old_delta, clip_rv_loss, clip_rv):
    """Squared error on the pairwise difference, PPO-style clipped against `old_delta`."""
    if not clip_rv_loss:
        return ((predicted - target) ** 2).mean(), torch.zeros((), device=predicted.device)
    unclipped = (predicted - target) ** 2
    clipped_prediction = old_delta + torch.clamp(predicted - old_delta, -clip_rv, clip_rv)
    clipped = (clipped_prediction - target) ** 2
    clip_fraction = ((predicted - old_delta).abs() > clip_rv).float().mean()
    return torch.max(unclipped, clipped).mean(), clip_fraction


def split_into_episodes(data, num_envs, num_steps):
    """Slice a (T, actors) rollout into per-episode chunks, in actor-major order.

    An episode ends at the transition where `next_done` is set. The partner
    sampler needs these offsets to honour p_same, and the pieces are
    concatenated back in this order so index ranges stay contiguous per episode.
    """
    episodes = []
    next_done = data["next_done"].T
    for env_index in range(num_envs):
        done_indices = torch.where(next_done[env_index])[0]
        start = 0
        for end in done_indices:
            episodes.append(data[start : int(end) + 1, env_index])
            start = int(end) + 1
        if start < num_steps:
            episodes.append(data[start:num_steps, env_index])
    return episodes


if __name__ == "__main__":
    import tensordict

    process_start = time.perf_counter()
    args = tyro.cli(Args)
    if args.num_envs < 1:
        raise ValueError("num_envs must be positive")
    if args.num_steps < 1:
        raise ValueError("num_steps must be positive")
    if args.num_minibatches < 1:
        raise ValueError("num_minibatches must be positive")
    if not 0.0 <= args.p_same_episode <= 1.0:
        raise ValueError("p_same_episode must be a probability")
    if args.n_step_cutoff < 1:
        raise ValueError("n_step_cutoff must be positive")
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
            [make_env(args.env_id, i, args.capture_video, run_name) for i in range(args.num_envs)],
        )
    else:
        raise ValueError(f"unsupported environment backend: {args.env_backend}")
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    agent = Agent(envs, args.value_head).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    observation_shape = envs.single_observation_space.shape
    episode_stats = EpisodeStats(20, None)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = to_tensor(next_obs, device).to(torch.uint8)
    next_done = torch.zeros(args.num_envs, device=device)
    benchmark_start = None
    benchmark_start_step = None

    for iteration in range(1, args.num_iterations + 1):
        if args.benchmark and iteration == args.benchmark_warmup_iterations + 1:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            benchmark_start = time.perf_counter()
            benchmark_start_step = global_step

        progress = (iteration - 1.0) / args.num_iterations
        if args.anneal_lr:
            optimizer.param_groups[0]["lr"] = max(args.lr_minimum, (1.0 - progress) * args.learning_rate)
        clip_coef = args.clip_coef
        if args.anneal_clip:
            clip_coef = max(args.clip_minimum, (1.0 - progress) * args.clip_coef)
        n_step_cutoff = args.n_step_cutoff
        if args.anneal_n_step_cutoff:
            frac = max(0.0, 1.0 - progress / args.anneal_n_step_cutoff_frac)
            n_step_cutoff = int(
                args.n_step_cutoff_minimum + frac * (args.n_step_cutoff - args.n_step_cutoff_minimum)
            )

        # ROLLOUT. RV needs next_obs stored per step, not just the bootstrap state.
        transitions = []
        for step in range(args.num_steps):
            global_step += args.num_envs
            current_obs, current_done = next_obs, next_done
            with torch.no_grad():
                action, logprob, _ = agent.get_action(current_obs.float())

            step_obs, reward, terminations, truncations, infos = step_env(envs, action)
            step_done = done_tensor(terminations, truncations, device)
            step_obs = to_tensor(step_obs, device).to(torch.uint8)

            transitions.append(
                tensordict.TensorDict(
                    {
                        "obs": current_obs,
                        "actions": action,
                        "logprobs": logprob,
                        "rewards": to_tensor(reward, device).view(-1).float(),
                        "episode_start": current_done,
                        "next_obs": step_obs,
                        "next_done": step_done,
                    },
                    batch_size=[args.num_envs],
                )
            )
            next_obs, next_done = step_obs, step_done

            if not args.benchmark:
                episode_stats.update(completed_episode_infos(infos, step_done), global_step, writer)

        data = torch.stack(transitions, dim=0)

        # R-GAE
        with torch.no_grad():
            flat_obs = data["obs"].reshape((-1,) + observation_shape).float()
            flat_next_obs = data["next_obs"].reshape((-1,) + observation_shape).float()
            encoded_obs = agent.encode(flat_obs)
            encoded_next_obs = agent.encode(flat_next_obs)
            delta = agent.encoding_to_rv(encoded_next_obs, encoded_obs).view(args.num_steps, args.num_envs)
            offset = None
            if args.trajectory_ranking:
                offset = init_values_optimal(
                    encoded_obs.view(args.num_steps, args.num_envs, -1), data["episode_start"], agent
                )
            values = relative_values(delta, data["next_done"], offset)
            data["advantages"] = relative_gae(
                values, data["rewards"], data["next_done"], args.gamma, args.gae_lambda
            )

        # n-step targets need per-episode reward sums and horizons
        prepare_data(data, args.gamma)
        episodes = split_into_episodes(data, args.num_envs, args.num_steps)
        episode_lengths = [len(episode) for episode in episodes]
        episode_offsets = torch.tensor(
            [0] + np.cumsum(episode_lengths).tolist(), device=device, dtype=torch.long
        )
        data = tensordict.cat(episodes, 0).view(-1)
        # No rows are dropped here. The official code filters `max_n_step > 0`
        # to clear the holes its envpool step-removal leaves behind, but that
        # filter also shifts every index out from under `episode_offsets`.
        # Under this file's convention max_n_step >= 1 everywhere, so the
        # offsets stay exact and each n-step window stays inside its episode.
        if data.shape[0] < args.minibatch_size:
            continue

        with torch.no_grad():
            data["encoded_obs"] = agent.encode(data["obs"].float())
            data["encoded_next_obs"] = agent.encode(data["next_obs"].float())

        if args.norm_adv:
            data["advantages"] = (data["advantages"] - data["advantages"].mean()) / (
                data["advantages"].std() + 1e-8
            )

        optimize_policy = True
        for epoch in range(args.update_epochs):
            batch_indices = torch.randperm(data.shape[0], device=device)
            partner_indices = get_target_indices(
                batch_indices, episode_offsets, p_same=args.p_same_episode
            )
            targets, old_deltas = rv_n_step_target(
                data, batch_indices, partner_indices, agent, args.gamma, n_step_cutoff
            )

            approx_kls = []
            for start in range(0, batch_indices.shape[0] - args.minibatch_size + 1, args.minibatch_size):
                stop = start + args.minibatch_size
                anchors = batch_indices[start:stop]
                partners = partner_indices[start:stop]

                _, newlogprob, entropy = agent.get_action(data["obs"][anchors].float(), data["actions"][anchors])
                logratio = newlogprob - data["logprobs"][anchors]
                ratio = logratio.exp()
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean()
                    approx_kls.append(approx_kl.item())

                advantages = data["advantages"][anchors]
                pg_loss = torch.max(
                    -advantages * ratio,
                    -advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef),
                ).mean()

                predicted = agent.get_rv(data["obs"][anchors].float(), data["obs"][partners].float())
                value_loss, clip_fraction = rv_loss(
                    predicted, targets[start:stop], old_deltas[start:stop], args.clip_rv_loss, args.clip_rv
                )

                policy_term = pg_loss - args.ent_coef * entropy.mean() if optimize_policy else 0.0
                loss = policy_term + args.rv_coef * value_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kls and np.mean(approx_kls) > args.target_kl:
                optimize_policy = False

        if writer is not None:
            writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
            writer.add_scalar("charts/clip_coef", clip_coef, global_step)
            writer.add_scalar("charts/n_step_cutoff", n_step_cutoff, global_step)
            writer.add_scalar("losses/rv_loss", value_loss.item(), global_step)
            writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
            writer.add_scalar("losses/entropy", entropy.mean().item(), global_step)
            writer.add_scalar("losses/approx_kl", float(np.mean(approx_kls)), global_step)
            writer.add_scalar("losses/rv_clipfrac", clip_fraction.item(), global_step)
            sps = int(global_step / (time.time() - start_time))
            writer.add_scalar("charts/SPS", sps, global_step)
            print("SPS:", sps)

    if args.benchmark:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        benchmark_end = time.perf_counter()
        measured_steps = global_step - benchmark_start_step
        measured_seconds = benchmark_end - benchmark_start
        result = {
            "algorithm": "ppo_rv",
            "backend": args.env_backend,
            "batch_size": args.batch_size,
            "benchmark": "full_training_loop",
            "compile": False,
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
        episode_stats.print_summary()

    envs.close()
    if writer is not None:
        writer.close()
