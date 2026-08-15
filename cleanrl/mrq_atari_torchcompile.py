# MR.Q: Towards General-Purpose Model-Free Reinforcement Learning
# (Fujimoto et al., ICLR 2025, https://arxiv.org/abs/2501.16142).
#
# The official implementation (https://github.com/facebookresearch/MRQ) is
# licensed CC BY-NC 4.0, which is incompatible with this repository's license,
# so NO code was copied from it. This is an independent PyTorch reimplementation
# written against the published paper and the semantics of the reference agent;
# `tests/test_mrq_equivalence.py` pins the parts that must match.
#
# What MR.Q does: a model-free TD3-style actor-critic whose value function is
# fed a learned model-based representation. An encoder maps observations to zs
# and (zs, action) to zsa, and is trained purely by a self-predictive objective
# rolled `enc_horizon` steps on its own predicted latents -- a masked MSE on the
# next latent (target encoder), a two-hot cross-entropy on the reward, and an
# MSE on the termination flag. The value and policy then train on top of a
# *frozen* zsa: the encoder receives no gradient from either RL loss. Twin Q
# with a min over the pair, LAP prioritization (Huber loss + |delta|^alpha
# priorities floored at 1), hard target syncs every `target_network_frequency`
# updates, and reward normalization by the buffer's mean |reward| so the value
# scale is game independent. The discrete policy emits logits turned into a soft
# one-hot by Gumbel-softmax at a deliberately high temperature (tau 10), which
# is what lets the deterministic policy gradient flow through a discrete action.
#
# Structure follows spr_atari.py -- it already has the prioritized subtrajectory
# replay and the K-step latent rollout MR.Q's encoder loss needs -- which is in
# turn adapted from CleanRL (https://github.com/vwxyzjn/cleanrl, MIT; license in
# cleanrl/LICENSE.md).
# Supports gymnasium, cule, and envpool.
#
# This is the torch.compile / CUDA-graph twin of mrq_atari.py, following
# LeanRL's structure (https://github.com/meta-pytorch/LeanRL, MIT). The three
# fixed-shape regions -- behavior policy, encoder update, and the value+policy
# update -- are compiled separately and may be captured as CUDA graphs. Two
# deliberate divergences from the eager file, neither of which changes the
# algorithm:
#   * the uniform-random warmup is expressed as a `where` on a probability
#     tensor instead of a Python branch, so the policy has one static graph from
#     the first environment step;
#   * `reward_scale`, `target_reward_scale` and the done-loss weight are 0-dim
#     tensors rather than Python floats, so changing them does not retrace.
#
# CONFIRMED against the official implementation: `tests/crosscheck/check_mrq.py`
# transplants facebookresearch/MRQ's own weights into this file's modules and
# diffs every forward pass and loss term. 24/24 components match on CPU (all
# bit-exact) and on CUDA (bit-exact except the symexp bin table, 9e-8
# relative, because upstream builds it on the GPU and this file on the CPU).
# Covered: zs / zsa / model_all, twin Q, policy logits and the Gumbel-softmax
# action, the two-hot bins / transform / inverse / cross-entropy, masked_mse,
# realign, multi_step_reward, shift_aug, the 5-step encoder loss, the scaled Q
# target, the smooth-L1 value loss, the LAP priority, and 31 hyperparameters.
import copy
import json
import os
import random
import sys
import time
from dataclasses import dataclass
from typing import NamedTuple

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
from tensordict import TensorDict
from tensordict.nn import CudaGraphModule
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

from cleanrl_utils.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)
from cleanrl_utils.buffers import AtariReplayBuffer, PrioritizedAtariReplayBuffer
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

    env_id: str = "BreakoutNoFrameskip-v4"
    """the id of the environment"""
    total_timesteps: int = 2500000
    """total timesteps of the experiments (official Atari budget: 2.5M agent steps)"""
    encoder_learning_rate: float = 1e-4
    """the learning rate of the encoder optimizer"""
    value_learning_rate: float = 3e-4
    """the learning rate of the value optimizer"""
    policy_learning_rate: float = 3e-4
    """the learning rate of the policy optimizer"""
    weight_decay: float = 1e-4
    """AdamW weight decay, shared by all three optimizers"""
    num_envs: int = 1
    """the number of parallel game environments"""
    buffer_size: int = 1000000
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    target_network_frequency: int = 250
    """learner updates between hard target syncs (also the encoder burst size)"""
    batch_size: int = 256
    """the batch size of sample from the reply memory"""
    learning_starts: int = 10000
    """timestep to start learning; before it, actions are uniformly random"""
    learner_updates_per_vector_step: float = 1.0
    """gradient updates accrued per vector environment step (official: one per step)"""
    replay_ratio: float | None = None
    """sampled replay items per collected transition; overrides learner-updates-per-vector-step"""
    max_training_seconds: float = 0.0
    """wall-clock training limit; zero disables it"""
    solve_reward: float | None = None
    """stop when the moving episodic return reaches this value"""
    solve_window: int = 20
    """number of completed episodes in the solve moving average"""

    enc_horizon: int = 5
    """steps the encoder unrolls its own latent for the self-predictive loss"""
    q_horizon: int = 3
    """steps of multi-step return used by the value loss"""
    dyn_weight: float = 1.0
    """weight of the latent dynamics MSE in the encoder loss"""
    reward_weight: float = 0.1
    """weight of the two-hot reward cross-entropy in the encoder loss"""
    done_weight: float = 0.1
    """weight of the termination MSE in the encoder loss"""
    num_bins: int = 65
    """number of two-hot bins for the reward head"""
    bin_lower: float = -10.0
    """lower end of the pre-symexp bin range"""
    bin_upper: float = 10.0
    """upper end of the pre-symexp bin range"""
    zs_dim: int = 512
    """observation-embedding width"""
    za_dim: int = 256
    """action-embedding width"""
    zsa_dim: int = 512
    """joint state-action embedding width, and the value network's input"""
    enc_hdim: int = 512
    """hidden width of the encoder MLPs"""
    value_hdim: int = 512
    """hidden width of the value MLPs"""
    policy_hdim: int = 512
    """hidden width of the policy MLP"""
    exploration_noise: float = 0.2
    """Gaussian noise on the behavior-policy action; halved for discrete actions"""
    target_policy_noise: float = 0.2
    """Gaussian noise on the target-policy action; halved for discrete actions"""
    noise_clip: float = 0.3
    """clip on the target-policy noise; halved for discrete actions"""
    gumbel_tau: float = 10.0
    """Gumbel-softmax temperature of the discrete policy"""
    pre_activ_weight: float = 1e-5
    """weight of the policy's squared-logit regularizer"""
    value_grad_clip: float = 20.0
    """gradient-norm clip on the value network"""
    prioritized_replay_alpha: float = 0.4
    """LAP priority exponent"""
    min_priority: float = 1.0
    """LAP priority floor, matched to the Huber loss transition point"""
    data_augmentation: bool = True
    """apply MR.Q's random-shift augmentation to learner inputs"""
    clip_rewards: bool = False
    """clip environment rewards to [-1, 1] (the official Atari setup does not)"""
    episodic_life: bool = False
    """end an episode on life loss (the official Atari setup does not)"""
    compile: bool = False
    """whether to compile the fixed-shape policy and learner regions"""
    cudagraphs: bool = False
    """whether to wrap the policy and learner updates in CudaGraphModule"""
    benchmark: bool = False
    """run a fixed warmup/measurement window and print a JSON benchmark result"""
    benchmark_warmup_iterations: int = 10
    """vector-environment steps excluded from benchmark timing"""
    benchmark_measure_iterations: int = 30
    """vector-environment steps included in benchmark timing"""


def make_env(env_id, seed, idx, capture_video, run_name, clip_rewards, episodic_life):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)

        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        if episodic_life:
            env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        if clip_rewards:
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


def random_shift_augmentation(images: torch.Tensor, pad: int = 4) -> torch.Tensor:
    """MR.Q's random shift: replicate-pad, then bilinear resample at an integer offset.

    This is the DrQ-v2 formulation the official implementation uses, not the
    crop-based one in `drq_atari.py` -- and unlike SPR's, there is no intensity
    noise. The sampling grid is built from the height alone, so square frames
    are assumed; Atari's 84x84 satisfies that.
    """
    batch, _, height, _ = images.shape
    padded = F.pad(images, (pad, pad, pad, pad), mode="replicate")
    eps = 1.0 / (height + 2 * pad)

    arange = torch.linspace(
        -1.0 + eps, 1.0 - eps, height + 2 * pad, device=images.device, dtype=images.dtype
    )[:height]
    arange = arange.unsqueeze(0).repeat(height, 1).unsqueeze(2)
    base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
    base_grid = base_grid.unsqueeze(0).repeat(batch, 1, 1, 1)

    shift = torch.randint(
        0, 2 * pad + 1, size=(batch, 1, 1, 2), device=images.device, dtype=images.dtype
    )
    shift = shift * (2.0 / (height + 2 * pad))
    return F.grid_sample(padded, base_grid + shift, padding_mode="zeros", align_corners=False)


def weight_init(layer: nn.Module) -> None:
    """Xavier-uniform at the ReLU gain, zero bias, for every Linear and Conv2d."""
    if isinstance(layer, (nn.Linear, nn.Conv2d)):
        nn.init.xavier_uniform_(layer.weight, nn.init.calculate_gain("relu"))
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)


def ln_activ(x: torch.Tensor, activ) -> torch.Tensor:
    """Parameter-free LayerNorm over the last dimension, then the activation."""
    return activ(F.layer_norm(x, (x.shape[-1],)))


class BaseMLP(nn.Module):
    """Three Linear layers with a parameter-free LayerNorm + activation between."""

    def __init__(self, input_dim: int, output_dim: int, hdim: int, activ="elu"):
        super().__init__()
        self.l1 = nn.Linear(input_dim, hdim)
        self.l2 = nn.Linear(hdim, hdim)
        self.l3 = nn.Linear(hdim, output_dim)
        self.activ = getattr(F, activ)
        self.apply(weight_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = ln_activ(self.l1(x), self.activ)
        y = ln_activ(self.l2(y), self.activ)
        return self.l3(y)


class TwoHot(nn.Module):
    """Two-hot encoding over symexp-spaced bins, as used by the reward head."""

    def __init__(self, lower: float = -10.0, upper: float = 10.0, num_bins: int = 65):
        super().__init__()
        bins = torch.linspace(lower, upper, num_bins)
        # symexp: sign(x) * (exp(|x|) - 1). The bins are linear in log space, so
        # they resolve small rewards finely and still reach +-e^10.
        bins = bins.sign() * (bins.abs().exp() - 1.0)
        self.register_buffer("bins", bins)
        self.num_bins = int(num_bins)

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        diff = x - self.bins.reshape(1, -1)
        # Push every bin above x out of contention: sign(diff) - 1 is -2 there
        # and 0 at or below x, so argmin lands on the closest bin <= x. A bin
        # exactly equal to x is also pushed out (sign 0), which puts all the
        # mass on it via weight == 1 below.
        diff = diff - 1e8 * (torch.sign(diff) - 1.0)
        index = torch.argmin(diff, dim=1, keepdim=True)

        upper_index = (index + 1).clamp(max=self.num_bins - 1)
        lower_bin = self.bins[index]
        upper_bin = self.bins[upper_index]
        weight = (x - lower_bin) / (upper_bin - lower_bin)

        two_hot = torch.zeros(x.shape[0], self.num_bins, device=x.device, dtype=x.dtype)
        two_hot.scatter_(1, index, 1.0 - weight)
        two_hot.scatter_(1, upper_index, weight)
        return two_hot

    def inverse(self, logits: torch.Tensor) -> torch.Tensor:
        return (F.softmax(logits, dim=-1) * self.bins).sum(-1, keepdim=True)

    def cross_entropy_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return -(self.transform(target) * F.log_softmax(pred, dim=-1)).sum(-1, keepdim=True)


class Encoder(nn.Module):
    """Observation and joint state-action embeddings, plus the one-step model head."""

    def __init__(
        self,
        frame_stack: int,
        n_actions: int,
        num_bins: int = 65,
        zs_dim: int = 512,
        za_dim: int = 256,
        zsa_dim: int = 512,
        hdim: int = 512,
        activ: str = "elu",
    ):
        super().__init__()
        self.zs_cnn1 = nn.Conv2d(frame_stack, 32, 3, stride=2)
        self.zs_cnn2 = nn.Conv2d(32, 32, 3, stride=2)
        self.zs_cnn3 = nn.Conv2d(32, 32, 3, stride=2)
        self.zs_cnn4 = nn.Conv2d(32, 32, 3, stride=1)
        # 84 -> 41 -> 20 -> 9 -> 7, so 32 * 7 * 7.
        self.zs_lin = nn.Linear(32 * 7 * 7, zs_dim)

        self.za = nn.Linear(n_actions, za_dim)
        self.zsa_mlp = BaseMLP(zs_dim + za_dim, zsa_dim, hdim, activ)
        # One head predicts done, the next latent, and the reward logits at once.
        self.model = nn.Linear(zsa_dim, num_bins + zs_dim + 1)

        self.zs_dim = int(zs_dim)
        self.activ = getattr(F, activ)
        self.apply(weight_init)

    def zs(self, state: torch.Tensor) -> torch.Tensor:
        state = state / 255.0 - 0.5
        h = self.activ(self.zs_cnn1(state))
        h = self.activ(self.zs_cnn2(h))
        h = self.activ(self.zs_cnn3(h))
        h = self.activ(self.zs_cnn4(h)).reshape(state.shape[0], -1)
        return ln_activ(self.zs_lin(h), self.activ)

    def forward(self, zs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        za = self.activ(self.za(action))
        return self.zsa_mlp(torch.cat([zs, za], 1))

    def model_all(self, zs: torch.Tensor, action: torch.Tensor):
        """Return (done logit, predicted next zs, reward logits)."""
        dzr = self.model(self(zs, action))
        return dzr[:, 0:1], dzr[:, 1 : self.zs_dim + 1], dzr[:, self.zs_dim + 1 :]


class Policy(nn.Module):
    """Deterministic-policy-gradient actor over a Gumbel-softmax relaxed action."""

    def __init__(
        self,
        n_actions: int,
        gumbel_tau: float = 10.0,
        zs_dim: int = 512,
        hdim: int = 512,
        activ: str = "relu",
    ):
        super().__init__()
        self.policy = BaseMLP(zs_dim, n_actions, hdim, activ)
        self.gumbel_tau = float(gumbel_tau)

    def forward(self, zs: torch.Tensor):
        pre_activ = self.policy(zs)
        # Soft (not straight-through) relaxation. tau is deliberately large, so
        # the actor has to grow its own logits to reach a near-one-hot action;
        # the squared-logit penalty in the policy loss is what keeps that in
        # check. The relaxation is order preserving, so argmax still samples
        # from softmax(logits).
        return F.gumbel_softmax(pre_activ, tau=self.gumbel_tau), pre_activ

    def act(self, zs: torch.Tensor) -> torch.Tensor:
        return self(zs)[0]


class Value(nn.Module):
    """Twin Q over zsa; forward returns both heads as (batch, 2)."""

    class _Head(nn.Module):
        def __init__(self, input_dim: int, hdim: int, activ: str):
            super().__init__()
            self.body = BaseMLP(input_dim, hdim, hdim, activ)
            self.out = nn.Linear(hdim, 1)
            self.activ = getattr(F, activ)
            self.apply(weight_init)

        def forward(self, zsa: torch.Tensor) -> torch.Tensor:
            return self.out(ln_activ(self.body(zsa), self.activ))

    def __init__(self, zsa_dim: int = 512, hdim: int = 512, activ: str = "elu"):
        super().__init__()
        self.q1 = self._Head(zsa_dim, hdim, activ)
        self.q2 = self._Head(zsa_dim, hdim, activ)

    def forward(self, zsa: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.q1(zsa), self.q2(zsa)], 1)


def realign_discrete(x: torch.Tensor) -> torch.Tensor:
    """Snap a noisy relaxed action back onto the one-hot simplex."""
    return F.one_hot(x.argmax(1), x.shape[1]).to(x.dtype)


def masked_mse(x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (F.mse_loss(x, y, reduction="none") * mask).mean()


def multi_step_reward(rewards: torch.Tensor, not_dones: torch.Tensor, gamma: float):
    """Discounted return over the sampled window and the surviving discount.

    `term_discount` is gamma^horizon while the window stays alive and exactly 0
    once it crosses a terminal transition, so the bootstrap term drops out
    without any separate done mask.
    """
    ms_reward = torch.zeros_like(rewards[:, 0])
    scale = torch.ones_like(rewards[:, 0])
    for i in range(rewards.shape[1]):
        ms_reward = ms_reward + scale * rewards[:, i]
        scale = scale * gamma * not_dones[:, i]
    return ms_reward, scale


def encoder_loss(
    encoder: Encoder,
    two_hot: TwoHot,
    observations: torch.Tensor,
    actions: torch.Tensor,
    rewards: torch.Tensor,
    not_dones: torch.Tensor,
    target_zs: torch.Tensor,
    dyn_weight: float,
    reward_weight: float,
    done_weight,
) -> torch.Tensor:
    """MR.Q's self-predictive objective, rolled on the encoder's own latents.

    `observations` is only the *first* state of each window: the encoder sees it
    once and then predicts forward from its own output, so the intermediate
    observations are never inputs -- they only appear through `target_zs`, the
    target encoder's latents for the window's next observations.

    Every term is masked by the running product of not-dones, and the masked
    means are taken over the full tensor rather than renormalized by the mask,
    so a window that terminates early simply contributes less.
    """
    horizon = actions.shape[1]
    pred_zs = encoder.zs(observations)
    prev_not_done = torch.ones_like(not_dones[:, 0])
    loss = torch.zeros((), device=pred_zs.device, dtype=pred_zs.dtype)
    for i in range(horizon):
        pred_done, pred_zs, pred_reward = encoder.model_all(pred_zs, actions[:, i])

        dyn_loss = masked_mse(pred_zs, target_zs[:, i], prev_not_done)
        reward_loss = (two_hot.cross_entropy_loss(pred_reward, rewards[:, i]) * prev_not_done).mean()
        done_loss = masked_mse(pred_done, 1.0 - not_dones[:, i], prev_not_done)
        loss = loss + dyn_weight * dyn_loss + reward_weight * reward_loss + done_weight * done_loss

        # Everything after the first terminal is off-episode; stop counting it.
        prev_not_done = prev_not_done * not_dones[:, i]
    return loss


def scaled_q_target(ms_reward, term_discount, next_q, reward_scale, target_reward_scale):
    """Normalize the TD target by the buffer's current mean |reward|.

    The target network was trained while the normalizer was
    `target_reward_scale`, so its output is put back on the raw reward scale
    before the whole target is renormalized by the current `reward_scale`. That
    keeps the value function's magnitude game independent without making the
    target jump every time the normalizer moves.
    """
    return (ms_reward + term_discount * next_q * target_reward_scale) / reward_scale


def lap_priority(q, q_target, min_priority: float, alpha: float):
    """LAP priority: the larger twin TD error, floored, then raised to alpha.

    Flooring at the Huber transition point is what makes the Huber loss and the
    |delta|^alpha priorities cancel into an unbiased update, which is why this
    buffer carries no importance-sampling weights: every transition with an
    error below the floor is sampled uniformly.
    """
    error = (q - q_target.expand(-1, q.shape[1])).abs().max(1).values
    return error.clamp(min=min_priority).pow(alpha)


class MRQSamples(NamedTuple):
    observations: torch.Tensor
    actions: torch.Tensor
    next_observations: torch.Tensor
    rewards: torch.Tensor
    not_dones: torch.Tensor
    indices: np.ndarray


class MRQReplayBuffer(PrioritizedAtariReplayBuffer):
    """LAP-prioritized replay serving fixed-length subtrajectories.

    Two shapes are served. `include_intermediate=True` (the encoder) returns
    every next observation over the window plus the per-step action, reward and
    not-done; `include_intermediate=False` (the value learner) returns only the
    window's endpoints plus the per-step rewards and not-dones the multi-step
    return needs.

    Unlike the other prioritized buffers here, a window is allowed to run past a
    terminal transition. The losses mask everything after the first
    ``not_done == 0``, and that is exactly what gives the done head and the
    terminal bootstrap anything to learn from.
    """

    def __init__(self, *pargs, enc_horizon: int, q_horizon: int, min_priority: float = 1.0, **kwargs):
        super().__init__(*pargs, **kwargs)
        self.enc_horizon = int(enc_horizon)
        self.q_horizon = int(q_horizon)
        self.window = max(self.enc_horizon, self.q_horizon)
        self.min_priority = float(min_priority)
        # LAP tracks the max of the already-exponentiated priorities, and seeds
        # new transitions with it so they are sampled at least once.
        self.max_priority = self.min_priority
        self.env_terminates = False

    def add(self, next_observations, actions, rewards, dones) -> None:
        # Reimplements the parent's candidate logic over the longer window and
        # without its "do not cross a terminal" restriction.
        AtariReplayBuffer.add(self, next_observations, actions, rewards, dones)
        if self.dones[(self.pos - 1) % self.time_capacity].any():
            self.env_terminates = True

        env_indices = np.arange(self.n_envs, dtype=np.int64)
        overwritten = self._flat_indices(np.full(self.n_envs, self.pos), env_indices)
        self.sum_tree.update(overwritten, np.zeros(self.n_envs, dtype=np.float32))

        if self.steps < self.window:
            return
        candidate_row = (self.pos - self.window) % self.time_capacity
        if self.transition_ids[candidate_row] != self.steps - self.window:
            return
        indices = self._flat_indices(np.full(self.n_envs, candidate_row), env_indices)
        self.sum_tree.update(indices, np.full(self.n_envs, self.max_priority, dtype=np.float32))

    def reward_scale(self, eps: float = 1e-8) -> float:
        """Mean |reward| over everything stored; the value target is divided by it."""
        stored = self.transition_ids >= 0
        if not stored.any():
            return eps
        return max(float(np.abs(self.rewards[stored]).mean()), eps)

    def sample_subtrajectory(self, batch_size: int, horizon: int, include_intermediate: bool):
        indices = self.sum_tree.sample(batch_size)
        rows = indices // self.n_envs
        env_indices = indices % self.n_envs
        start_ids = self.transition_ids[rows]

        offsets = np.arange(horizon, dtype=np.int64)
        step_rows = (rows[:, None] + offsets[None, :]) % self.time_capacity
        # A window can only go stale at its tail; treat any non-contiguous step
        # as terminal so the masks below drop it.
        contiguous = self.transition_ids[step_rows] == (start_ids[:, None] + offsets[None, :])
        step_rewards = self.rewards[step_rows, env_indices[:, None]] * contiguous
        not_dones = (~self.dones[step_rows, env_indices[:, None]] & contiguous).astype(np.float32)

        observations = self._encode_stack(rows, env_indices, start_ids)
        actions = self.actions[step_rows, env_indices[:, None]]
        if include_intermediate:
            # Only next observations are needed: the encoder is fed the first
            # observation and then rolls forward on its own predicted latent, so
            # the intermediate *inputs* are never used and are not materialized.
            next_observations = np.stack(
                [
                    self._encode_stack(
                        (rows + k + 1) % self.time_capacity, env_indices, start_ids + k + 1
                    )
                    for k in range(horizon)
                ],
                axis=1,
            )
        else:
            next_observations = self._encode_stack(
                (rows + horizon) % self.time_capacity, env_indices, start_ids + horizon
            )
            actions = actions[:, 0]

        return MRQSamples(
            observations=self._to_torch(observations),
            actions=self._to_torch(actions),
            next_observations=self._to_torch(next_observations),
            rewards=self._to_torch(step_rewards.astype(np.float32)[..., None]),
            not_dones=self._to_torch(not_dones[..., None]),
            indices=indices,
        )

    def update_priorities(self, indices, priorities) -> None:
        """LAP: the caller has already floored and exponentiated the TD errors."""
        priorities = np.asarray(priorities, dtype=np.float32).reshape(-1)
        self.max_priority = max(self.max_priority, float(priorities.max()))
        self.sum_tree.update(indices, priorities)


if __name__ == "__main__":
    process_start = time.perf_counter()
    args = tyro.cli(Args)
    if args.learner_updates_per_vector_step < 0:
        raise ValueError("learner_updates_per_vector_step must be non-negative")
    if args.replay_ratio is not None:
        if args.replay_ratio < 0:
            raise ValueError("replay_ratio must be non-negative")
        args.learner_updates_per_vector_step = args.replay_ratio * args.num_envs / args.batch_size
    if args.max_training_seconds < 0:
        raise ValueError("max_training_seconds must be non-negative")
    if args.num_envs < 1:
        raise ValueError("num_envs must be positive")
    if args.enc_horizon < 1:
        raise ValueError("enc_horizon must be positive")
    if args.q_horizon < 1:
        raise ValueError("q_horizon must be positive")
    if args.target_network_frequency < 1:
        raise ValueError("target_network_frequency must be positive")
    if args.benchmark_warmup_iterations < 0:
        raise ValueError("benchmark_warmup_iterations cannot be negative")
    if args.benchmark_measure_iterations < 1:
        raise ValueError("benchmark_measure_iterations must be positive")
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
                episodic_life=args.episodic_life,
                reward_clip=args.clip_rewards,
                seed=args.seed,
            )
        )
    elif args.env_backend == "gymnasium":
        envs = gym.vector.SyncVectorEnv(
            [
                make_env(
                    args.env_id,
                    args.seed + i,
                    i,
                    args.capture_video,
                    run_name,
                    args.clip_rewards,
                    args.episodic_life,
                )
                for i in range(args.num_envs)
            ]
        )
    else:
        raise ValueError(f"unsupported environment backend: {args.env_backend}")
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"
    n_actions = int(envs.single_action_space.n)

    # Discrete actions live on the [0, 1] simplex rather than the [-1, 1] cube,
    # so every action-space noise scale is halved.
    exploration_noise = 0.5 * args.exploration_noise
    target_policy_noise = 0.5 * args.target_policy_noise
    noise_clip = 0.5 * args.noise_clip

    encoder = Encoder(
        4, n_actions, args.num_bins, args.zs_dim, args.za_dim, args.zsa_dim, args.enc_hdim
    ).to(device)
    policy = Policy(n_actions, args.gumbel_tau, args.zs_dim, args.policy_hdim).to(device)
    value = Value(args.zsa_dim, args.value_hdim).to(device)
    encoder_target = copy.deepcopy(encoder).requires_grad_(False)
    policy_target = copy.deepcopy(policy).requires_grad_(False)
    value_target = copy.deepcopy(value).requires_grad_(False)

    capturable = args.cudagraphs and not args.compile
    encoder_optimizer = optim.AdamW(
        encoder.parameters(),
        lr=args.encoder_learning_rate,
        weight_decay=args.weight_decay,
        capturable=capturable,
    )
    value_optimizer = optim.AdamW(
        value.parameters(),
        lr=args.value_learning_rate,
        weight_decay=args.weight_decay,
        capturable=capturable,
    )
    policy_optimizer = optim.AdamW(
        policy.parameters(),
        lr=args.policy_learning_rate,
        weight_decay=args.weight_decay,
        capturable=capturable,
    )
    two_hot = TwoHot(args.bin_lower, args.bin_upper, args.num_bins).to(device)

    rb = MRQReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        n_envs=args.num_envs,
        n_step=args.q_horizon,
        gamma=args.gamma,
        alpha=args.prioritized_replay_alpha,
        beta=0.0,  # LAP does not use importance-sampling weights
        enc_horizon=args.enc_horizon,
        q_horizon=args.q_horizon,
        min_priority=args.min_priority,
    )

    def policy_fn(observations: torch.Tensor, random_prob: torch.Tensor) -> torch.Tensor:
        relaxed = policy.act(encoder.zs(observations.float()))
        relaxed = relaxed + torch.randn_like(relaxed) * exploration_noise
        greedy_actions = relaxed.argmax(1)
        random_actions = torch.randint(n_actions, greedy_actions.shape, device=greedy_actions.device)
        explore = torch.rand(greedy_actions.shape, device=greedy_actions.device) < random_prob
        return torch.where(explore, random_actions, greedy_actions)

    def update_encoder(data: TensorDict, done_weight: torch.Tensor):
        observations = data["observations"]
        with torch.no_grad():
            flat_next = data["next_observations"].flatten(0, 1).float()
            if args.data_augmentation:
                flat_next = random_shift_augmentation(flat_next)
            target_zs = encoder_target.zs(flat_next).view(observations.shape[0], -1, args.zs_dim)

        observations = observations.float()
        if args.data_augmentation:
            observations = random_shift_augmentation(observations)

        loss = encoder_loss(
            encoder,
            two_hot,
            observations,
            F.one_hot(data["actions"], n_actions).float(),
            data["rewards"],
            data["not_dones"],
            target_zs,
            args.dyn_weight,
            args.reward_weight,
            done_weight,
        )
        # set_to_none=False keeps the gradient buffers at fixed addresses, which
        # a captured graph replays into.
        encoder_optimizer.zero_grad(set_to_none=False)
        loss.backward()
        encoder_optimizer.step()
        return loss.detach()

    def update_rl(data: TensorDict, reward_scale: torch.Tensor, target_reward_scale: torch.Tensor):
        observations = data["observations"].float()
        next_observations = data["next_observations"].float()
        if args.data_augmentation:
            observations = random_shift_augmentation(observations)
            next_observations = random_shift_augmentation(next_observations)
        actions = F.one_hot(data["actions"], n_actions).float()
        ms_reward, term_discount = multi_step_reward(data["rewards"], data["not_dones"], args.gamma)

        with torch.no_grad():
            next_zs = encoder_target.zs(next_observations)
            noise = (torch.randn_like(actions) * target_policy_noise).clamp(-noise_clip, noise_clip)
            next_action = realign_discrete(policy_target.act(next_zs) + noise)
            next_zsa = encoder_target(next_zs, next_action)
            next_q = value_target(next_zsa).min(1, keepdim=True).values
            q_target = scaled_q_target(
                ms_reward, term_discount, next_q, reward_scale, target_reward_scale
            )

            # The encoder is trained by its own objective only; the value and
            # policy losses see zs / zsa as constants.
            zs = encoder.zs(observations)
            zsa = encoder(zs, actions)

        q = value(zsa)
        value_loss = F.smooth_l1_loss(q, q_target.expand(-1, 2))

        # The policy loss backward below leaves gradients on the value and
        # encoder parameters; both are zeroed before their own backward, so
        # nothing leaks between updates.
        value_optimizer.zero_grad(set_to_none=False)
        value_loss.backward()
        nn.utils.clip_grad_norm_(value.parameters(), args.value_grad_clip)
        value_optimizer.step()

        policy_action, pre_activ = policy(zs)
        q_policy = value(encoder(zs, policy_action))
        policy_loss = -q_policy.mean() + args.pre_activ_weight * pre_activ.pow(2).mean()

        policy_optimizer.zero_grad(set_to_none=False)
        policy_loss.backward()
        policy_optimizer.step()

        with torch.no_grad():
            priority = lap_priority(q, q_target, args.min_priority, args.prioritized_replay_alpha)
        return value_loss.detach(), policy_loss.detach(), q.detach().mean(), priority

    if args.compile:
        # CuLE observation storage is updated in place, so do not use
        # reduce-overhead's implicit CUDA graphs here.
        policy_fn = torch.compile(policy_fn, mode=None, fullgraph=True)
        update_encoder = torch.compile(update_encoder, mode=None)
        update_rl = torch.compile(update_rl, mode=None)

    if args.cudagraphs:
        # CudaGraphModule copies inputs into static buffers and clones outputs.
        # The hard target syncs mutate module tensors in place, so graph replays
        # observe them.
        policy_fn = CudaGraphModule(policy_fn, warmup=20)
        update_encoder = CudaGraphModule(update_encoder, warmup=20)
        update_rl = CudaGraphModule(update_rl, warmup=20)

    random_prob_tensor = torch.ones((), device=device)
    done_weight_tensor = torch.zeros((), device=device)
    reward_scale_tensor = torch.ones((), device=device)
    target_reward_scale_tensor = torch.zeros((), device=device)
    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    reset_result = envs.reset(seed=args.seed) if args.env_backend != "envpool" else envs.reset()
    obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result
    rb.initialize(obs)
    global_step = 0
    update_budget = 0.0
    learner_updates = 0
    last_encoder_loss = torch.zeros((), device=device)
    episode_stats = EpisodeStats(args.solve_window, args.solve_reward)
    next_log_step = max(10000, args.num_envs)
    num_vector_steps = int(np.ceil(args.total_timesteps / args.num_envs))
    if args.benchmark:
        num_vector_steps = args.benchmark_warmup_iterations + args.benchmark_measure_iterations
    benchmark_start = None
    benchmark_start_step = None
    benchmark_start_updates = None
    for vector_step in range(num_vector_steps):
        if args.benchmark and vector_step == args.benchmark_warmup_iterations:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            benchmark_start = time.perf_counter()
            benchmark_start_step = global_step
            benchmark_start_updates = learner_updates

        if args.max_training_seconds and time.time() - start_time >= args.max_training_seconds:
            break

        # ALGO LOGIC: act from the relaxed policy plus Gaussian action noise,
        # uniformly at random until the buffer has filled to learning_starts.
        random_prob_tensor.fill_(1.0 if global_step < args.learning_starts else 0.0)
        if args.compile:
            torch.compiler.cudagraph_mark_step_begin()
        with torch.no_grad():
            actions = policy_fn(to_tensor(obs, device), random_prob_tensor)

        # TRY NOT TO MODIFY: execute the game and log data.
        step_result = step_env(envs, actions)
        if len(step_result) == 5:
            next_obs, rewards, terminations, truncations, infos = step_result
        else:
            next_obs, rewards, terminations, infos = step_result
            truncations = np.zeros_like(np.asarray(terminations), dtype=bool)
        transition_dones = done_tensor(terminations, truncations, device).bool()
        global_step += args.num_envs

        solved = False
        if not args.benchmark:
            solved = episode_stats.update(
                completed_episode_infos(infos, transition_dones), global_step, writer
            )

        rb.add(next_obs, actions, rewards, transition_dones)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if (
            global_step > args.learning_starts
            and len(rb) >= args.batch_size
            and rb.sum_tree.total > 0
        ):
            update_budget += args.learner_updates_per_vector_step
            num_updates = int(update_budget)
            update_budget -= num_updates
            for _ in range(num_updates):
                learner_updates += 1
                if (learner_updates - 1) % args.target_network_frequency == 0:
                    # Copy in place so captured graphs see the new weights.
                    with torch.no_grad():
                        for target_module, online_module in (
                            (encoder_target, encoder),
                            (policy_target, policy),
                            (value_target, value),
                        ):
                            for target_param, param in zip(
                                target_module.parameters(), online_module.parameters()
                            ):
                                target_param.copy_(param)
                            for target_buf, buf in zip(
                                target_module.buffers(), online_module.buffers()
                            ):
                                target_buf.copy_(buf)
                    target_reward_scale_tensor.copy_(reward_scale_tensor)
                    reward_scale_tensor.fill_(rb.reward_scale())
                    # The encoder's updates are batched between target syncs
                    # rather than interleaved, so it still averages one update
                    # per value update but only pays the sampling cost in bursts.
                    done_weight_tensor.fill_(args.done_weight if rb.env_terminates else 0.0)
                    for _ in range(args.target_network_frequency):
                        data = rb.sample_subtrajectory(args.batch_size, args.enc_horizon, True)
                        batch = TensorDict(
                            {
                                "observations": data.observations,
                                "actions": data.actions,
                                "next_observations": data.next_observations,
                                "rewards": data.rewards,
                                "not_dones": data.not_dones,
                            },
                            batch_size=[args.batch_size],
                            device=device,
                        )
                        if args.compile:
                            torch.compiler.cudagraph_mark_step_begin()
                        last_encoder_loss = update_encoder(batch, done_weight_tensor)

                data = rb.sample_subtrajectory(args.batch_size, args.q_horizon, False)
                batch = TensorDict(
                    {
                        "observations": data.observations,
                        "actions": data.actions,
                        "next_observations": data.next_observations,
                        "rewards": data.rewards,
                        "not_dones": data.not_dones,
                    },
                    batch_size=[args.batch_size],
                    device=device,
                )
                if args.compile:
                    torch.compiler.cudagraph_mark_step_begin()
                last_value_loss, last_policy_loss, last_q_value, priority = update_rl(
                    batch, reward_scale_tensor, target_reward_scale_tensor
                )
                rb.update_priorities(data.indices, priority.cpu().numpy())

            if not args.benchmark and global_step >= next_log_step and num_updates:
                sps = int(global_step / (time.time() - start_time))
                writer.add_scalar("losses/encoder_loss", last_encoder_loss.item(), global_step)
                writer.add_scalar("losses/value_loss", last_value_loss.item(), global_step)
                writer.add_scalar("losses/policy_loss", last_policy_loss.item(), global_step)
                writer.add_scalar("losses/q_values", last_q_value.item(), global_step)
                writer.add_scalar("charts/reward_scale", reward_scale_tensor.item(), global_step)
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
                next_log_step = global_step + max(10000, args.num_envs)
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
            "algorithm": "mrq",
            "backend": args.env_backend,
            "batch_size": args.batch_size,
            "benchmark": "full_training_loop",
            "compile": args.compile,
            "cudagraphs": args.cudagraphs,
            "data_augmentation": args.data_augmentation,
            "enc_horizon": args.enc_horizon,
            "env_device": str(getattr(envs, "device", "cpu")),
            "env_id": args.env_id,
            "implementation": "torchcompile",
            "learner_updates": measured_updates,
            "measure_iterations": args.benchmark_measure_iterations,
            "measured_seconds": measured_seconds,
            "measured_steps": measured_steps,
            "num_envs": args.num_envs,
            "peak_cuda_memory_mb": (
                torch.cuda.max_memory_allocated(device) / (1024**2) if device.type == "cuda" else 0.0
            ),
            "process_seconds": benchmark_end - process_start,
            "q_horizon": args.q_horizon,
            "replay_backend": "numpy_frame_efficient_lap_subtrajectories",
            "replay_ratio": measured_updates * args.batch_size / max(measured_steps, 1),
            "schema_version": 1,
            "sps": measured_steps / measured_seconds,
            "ups": measured_updates / measured_seconds,
            "warmup_iterations": args.benchmark_warmup_iterations,
        }
        print(f"BENCHMARK_RESULT {json.dumps(result, sort_keys=True)}", flush=True)
    else:
        elapsed = time.time() - start_time
        print("SPS:", int(global_step / elapsed))
        print("learner updates:", learner_updates)
        print("UPS:", learner_updates / elapsed)
        print("effective UTD:", learner_updates / max(global_step - args.learning_starts, 1))
        print(
            "replay ratio:",
            learner_updates * args.batch_size / max(global_step - args.learning_starts, 1),
        )
        episode_stats.print_summary()

    if args.save_model and not args.benchmark:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        model_data = {
            "model_weights": {
                "encoder": encoder.state_dict(),
                "policy": policy.state_dict(),
                "value": value.state_dict(),
            },
            "args": vars(args),
        }
        torch.save(model_data, model_path)
        print(f"model saved to {model_path}")

    envs.close()
    if writer is not None:
        writer.close()
