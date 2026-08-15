# DiscoRL / Disco103: Discovering state-of-the-art reinforcement learning
# algorithms (Oh et al., Nature 2025). Ported from the official implementation
# (https://github.com/google-deepmind/disco_rl, Apache-2.0), which is JAX/Haiku;
# this is an independent PyTorch reimplementation of the same computation. No
# code was copied. `tests/test_disco_equivalence.py` diffs it against a NumPy
# transcription of the Haiku modules, and against the real JAX implementation
# when jax/haiku/rlax happen to be installed.
#
# What makes this different from every other trainer here: the update rule was
# not written by a person. A meta-network -- an LSTM with 754,778 published
# weights (`disco_103.npz`) -- reads a rollout's rewards, terminations, policy,
# value and advantage statistics and emits three prediction targets:
#
#   pi_hat  [T, B, A]  the policy target      -> KL(pi_hat  || logits)
#   y_hat   [T, B, Y]  a "flat" prediction    -> KL(y_hat   || y)
#   z_hat   [T, B, Y]  an action prediction   -> KL(z_hat   || z[a])
#
# The agent's loss is just those three KLs, plus a 1-step auxiliary policy
# prediction and a categorical value loss driven by the meta-net's own TD error.
# The meta-network is frozen here (meta-training is a separate, much larger
# job): it is evaluated under no_grad and only its LSTM state advances, so from
# the agent's point of view it is a fixed, learned objective function.
#
# Structure: the rollout scaffolding follows ppo_atari.py (CleanRL, MIT; license
# in cleanrl/LICENSE.md); the value machinery (Retrace from Q and V, categorical
# values over 601 bins with a signed-hyperbolic transform, EMA-normalized
# advantages) follows the reference. Supports gymnasium, cule, and envpool.
#
# The published weights are not vendored. They are fetched once to
# ~/.cache/cule-disco/disco_103.npz, or supplied with --meta-weights.
#
# CONFIRMED against the official implementation:
# `tests/crosscheck/check_disco.py` runs google-deepmind/disco_rl's real Haiku
# `meta_nets.LSTM` with the published disco_103.npz loaded and diffs it against
# this file. 21/21 components match on CPU and CUDA to float32 precision
# (<= 1.2e-6): pi_hat, y_hat and z_hat plus the lifetime LSTM state over three
# consecutive calls, the parameter names and the 754,778-parameter count, the
# 27-wide constructed input, and 15 fields of get_settings_disco().
import json
import math
import os
import random
import sys
import time
import urllib.request
from dataclasses import dataclass
from typing import NamedTuple, Sequence

try:
    import envpool
except ImportError:
    envpool = None
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tyro
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
from cleanrl_utils.episode_stats import EpisodeStats

META_WEIGHTS_URL = (
    "https://raw.githubusercontent.com/google-deepmind/disco_rl/main/"
    "disco_rl/update_rules/weights/disco_103.npz"
)
# Fixed by the published meta-parameters: the y/z prediction heads are 600-wide
# because the meta-net's y_net and z_net take 600 inputs, and the constructed
# meta input is 27-wide (23 base features + two 2-wide action-conditional
# poolings) because the per-trajectory LSTM's kernel is (27 + 256, 4 * 256).
PREDICTION_SIZE = 600
META_INPUT_SIZE = 27
ACTION_CONDITIONAL_INPUT_SIZE = 9


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
    total_timesteps: int = 10000000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    max_abs_update: float = 1.0
    """element-wise clip on the Adam update, before the learning rate is applied"""
    adam_beta1: float = 0.9
    """Adam first-moment decay"""
    adam_beta2: float = 0.999
    """Adam second-moment decay"""
    adam_eps: float = 1e-8
    """Adam epsilon"""
    num_envs: int = 32
    """the number of parallel game environments"""
    num_steps: int = 29
    """environment transitions per rollout (T); one more observation is stored to bootstrap"""
    meta_weights: str = ""
    """path to disco_103.npz; empty downloads it to ~/.cache/cule-disco"""

    discount: float = 0.997
    """the discount factor used by the value function"""
    td_lambda: float = 0.95
    """the Retrace trace parameter"""
    num_bins: int = 601
    """bins of the categorical value function"""
    max_abs_value: float = 300.0
    """the largest absolute return the categorical value function can represent"""
    target_params_coeff: float = 0.9
    """Polyak coefficient of the target network: target = coeff * target + (1 - coeff) * online"""
    pi_cost: float = 1.0
    """weight of the discovered policy target's KL"""
    y_cost: float = 1.0
    """weight of the discovered flat prediction's KL"""
    z_cost: float = 1.0
    """weight of the discovered action prediction's KL"""
    aux_policy_cost: float = 1.0
    """weight of the 1-step auxiliary policy prediction"""
    value_cost: float = 0.2
    """weight of the categorical value loss"""
    moving_average_decay: float = 0.99
    """decay of the advantage and TD normalizers"""
    moving_average_eps: float = 1e-6
    """epsilon of the advantage and TD normalizers"""

    torso_dense: tuple[int, ...] = (512,)
    """widths of the agent torso's MLP, applied after the convolutional stack"""
    head_w_init_std: float = 1e-2
    """truncated-normal stddev of the agent's output heads"""
    head_mlp_hiddens: tuple[int, ...] = (256,)
    """hidden widths of the action-conditional model heads"""
    model_lstm_size: int = 256
    """width of the action-conditional model's LSTM"""

    replay_capacity: int = 256
    """trajectories held in the FIFO replay; equal to num-envs makes it on-policy"""
    batch_size: int = 32
    """trajectories per learner step"""
    learner_steps_per_rollout: int = 1
    """gradient steps taken after each rollout"""
    max_training_seconds: float = 0.0
    """wall-clock training limit; zero disables it"""
    solve_reward: float | None = None
    """stop when the moving episodic return reaches this value"""
    solve_window: int = 20
    """number of completed episodes in the solve moving average"""
    benchmark: bool = False
    """run a fixed warmup/measurement window and print a JSON benchmark result"""
    benchmark_warmup_iterations: int = 3
    """rollouts excluded from benchmark timing"""
    benchmark_measure_iterations: int = 8
    """rollouts included in benchmark timing"""


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
# rlax / Haiku primitives, reimplemented
# ---------------------------------------------------------------------------

def signed_hyperbolic(x: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    """h(x): squashes returns so one value head covers many reward scales."""
    return torch.sign(x) * (torch.sqrt(torch.abs(x) + 1.0) - 1.0) + eps * x


def signed_parabolic(x: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    """The exact inverse of `signed_hyperbolic`."""
    root = (torch.sqrt(1.0 + 4.0 * eps * (torch.abs(x) + 1.0 + eps)) - 1.0) / (2.0 * eps)
    return torch.sign(x) * (root * root - 1.0)


def signed_logp1(x: torch.Tensor) -> torch.Tensor:
    """sign(x) * log(|x| + 1); the meta-net's compressor for unbounded inputs."""
    return torch.sign(x) * torch.log1p(torch.abs(x))


def transform_to_2hot(scalar, min_value: float, max_value: float, num_bins: int):
    """Distribute a scalar over the two nearest of `num_bins` evenly spaced bins."""
    scalar = scalar.clamp(min_value, max_value)
    position = (scalar - min_value) / (max_value - min_value) * (num_bins - 1)
    lower = position.floor()
    upper = position.ceil()
    lower_value = lower / (num_bins - 1.0) * (max_value - min_value) + min_value
    upper_value = upper / (num_bins - 1.0) * (max_value - min_value) + min_value
    p_lower = (upper_value - scalar) / (upper_value - lower_value + 1e-5)
    p_upper = 1.0 - p_lower

    probs = torch.zeros(*scalar.shape, num_bins, device=scalar.device, dtype=scalar.dtype)
    probs.scatter_add_(-1, lower.long().clamp(0, num_bins - 1).unsqueeze(-1), p_lower.unsqueeze(-1))
    probs.scatter_add_(-1, upper.long().clamp(0, num_bins - 1).unsqueeze(-1), p_upper.unsqueeze(-1))
    return probs


def transform_from_2hot(probs, min_value: float, max_value: float, num_bins: int):
    support = torch.linspace(min_value, max_value, num_bins, device=probs.device, dtype=probs.dtype)
    return (probs * support).sum(-1)


def categorical_kl_divergence(p_logits: torch.Tensor, q_logits: torch.Tensor) -> torch.Tensor:
    """KL(softmax(p) || softmax(q)), floored at zero as in rlax."""
    p = F.softmax(p_logits, dim=-1)
    log_p = F.log_softmax(p_logits, dim=-1)
    log_q = F.log_softmax(q_logits, dim=-1)
    return (p * (log_p - log_q)).sum(-1).clamp_min(0.0)


def batch_lookup(table: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """Index axis 2 of a [T, B, A, ...] tensor with a [T, B] action array."""
    trailing = table.shape[3:]
    expanded = index.reshape(*index.shape, 1, *([1] * len(trailing)))
    expanded = expanded.expand(*index.shape, 1, *trailing)
    return table.gather(2, expanded).squeeze(2)


def td_pair(x: torch.Tensor) -> torch.Tensor:
    """Concatenate the value at t with the value at t+1 along the feature axis.

    The meta-network is never told how to compute a TD error; giving it both
    ends of every quantity is what lets it discover one.
    """
    return torch.cat([x[:-1], x[1:]], dim=-1)


def haiku_truncated_normal_(tensor: torch.Tensor, stddev: float) -> torch.Tensor:
    """Haiku's TruncatedNormal: N(0, s) cut at +-2s, rescaled to keep stddev s."""
    scale = stddev / 0.87962566103423978
    return nn.init.trunc_normal_(tensor, mean=0.0, std=scale, a=-2.0 * scale, b=2.0 * scale)


def haiku_linear(in_features: int, out_features: int, stddev: float | None = None) -> nn.Linear:
    """A Linear with Haiku's defaults: truncated normal 1/sqrt(fan_in), zero bias."""
    layer = nn.Linear(in_features, out_features)
    haiku_truncated_normal_(layer.weight, stddev if stddev is not None else 1.0 / math.sqrt(in_features))
    nn.init.zeros_(layer.bias)
    return layer


class HaikuMLP(nn.Module):
    """hk.nets.MLP: ReLU between layers, and none on the output."""

    def __init__(self, in_features: int, sizes: Sequence[int], stddev: float | None = None):
        super().__init__()
        layers = []
        width = in_features
        for size in sizes:
            layers.append(haiku_linear(width, size, stddev))
            width = size
        self.layers = nn.ModuleList(layers)
        self.out_features = width

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for index, layer in enumerate(self.layers):
            x = layer(x)
            if index < len(self.layers) - 1:
                x = F.relu(x)
        return x


class HaikuLSTMCell(nn.Module):
    """hk.LSTM: one Linear over [input, hidden], gates ordered i, g, f, o.

    PyTorch's LSTMCell orders them i, f, g, o, splits the kernel into two
    matrices and has no forget-bias offset, so the published kernels cannot be
    loaded into it. This is the Haiku formulation, verbatim.
    """

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.linear = haiku_linear(input_size + hidden_size, 4 * hidden_size)

    def forward(self, x: torch.Tensor, hidden: torch.Tensor, cell: torch.Tensor):
        gates = self.linear(torch.cat([x, hidden], dim=-1))
        i, g, f, o = gates.chunk(4, dim=-1)
        # The +1 forget bias comes from Sonnet and is part of the trained model.
        f = torch.sigmoid(f + 1.0)
        new_cell = f * cell + torch.sigmoid(i) * torch.tanh(g)
        new_hidden = torch.sigmoid(o) * torch.tanh(new_cell)
        return new_hidden, new_cell


class Conv1DNet(nn.Module):
    """A stack of hk.Conv1D(kernel_shape=1) blocks over the action axis.

    Each block concatenates every action's features with the mean over actions
    before the 1x1 convolution, so actions can see the others without giving up
    permutation equivariance. With kernel width 1 the convolution is a Linear on
    the last axis.
    """

    def __init__(self, in_channels: int, channels: Sequence[int]):
        super().__init__()
        layers = []
        width = in_channels
        for size in channels:
            layers.append(haiku_linear(2 * width, size))
            width = size
        self.layers = nn.ModuleList(layers)
        self.out_channels = width

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            pooled = x.mean(dim=2, keepdim=True).expand_as(x)
            x = F.relu(layer(torch.cat([x, pooled], dim=-1)))
        return x


class MovingAverage(nn.Module):
    """Scalar EMA of the first two moments, debiased exactly as Adam is."""

    def __init__(self, decay: float = 0.99, eps: float = 1e-6):
        super().__init__()
        self.decay = float(decay)
        self.eps = float(eps)
        self.register_buffer("moment1", torch.zeros(()))
        self.register_buffer("moment2", torch.zeros(()))
        self.register_buffer("decay_product", torch.ones(()))

    @torch.no_grad()
    def update(self, value: torch.Tensor) -> None:
        self.moment1.mul_(self.decay).add_((1.0 - self.decay) * value.mean())
        self.moment2.mul_(self.decay).add_((1.0 - self.decay) * value.pow(2).mean())
        self.decay_product.mul_(self.decay)

    def normalize(self, value: torch.Tensor, subtract_mean: bool = True, root_eps: float = 1e-12):
        debias = 1.0 / (1.0 - self.decay_product)
        mean = self.moment1 * debias
        variance = (self.moment2 * debias - mean.pow(2)).clamp_min(0.0)
        scale = torch.sqrt(variance + root_eps) + self.eps
        return (value - mean) / scale if subtract_mean else value / scale


# ---------------------------------------------------------------------------
# Value machinery
# ---------------------------------------------------------------------------

class ValueOuts(NamedTuple):
    value: torch.Tensor            # [T+1, B] online state values
    target_value: torch.Tensor     # [T+1, B]
    adv: torch.Tensor              # [T, B]
    normalized_adv: torch.Tensor   # [T, B]
    qv_adv: torch.Tensor           # [T+1, B, A]
    normalized_qv_adv: torch.Tensor
    target_q_value: torch.Tensor   # [T+1, B, A]
    q_target: torch.Tensor         # [T, B]
    q_td: torch.Tensor             # [T, B]


def value_logits_to_scalar(logits, max_abs_value: float, nonlinear_transform: bool = True):
    """601 bin logits -> a scalar return, undoing the signed-hyperbolic squash."""
    num_bins = logits.shape[-1]
    scalar = transform_from_2hot(
        F.softmax(logits, dim=-1), -max_abs_value, max_abs_value, num_bins
    )
    return signed_parabolic(scalar) if nonlinear_transform else scalar


def retrace_from_q_and_v(q_t, v_t, r_t, discount_t, c_t):
    """G_t = r_t + d_t * (v_t - c_t * q_t + c_t * G_{t+1}), run backwards.

    Shapes follow rlax: `v_t`, `r_t` and `discount_t` cover [1..K]; `q_t` and
    `c_t` cover [1..K-1]. Returns the K targets G_0..G_{K-1}.
    """
    returns = [r_t[-1] + discount_t[-1] * v_t[-1]]
    for index in reversed(range(q_t.shape[0])):
        returns.insert(
            0,
            r_t[index]
            + discount_t[index] * (v_t[index] - c_t[index] * q_t[index] + c_t[index] * returns[0]),
        )
    return torch.stack(returns, dim=0)


def estimate_q_values(
    rewards,
    actions,
    env_discounts,
    rho,
    values,
    target_values,
    q_values,
    target_q_values,
    discount: float,
    lambda_: float,
):
    """Retrace targets for Q, and every derived quantity the meta-net reads."""
    q_a = batch_lookup(q_values[:-1], actions)
    target_q_a = batch_lookup(target_q_values[:-1], actions)

    discounts = env_discounts * discount
    zeros = torch.zeros_like(rewards[:1])
    # Retrace is defined from t = 1, so a dummy zeroth step is prepended and the
    # corresponding target dropped afterwards.
    padded_rewards = torch.cat([zeros, rewards], dim=0)
    padded_discounts = torch.cat([zeros, discounts], dim=0)

    clipped_rho = rho.clamp(max=1.0)
    q_target = retrace_from_q_and_v(
        target_q_a, target_values, padded_rewards, padded_discounts, lambda_ * clipped_rho
    )[1:]

    qv_adv = target_q_values - target_values.unsqueeze(-1)
    adv = q_target - target_values[:-1]
    q_td = q_target - q_a
    return ValueOuts(
        value=values,
        target_value=target_values,
        adv=adv,
        normalized_adv=torch.zeros_like(adv),
        qv_adv=qv_adv,
        normalized_qv_adv=torch.zeros_like(qv_adv),
        target_q_value=target_q_values,
        q_target=q_target,
        q_td=q_td,
    )


def importance_weight(pi_logits, mu_logits, actions):
    log_pi = F.log_softmax(pi_logits, dim=-1).gather(-1, actions.unsqueeze(-1)).squeeze(-1)
    log_mu = F.log_softmax(mu_logits, dim=-1).gather(-1, actions.unsqueeze(-1)).squeeze(-1)
    return torch.exp(log_pi - log_mu)


# ---------------------------------------------------------------------------
# The discovered meta-network
# ---------------------------------------------------------------------------

class MetaInputs(NamedTuple):
    """Everything the meta-network is allowed to see about a rollout."""

    actions: torch.Tensor            # [T+1, B]
    rewards: torch.Tensor            # [T, B]
    is_terminal: torch.Tensor        # [T, B] float
    logits: torch.Tensor             # [T+1, B, A] online policy
    behaviour_logits: torch.Tensor   # [T+1, B, A]
    target_logits: torch.Tensor      # [T+1, B, A]
    y: torch.Tensor                  # [T+1, B, Y]
    target_y: torch.Tensor           # [T+1, B, Y]
    z: torch.Tensor                  # [T+1, B, A, Y]
    target_z: torch.Tensor           # [T+1, B, A, Y]
    v_scalar: torch.Tensor           # [T+1, B]
    adv: torch.Tensor                # [T, B]
    normalized_adv: torch.Tensor     # [T, B]
    q: torch.Tensor                  # [T+1, B, A] target Q
    qv_adv: torch.Tensor             # [T+1, B, A]
    normalized_qv_adv: torch.Tensor  # [T+1, B, A]


class MetaInputEncoder(nn.Module):
    """`_construct_input`: 16 base features and 8 action-conditional ones.

    The list below is the published input option, in order. Each entry is
    flattened to [T, B, -1] and concatenated; the action-conditional entries are
    flattened to [T, B, A, -1], passed through a 1x1 conv stack, and summarized
    twice -- averaged over actions and looked up at the taken action.
    """

    def __init__(self, embedding_size: Sequence[int], policy_channels: Sequence[int]):
        super().__init__()
        self.y_net = HaikuMLP(PREDICTION_SIZE, embedding_size)
        self.z_net = HaikuMLP(PREDICTION_SIZE, embedding_size)
        self.policy_net = Conv1DNet(ACTION_CONDITIONAL_INPUT_SIZE, policy_channels)

    def forward(self, inputs: MetaInputs):
        horizon, batch = inputs.rewards.shape
        actions = inputs.actions[:-1]                       # [T, B]
        policy = F.softmax(inputs.logits, dim=-1)           # [T+1, B, A]
        n_actions = policy.shape[-1]

        y_emb = self.y_net(F.softmax(inputs.y, dim=-1))                 # [T+1, B, E]
        target_y_emb = self.y_net(F.softmax(inputs.target_y, dim=-1))
        z_emb = self.z_net(F.softmax(inputs.z, dim=-1))                 # [T+1, B, A, E]
        target_z_emb = self.z_net(F.softmax(inputs.target_z, dim=-1))

        def weighted(x):  # pi_weighted_avg
            return (x * policy.unsqueeze(-1)).sum(dim=2)

        base = [
            batch_lookup(F.softmax(inputs.logits, -1)[:-1], actions),            # 1  pi(a)
            batch_lookup(F.softmax(inputs.behaviour_logits, -1)[:-1], actions),  # 2  mu(a)
            signed_logp1(inputs.rewards),                                        # 3  reward
            1.0 - inputs.is_terminal,                                            # 4  discount
            td_pair(signed_logp1(inputs.v_scalar.unsqueeze(-1))),                # 5  V, V'
            signed_logp1(inputs.adv.unsqueeze(-1)),                              # 6  advantage
            inputs.normalized_adv.unsqueeze(-1),                                 # 7
            batch_lookup(F.softmax(inputs.target_logits, -1)[:-1], actions),     # 8  target pi(a)
            td_pair(y_emb),                                                      # 9
            td_pair(target_y_emb),                                               # 10
            batch_lookup(z_emb[:-1], actions),                                   # 11
            td_pair(weighted(z_emb)),                                            # 12
            td_pair(z_emb.max(dim=2).values),                                    # 13
            batch_lookup(target_z_emb[:-1], actions),                            # 14
            td_pair(weighted(target_z_emb)),                                     # 15
            td_pair(target_z_emb.max(dim=2).values),                             # 16
        ]
        base = [feature.reshape(horizon, batch, -1) for feature in base]

        action_conditional = [
            F.softmax(inputs.logits, -1)[:-1],
            F.softmax(inputs.behaviour_logits, -1)[:-1],
            F.softmax(inputs.target_logits, -1)[:-1],
            z_emb[:-1],
            target_z_emb[:-1],
            signed_logp1(inputs.q.unsqueeze(-1))[:-1],
            signed_logp1(inputs.qv_adv.unsqueeze(-1))[:-1],
            inputs.normalized_qv_adv.unsqueeze(-1)[:-1],
            F.one_hot(actions, n_actions).to(policy.dtype).unsqueeze(-1),
        ]
        action_conditional = [
            feature.reshape(horizon, batch, n_actions, -1) for feature in action_conditional
        ]
        policy_emb = self.policy_net(torch.cat(action_conditional, dim=-1))  # [T, B, A, C]

        base.append(policy_emb.mean(dim=2))
        base.append(batch_lookup(policy_emb, actions))
        return torch.cat(base, dim=-1), policy_emb


class DiscoMetaNet(nn.Module):
    """Disco103: the discovered objective, as a frozen network.

    Two recurrences. A per-trajectory LSTM runs *backwards* over the rollout,
    which is how a bootstrapped target gets propagated back through time without
    anyone writing down a lambda-return. A second, per-lifetime LSTM consumes one
    pooled vector per learner step and modulates the first one multiplicatively,
    which is how the rule can change its own behaviour as training progresses.
    """

    def __init__(
        self,
        hidden_size: int = 256,
        embedding_size: Sequence[int] = (16, 1),
        policy_channels: Sequence[int] = (16, 2),
        policy_target_channels: Sequence[int] = (16,),
        meta_hidden_size: int = 128,
        meta_embedding_size: Sequence[int] = (16,),
        meta_pred_embedding_size: Sequence[int] = (16, 1),
        meta_policy_channels: Sequence[int] = (16, 2),
    ):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.encoder = MetaInputEncoder(embedding_size, policy_channels)
        self.trajectory_lstm = HaikuLSTMCell(META_INPUT_SIZE, hidden_size)
        self.state_linear = haiku_linear(meta_hidden_size, hidden_size)
        self.emb_linear = haiku_linear(hidden_size, 1)
        self.y_head = haiku_linear(hidden_size, PREDICTION_SIZE)
        self.z_head = haiku_linear(hidden_size, PREDICTION_SIZE)
        self.policy_target_net = Conv1DNet(
            hidden_size + policy_channels[-1], policy_target_channels
        )
        self.pi_head = haiku_linear(policy_target_channels[-1], 1)

        self.meta_encoder = MetaInputEncoder(meta_pred_embedding_size, meta_policy_channels)
        self.meta_embed = HaikuMLP(
            META_INPUT_SIZE + 1 + meta_pred_embedding_size[-1], meta_embedding_size
        )
        self.meta_lstm = HaikuLSTMCell(meta_embedding_size[-1], meta_hidden_size)
        self.register_buffer("meta_hidden", torch.zeros(meta_hidden_size))
        self.register_buffer("meta_cell", torch.zeros(meta_hidden_size))

    def unroll_backwards(self, x: torch.Tensor, should_reset: torch.Tensor) -> torch.Tensor:
        """hk.ResetCore + hk.dynamic_unroll(reverse=True) over the time axis."""
        horizon, batch, _ = x.shape
        hidden = x.new_zeros(batch, self.hidden_size)
        cell = x.new_zeros(batch, self.hidden_size)
        outputs = [None] * horizon
        for step in reversed(range(horizon)):
            # Reset before the step, so a terminal transition stops the backward
            # trace from carrying the next episode's bootstrap into this one.
            keep = (1.0 - should_reset[step]).unsqueeze(-1)
            hidden, cell = self.trajectory_lstm(x[step], hidden * keep, cell * keep)
            outputs[step] = hidden
        return torch.stack(outputs, dim=0)

    def forward(self, inputs: MetaInputs):
        x, policy_emb = self.encoder(inputs)
        x = self.unroll_backwards(x, inputs.is_terminal)

        # Condition on the lifetime state as it was *before* this step.
        x = x * self.state_linear(self.meta_hidden)

        meta_input_emb = self.emb_linear(x)
        y_hat = self.y_head(x)
        z_hat = self.z_head(x)

        n_actions = inputs.logits.shape[-1]
        w = x.unsqueeze(2).expand(-1, -1, n_actions, -1)
        w = torch.cat([w, policy_emb], dim=-1)
        pi_hat = self.pi_head(self.policy_target_net(w)).squeeze(-1)

        meta_inputs, _ = self.meta_encoder(inputs)
        pooled = self.meta_embed(
            torch.cat(
                [meta_inputs, meta_input_emb, self.meta_encoder.y_net(F.softmax(y_hat, dim=-1))],
                dim=-1,
            )
        ).mean(dim=(0, 1))
        hidden, cell = self.meta_lstm(pooled, self.meta_hidden, self.meta_cell)
        self.meta_hidden.copy_(hidden)
        self.meta_cell.copy_(cell)
        return pi_hat, y_hat, z_hat

    def load_published_weights(self, arrays) -> None:
        """Load `disco_103.npz` (Haiku parameter paths) into this module."""

        def linear(module: nn.Linear, prefix: str) -> None:
            weight = arrays[f"{prefix}/w"]
            if weight.ndim == 3:  # hk.Conv1D with kernel_shape 1
                weight = weight[0]
            module.weight.data.copy_(torch.as_tensor(weight.T.copy()))
            module.bias.data.copy_(torch.as_tensor(arrays[f"{prefix}/b"]))

        def encoder(module: MetaInputEncoder, prefix: str, mlp: str, mlp1: str, seq: str) -> None:
            for index, layer in enumerate(module.y_net.layers):
                linear(layer, f"{prefix}/{mlp}/~/linear_{index}")
            for index, layer in enumerate(module.z_net.layers):
                linear(layer, f"{prefix}/{mlp1}/~/linear_{index}")
            for index, layer in enumerate(module.policy_net.layers):
                suffix = "" if index == 0 else f"_{index}"
                linear(layer, f"{prefix}/{seq}/conv1_d{suffix}")

        encoder(self.encoder, "lstm", "mlp", "mlp_1", "sequential")
        linear(self.trajectory_lstm.linear, "lstm/lstm/linear")
        linear(self.state_linear, "lstm/linear")
        linear(self.emb_linear, "lstm/linear_1")
        linear(self.y_head, "lstm/linear_2")
        linear(self.z_head, "lstm/linear_3")
        for index, layer in enumerate(self.policy_target_net.layers):
            suffix = "" if index == 0 else f"_{index}"
            linear(layer, f"lstm/sequential_1/conv1_d{suffix}")
        linear(self.pi_head, "lstm/linear_4")

        meta = "lstm/~/meta_lstm/~unroll"
        encoder(self.meta_encoder, meta, "mlp", "mlp_1", "sequential")
        for index, layer in enumerate(self.meta_embed.layers):
            linear(layer, f"{meta}/mlp_2/~/linear_{index}")
        linear(self.meta_lstm.linear, f"{meta}/lstm/linear")


def resolve_meta_weights(path: str) -> str:
    """Find disco_103.npz, downloading it to the user cache if it is not local."""
    if path:
        return path
    override = os.environ.get("DISCO_META_WEIGHTS")
    if override:
        return override
    cache = os.path.join(os.path.expanduser("~"), ".cache", "cule-disco")
    target = os.path.join(cache, "disco_103.npz")
    if not os.path.exists(target):
        os.makedirs(cache, exist_ok=True)
        print(f"downloading the published Disco103 meta-parameters to {target}")
        urllib.request.urlretrieve(META_WEIGHTS_URL, target)
    return target


# ---------------------------------------------------------------------------
# The agent
# ---------------------------------------------------------------------------

class DiscoAgentNet(nn.Module):
    """Torso + flat heads (logits, y) + a Muesli-style action-conditional model.

    The published release only pins the meta-network and the *shape* of what the
    agent must predict; the torso is task specific and is the usual Atari CNN
    here rather than the reference's Catch-sized MLP.
    """

    def __init__(
        self,
        n_actions: int,
        num_bins: int = 601,
        prediction_size: int = PREDICTION_SIZE,
        torso_dense: Sequence[int] = (512,),
        head_w_init_std: float = 1e-2,
        head_mlp_hiddens: Sequence[int] = (256,),
        lstm_size: int = 256,
    ):
        super().__init__()
        self.n_actions = int(n_actions)
        self.num_bins = int(num_bins)
        self.prediction_size = int(prediction_size)
        self.lstm_size = int(lstm_size)

        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.torso = HaikuMLP(64 * 7 * 7, torso_dense)
        torso_width = self.torso.out_features

        self.logits_head = haiku_linear(torso_width, n_actions, head_w_init_std)
        self.y_head = haiku_linear(torso_width, prediction_size, head_w_init_std)

        self.root = haiku_linear(torso_width, lstm_size)
        self.action_lstm = HaikuLSTMCell(n_actions, lstm_size)
        self.z_head = HaikuMLP(lstm_size, (*head_mlp_hiddens, prediction_size))
        self.aux_pi_head = HaikuMLP(lstm_size, (*head_mlp_hiddens, n_actions))
        self.q_head = HaikuMLP(lstm_size, (*head_mlp_hiddens, num_bins))

    def forward(self, observations: torch.Tensor):
        """observations: [N, 4, 84, 84] uint8 or float -> a dict of outputs."""
        batch = observations.shape[0]
        # Cast to the module's own dtype rather than hard-coding float32, so a
        # double-precision copy can be diffed against the reference.
        dtype = self.root.weight.dtype
        torso = self.torso(self.conv(observations.to(dtype) / 255.0))

        cell = self.root(torso)
        hidden = torch.tanh(cell)
        # Expand one model step for every action at once: [N * A, ...].
        hidden = hidden.repeat_interleave(self.n_actions, dim=0)
        cell = cell.repeat_interleave(self.n_actions, dim=0)
        one_hot = torch.eye(self.n_actions, device=observations.device, dtype=torso.dtype)
        one_hot = one_hot.repeat(batch, 1)
        model_out, _ = self.action_lstm(one_hot, hidden, cell)

        return {
            "logits": self.logits_head(torso),
            "y": self.y_head(torso),
            "z": self.z_head(model_out).view(batch, self.n_actions, self.prediction_size),
            "aux_pi": self.aux_pi_head(model_out).view(batch, self.n_actions, self.n_actions),
            "q": self.q_head(model_out).view(batch, self.n_actions, self.num_bins),
        }

    def unroll(self, observations: torch.Tensor):
        """observations: [L, B, 4, 84, 84] -> outputs with a leading [L, B]."""
        horizon, batch = observations.shape[:2]
        flat = self.forward(observations.flatten(0, 1))
        return {key: value.view(horizon, batch, *value.shape[1:]) for key, value in flat.items()}


class ClippedAdam(torch.optim.Optimizer):
    """Adam whose normalized update is clipped element-wise before scaling by lr.

    `optax.chain(scale_by_adam, clip(max_delta), scale(-lr))`. The clip is on the
    *update*, not on the gradient, so no single parameter can ever move by more
    than `lr * max_delta` in one step whatever the loss does.
    """

    def __init__(self, params, lr=3e-4, betas=(0.9, 0.999), eps=1e-8, max_abs_update=1.0):
        super().__init__(
            params, dict(lr=lr, betas=betas, eps=eps, max_abs_update=max_abs_update)
        )

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                state = self.state[parameter]
                if not state:
                    state["step"] = 0
                    state["mu"] = torch.zeros_like(parameter)
                    state["nu"] = torch.zeros_like(parameter)
                state["step"] += 1
                mu, nu = state["mu"], state["nu"]
                mu.lerp_(parameter.grad, 1.0 - beta1)
                nu.mul_(beta2).addcmul_(parameter.grad, parameter.grad, value=1.0 - beta2)
                mu_hat = mu / (1.0 - beta1 ** state["step"])
                nu_hat = nu / (1.0 - beta2 ** state["step"])
                update = mu_hat / (nu_hat.sqrt() + group["eps"])
                update.clamp_(-group["max_abs_update"], group["max_abs_update"])
                parameter.add_(update, alpha=-group["lr"])


class TrajectoryReplay:
    """FIFO of whole trajectories, as in the reference evaluation loop."""

    def __init__(self, capacity: int):
        self.capacity = int(capacity)
        self.data: list[tuple] = []
        self.position = 0

    def add(self, trajectories: Sequence[tuple]) -> None:
        for trajectory in trajectories:
            if len(self.data) < self.capacity:
                self.data.append(trajectory)
            else:
                self.data[self.position] = trajectory
                self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int):
        indices = np.random.randint(len(self.data), size=batch_size)
        columns = list(zip(*(self.data[index] for index in indices)))
        return [np.stack(column, axis=1) for column in columns]

    def __len__(self) -> int:
        return len(self.data)


def compute_value_outs(
    agent_out,
    target_out,
    behaviour_logits,
    rewards,
    is_terminal,
    actions,
    discount: float,
    lambda_: float,
    max_abs_value: float,
    adv_ema: MovingAverage,
    td_ema: MovingAverage,
):
    """Everything the meta-network is handed about values, in reference order."""
    logits = agent_out["logits"]
    policy = F.softmax(logits, dim=-1)
    q_values = value_logits_to_scalar(agent_out["q"], max_abs_value)
    target_q_values = value_logits_to_scalar(target_out["q"], max_abs_value)
    # No separate value head: V is the policy-weighted average of Q, for both
    # the online and the target network.
    values = (policy * q_values).sum(dim=2)
    target_values = (policy * target_q_values).sum(dim=2)

    rho = importance_weight(logits[:-1], behaviour_logits[:-1], actions[:-1])
    value_outs = estimate_q_values(
        rewards,
        actions[:-1],
        1.0 - is_terminal,
        rho,
        values,
        target_values,
        q_values,
        target_q_values,
        discount,
        lambda_,
    )

    adv_ema.update(value_outs.adv)
    td_ema.update(value_outs.q_td)
    return value_outs._replace(
        normalized_adv=adv_ema.normalize(value_outs.adv),
        normalized_qv_adv=adv_ema.normalize(value_outs.qv_adv),
        q_td=value_outs.q_td,
    ), td_ema.normalize(value_outs.q_td, subtract_mean=False)


def disco_agent_loss(
    agent_out,
    actions,
    is_terminal,
    pi_hat,
    y_hat,
    z_hat,
    q_td,
    max_abs_value: float,
    pi_cost: float,
    y_cost: float,
    z_cost: float,
    aux_policy_cost: float,
    value_cost: float,
):
    """The discovered objective: three KLs to the meta-net's targets, plus value.

    Nothing here is a policy gradient or a TD error written by hand. `pi_hat`,
    `y_hat` and `z_hat` are whatever the meta-network decided the agent should
    predict, and the agent is simply regressed onto them in KL.
    """
    logits = agent_out["logits"][:-1]
    y = agent_out["y"][:-1]
    z_a = batch_lookup(agent_out["z"][:-1], actions[:-1])

    pi_loss = categorical_kl_divergence(pi_hat, logits)
    y_loss = categorical_kl_divergence(y_hat, y)
    z_loss = categorical_kl_divergence(z_hat, z_a)

    # One-step policy prediction: the model head at the taken action must match
    # the policy the torso produces at the next observation.
    aux_pi_a = batch_lookup(agent_out["aux_pi"][:-1], actions[:-1])
    aux_target = agent_out["logits"][1:].detach()
    aux_loss = categorical_kl_divergence(aux_target, aux_pi_a) * (1.0 - is_terminal)

    # The value head is trained to move by exactly the meta-net's TD error.
    q_a = batch_lookup(agent_out["q"], actions)[:-1]
    values = value_logits_to_scalar(q_a, max_abs_value)
    value_target = signed_hyperbolic((values + q_td).detach())
    target_probs = transform_to_2hot(value_target, -max_abs_value, max_abs_value, q_a.shape[-1])
    value_loss = -(target_probs * F.log_softmax(q_a, dim=-1)).sum(-1)

    total = (
        pi_cost * pi_loss
        + y_cost * y_loss
        + z_cost * z_loss
        + aux_policy_cost * aux_loss
        + value_cost * value_loss
    )
    return total, dict(
        pi_loss=pi_loss.mean().detach(),
        y_loss=y_loss.mean().detach(),
        z_loss=z_loss.mean().detach(),
        aux_loss=aux_loss.mean().detach(),
        value_loss=value_loss.mean().detach(),
    )


if __name__ == "__main__":
    process_start = time.perf_counter()
    args = tyro.cli(Args)
    if args.num_envs < 1:
        raise ValueError("num_envs must be positive")
    if args.num_steps < 2:
        raise ValueError("num_steps must be at least 2")
    if args.batch_size < 1:
        raise ValueError("batch_size must be positive")
    if args.replay_capacity < args.batch_size:
        raise ValueError("replay_capacity must be at least batch_size")
    if args.max_training_seconds < 0:
        raise ValueError("max_training_seconds must be non-negative")
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
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"
    n_actions = int(envs.single_action_space.n)

    agent = DiscoAgentNet(
        n_actions,
        num_bins=args.num_bins,
        torso_dense=args.torso_dense,
        head_w_init_std=args.head_w_init_std,
        head_mlp_hiddens=args.head_mlp_hiddens,
        lstm_size=args.model_lstm_size,
    ).to(device)
    target_agent = DiscoAgentNet(
        n_actions,
        num_bins=args.num_bins,
        torso_dense=args.torso_dense,
        head_w_init_std=args.head_w_init_std,
        head_mlp_hiddens=args.head_mlp_hiddens,
        lstm_size=args.model_lstm_size,
    ).to(device)
    target_agent.load_state_dict(agent.state_dict())
    target_agent.requires_grad_(False)

    meta_net = DiscoMetaNet().to(device)
    weights_path = resolve_meta_weights(args.meta_weights)
    with np.load(weights_path) as arrays:
        meta_net.load_published_weights(arrays)
    # The rule is fixed: meta-training is out of scope here.
    meta_net.requires_grad_(False).eval()
    print(
        f"loaded {sum(p.numel() for p in meta_net.parameters()):,} discovered "
        f"meta-parameters from {weights_path}"
    )

    optimizer = ClippedAdam(
        agent.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=args.adam_eps,
        max_abs_update=args.max_abs_update,
    )
    adv_ema = MovingAverage(args.moving_average_decay, args.moving_average_eps).to(device)
    td_ema = MovingAverage(args.moving_average_decay, args.moving_average_eps).to(device)
    replay = TrajectoryReplay(args.replay_capacity)

    horizon = args.num_steps  # T
    observations = np.zeros((horizon + 1, args.num_envs, 4, 84, 84), dtype=np.uint8)
    stored_actions = np.zeros((horizon + 1, args.num_envs), dtype=np.int64)
    stored_logits = np.zeros((horizon + 1, args.num_envs, n_actions), dtype=np.float32)
    stored_rewards = np.zeros((horizon, args.num_envs), dtype=np.float32)
    stored_terminal = np.zeros((horizon, args.num_envs), dtype=np.float32)

    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    reset_result = envs.reset(seed=args.seed) if args.env_backend != "envpool" else envs.reset()
    obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result

    def act(current_obs):
        with torch.no_grad():
            logits = agent(to_tensor(current_obs, device))["logits"]
            actions = torch.multinomial(F.softmax(logits, dim=-1), 1).squeeze(-1)
        return logits, actions

    current_logits, current_actions = act(obs)
    global_step = 0
    learner_steps = 0
    episode_stats = EpisodeStats(args.solve_window, args.solve_reward)
    next_log_step = max(10000, args.num_envs)
    num_iterations = int(np.ceil(args.total_timesteps / (args.num_envs * horizon)))
    if args.benchmark:
        num_iterations = args.benchmark_warmup_iterations + args.benchmark_measure_iterations
    benchmark_start = None
    benchmark_start_step = None
    benchmark_start_updates = None
    last_log = {}
    for iteration in range(num_iterations):
        if args.benchmark and iteration == args.benchmark_warmup_iterations:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            benchmark_start = time.perf_counter()
            benchmark_start_step = global_step
            benchmark_start_updates = learner_steps
        if args.max_training_seconds and time.time() - start_time >= args.max_training_seconds:
            break

        solved = False
        for step in range(horizon + 1):
            observations[step] = to_numpy(obs)
            stored_actions[step] = to_numpy(current_actions)
            stored_logits[step] = to_numpy(current_logits)
            if step == horizon:
                break

            # TRY NOT TO MODIFY: execute the game and log data.
            step_result = step_env(envs, current_actions)
            if len(step_result) == 5:
                next_obs, rewards, terminations, truncations, infos = step_result
            else:
                next_obs, rewards, terminations, infos = step_result
                truncations = np.zeros_like(np.asarray(terminations), dtype=bool)
            dones = done_tensor(terminations, truncations, device).bool()
            stored_rewards[step] = to_numpy(rewards).reshape(args.num_envs)
            stored_terminal[step] = to_numpy(dones).reshape(args.num_envs).astype(np.float32)
            global_step += args.num_envs

            if not args.benchmark:
                solved |= episode_stats.update(
                    completed_episode_infos(infos, dones), global_step, writer
                )

            obs = next_obs
            current_logits, current_actions = act(obs)

        replay.add(
            [
                (
                    observations[:, env].copy(),
                    stored_actions[:, env].copy(),
                    stored_logits[:, env].copy(),
                    stored_rewards[:, env].copy(),
                    stored_terminal[:, env].copy(),
                )
                for env in range(args.num_envs)
            ]
        )

        if len(replay) >= args.batch_size:
            for _ in range(args.learner_steps_per_rollout):
                batch = replay.sample(args.batch_size)
                batch_obs, batch_actions, batch_logits, batch_rewards, batch_terminal = (
                    torch.as_tensor(array, device=device) for array in batch
                )

                agent_out = agent.unroll(batch_obs)
                with torch.no_grad():
                    target_out = target_agent.unroll(batch_obs)
                    detached = {key: value.detach() for key, value in agent_out.items()}
                    value_outs, normalized_q_td = compute_value_outs(
                        detached,
                        target_out,
                        batch_logits,
                        batch_rewards,
                        batch_terminal,
                        batch_actions,
                        args.discount,
                        args.td_lambda,
                        args.max_abs_value,
                        adv_ema,
                        td_ema,
                    )
                    meta_inputs = MetaInputs(
                        actions=batch_actions,
                        rewards=batch_rewards,
                        is_terminal=batch_terminal,
                        logits=detached["logits"],
                        behaviour_logits=batch_logits,
                        target_logits=target_out["logits"],
                        y=detached["y"],
                        target_y=target_out["y"],
                        z=detached["z"],
                        target_z=target_out["z"],
                        v_scalar=value_outs.value,
                        adv=value_outs.adv,
                        normalized_adv=value_outs.normalized_adv,
                        q=value_outs.target_q_value,
                        qv_adv=value_outs.qv_adv,
                        normalized_qv_adv=value_outs.normalized_qv_adv,
                    )
                    pi_hat, y_hat, z_hat = meta_net(meta_inputs)

                loss_per_step, log = disco_agent_loss(
                    agent_out,
                    batch_actions,
                    batch_terminal,
                    pi_hat,
                    y_hat,
                    z_hat,
                    value_outs.q_td,
                    args.max_abs_value,
                    args.pi_cost,
                    args.y_cost,
                    args.z_cost,
                    args.aux_policy_cost,
                    args.value_cost,
                )
                loss = loss_per_step.mean()

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                learner_steps += 1

                with torch.no_grad():
                    for target_param, param in zip(target_agent.parameters(), agent.parameters()):
                        target_param.mul_(args.target_params_coeff).add_(
                            param, alpha=1.0 - args.target_params_coeff
                        )
                last_log = {"loss": loss.detach(), **log, "normalized_q_td": normalized_q_td.mean()}

        if not args.benchmark and global_step >= next_log_step and last_log:
            sps = int(global_step / (time.time() - start_time))
            for name, value in last_log.items():
                writer.add_scalar(f"losses/{name}", value.item(), global_step)
            writer.add_scalar("charts/SPS", sps, global_step)
            writer.add_scalar("charts/learner_steps", learner_steps, global_step)
            writer.add_scalar("charts/replay_size", len(replay), global_step)
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
        measured_updates = learner_steps - benchmark_start_updates
        measured_seconds = benchmark_end - benchmark_start
        result = {
            "algorithm": "disco",
            "backend": args.env_backend,
            "batch_size": args.batch_size,
            "benchmark": "full_training_loop",
            "compile": False,
            "env_device": str(getattr(envs, "device", "cpu")),
            "env_id": args.env_id,
            "implementation": "original",
            "learner_updates": measured_updates,
            "measure_iterations": args.benchmark_measure_iterations,
            "measured_seconds": measured_seconds,
            "measured_steps": measured_steps,
            "num_envs": args.num_envs,
            "num_steps": args.num_steps,
            "peak_cuda_memory_mb": (
                torch.cuda.max_memory_allocated(device) / (1024**2) if device.type == "cuda" else 0.0
            ),
            "process_seconds": benchmark_end - process_start,
            "replay_backend": "numpy_trajectory_fifo",
            "schema_version": 1,
            "sps": measured_steps / measured_seconds,
            "ups": measured_updates / measured_seconds,
            "warmup_iterations": args.benchmark_warmup_iterations,
        }
        print(f"BENCHMARK_RESULT {json.dumps(result, sort_keys=True)}", flush=True)
    else:
        elapsed = time.time() - start_time
        print("SPS:", int(global_step / elapsed))
        print("learner steps:", learner_steps)
        print("UPS:", learner_steps / elapsed)
        episode_stats.print_summary()

    if args.save_model and not args.benchmark:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save({"model_weights": agent.state_dict(), "args": vars(args)}, model_path)
        print(f"model saved to {model_path}")

    envs.close()
    if writer is not None:
        writer.close()
