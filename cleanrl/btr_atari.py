# BTR: Beyond The Rainbow (Clark et al., ICML 2025,
# https://arxiv.org/abs/2411.03820).  Ported from the official implementation
# (https://github.com/VIPTankz/BTR: networks.py `ImpalaCNNLargeIQN`, Agent.py
# `learn_call` branch `iqn and munchausen`).
#
# BTR is Rainbow plus six changes, and *five of the six are already in this
# repo's M-IQN* -- BTR's learning rule is exactly Munchausen-IQN.  So this file
# is miqn_atari.py with the encoder and the replay swapped:
#
#   1. Impala-large CNN, width 2x  -> replaces the Nature CNN (`ImpalaEncoder`)
#   2. Spectral norm               -> on every residual conv (not the stem conv)
#   3. Adaptive 6x6 max-pool       -> 11x11x64 feature map -> 6x6x64 = 2304
#   4. IQN                         -> INHERITED from miqn_atari.py
#   5. Munchausen RL               -> INHERITED from miqn_atari.py
#   6. Vectorised envs + retuned   -> 64 envs, batch 256, gamma 0.997, lr 1e-4,
#      hyperparameters                target every 500 updates, PER alpha 0.2,
#                                     noisy nets + annealed epsilon-greedy
#
# Deviations from the official BTR, all of which follow the official code rather
# than the paper text:
#   * the Munchausen log-policy bonus is read from the ONLINE network on s_t
#     (official `q_k_target = self.net.qvals(states)`), whereas M-IQN and
#     miqn_atari.py read it from the target network;
#   * the cosine basis uses i = 0..n_cos-1 (official `range(self.n_cos)`), while
#     miqn_atari.py follows Dopamine's shipped i = 1..n_cos;
#   * a single tau draw of size num_tau=8 serves prediction, target, and policy.
#
# The trainer structure is inherited from miqn_atari.py, which follows
# iqn_atari.py / c51_atari.py, adapted from CleanRL
# (https://github.com/vwxyzjn/cleanrl, MIT; license in cleanrl/LICENSE.md).
# Supports gymnasium, cule, and envpool.
#
# CONFIRMED against the official implementation: `tests/crosscheck/check_btr.py`
# transplants VIPTankz/BTR's `ImpalaCNNLargeIQN` weights into this file's
# QNetwork and diffs the whole pipeline. 24/24 components match on CPU and CUDA
# (<= 2e-6, the residue being float32 reduction order): every Impala block, the
# pooled trunk, the quantile head at pinned taus, the advantages-only path, the
# factorised noisy layers, the Munchausen target, the quantile Huber loss and
# its gradient, the PER priority, and the defaults parsed out of upstream's own
# main.py and Agent.py.
import json
import math
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
import torch.nn.functional as F
import torch.optim as optim
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
from cleanrl_utils.buffers import PrioritizedAtariReplayBuffer
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
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""

    # Algorithm specific arguments
    env_id: str = "BreakoutNoFrameskip-v4"
    """the id of the environment"""
    total_timesteps: int = 10000000
    """total timesteps of the experiments"""
    learning_rate: float = 1e-4
    """the learning rate of the optimizer"""
    num_envs: int = 64
    """the number of parallel game environments"""
    n_taus: int = 8
    """the number of quantile samples for the online network"""
    n_target_taus: int = 8
    """the number of quantile samples for the TD target"""
    n_policy_taus: int = 8
    """the number of quantile samples for action selection"""
    n_cos: int = 64
    """the dimension of the cosine quantile embedding"""
    adam_eps_ratio: float = 0.005
    """Adam epsilon is this divided by the batch size (official `0.005 / batch_size`)"""
    kappa: float = 1.0
    """the Huber threshold of the quantile regression loss"""
    munchausen_alpha: float = 0.9
    """the Munchausen log-policy bonus scale"""
    munchausen_tau: float = 0.03
    """the entropy temperature of the Munchausen soft policy"""
    munchausen_clip: float = -1.0
    """the lower clip of the scaled log-policy bonus (official l0)"""
    interact: str = "greedy"
    """behavior policy: `greedy` takes argmax (official BTR), `stochastic` samples softmax(Q/tau)"""
    model_size: int = 2
    """the Impala CNN width multiplier"""
    maxpool_size: int = 6
    """the side length of the adaptive max-pool applied after the Impala trunk"""
    linear_size: int = 512
    """the width of the dueling value and advantage branches"""
    noisy_std: float = 0.5
    """sigma_0 of the factorised noisy linear layers"""
    spectral_norm: bool = True
    """whether to spectrally normalise the Impala residual convolutions"""
    max_grad_norm: float = 10.0
    """the maximum gradient norm"""
    buffer_size: int = 1048576
    """the replay memory buffer size"""
    gamma: float = 0.997
    """the discount factor gamma"""
    n_step: int = 3
    """the number of steps to look ahead for n-step Q learning"""
    prioritized_replay_alpha: float = 0.2
    """alpha parameter for prioritized replay buffer"""
    prioritized_replay_beta: float = 0.4
    """initial beta parameter for prioritized replay buffer"""
    prioritized_replay_eps: float = 1e-6
    """epsilon parameter for prioritized replay buffer"""
    target_network_frequency: int = 500
    """learner updates between target-network updates"""
    batch_size: int = 256
    """the batch size of sample from the reply memory"""
    start_e: float = 1
    """the starting epsilon for exploration"""
    end_e: float = 0.01
    """the ending epsilon for exploration"""
    epsilon_decay_steps: int = 2000000
    """transitions over which epsilon anneals from start-e to end-e (official eps_steps)"""
    epsilon_disable_fraction: float = 0.5
    """fraction of `total-timesteps` after which epsilon-greedy is switched off and
    exploration is left entirely to the noisy layers (official eps_disable)"""
    learning_starts: int = 200000
    """timestep to start learning"""
    learner_updates_per_vector_step: float = 1.0
    """gradient updates accrued per vector environment step; may be fractional"""
    replay_ratio: float | None = None
    """sampled replay items per collected transition; overrides learner-updates-per-vector-step"""
    max_training_seconds: float = 0.0
    """wall-clock training limit; zero disables it"""
    solve_reward: float | None = None
    """stop when the moving episodic return reaches this value"""
    solve_window: int = 20
    """number of completed episodes in the solve moving average"""
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


class FactorizedNoisyLinear(nn.Module):
    """Factorised Gaussian noisy linear layer, as BTR implements it.

    Two deliberate differences from the `NoisyLinear` in rainbow_atari.py, both
    taken from the official `FactorizedNoisyLinear`:
      * the bias sigma is initialised to sigma_0 / sqrt(fan_in), not
        sigma_0 / sqrt(fan_out);
      * the forward pass has no `self.training` branch -- noise is switched off
        by zeroing the epsilon buffers, and they start zeroed, so an agent is
        deterministic until the first `reset_noise()` call.
    """

    def __init__(self, in_features: int, out_features: int, sigma_0: float = 0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer("bias_epsilon", torch.empty(out_features))
        scale = 1 / math.sqrt(in_features)
        nn.init.uniform_(self.weight_mu, -scale, scale)
        nn.init.uniform_(self.bias_mu, -scale, scale)
        nn.init.constant_(self.weight_sigma, sigma_0 * scale)
        nn.init.constant_(self.bias_sigma, sigma_0 * scale)
        self.disable_noise()

    @torch.no_grad()
    def _get_noise(self, size: int) -> torch.Tensor:
        noise = torch.randn(size, device=self.weight_mu.device)
        return noise.sign().mul_(noise.abs().sqrt_())  # f(x) = sgn(x) sqrt(|x|)

    @torch.no_grad()
    def reset_noise(self) -> None:
        epsilon_in = self._get_noise(self.in_features)
        epsilon_out = self._get_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.outer(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    @torch.no_grad()
    def disable_noise(self) -> None:
        self.weight_epsilon.zero_()
        self.bias_epsilon.zero_()

    def forward(self, input):
        return F.linear(
            input,
            self.weight_mu + self.weight_sigma * self.weight_epsilon,
            self.bias_mu + self.bias_sigma * self.bias_epsilon,
        )


class ImpalaResidual(nn.Module):
    """Pre-activation residual block of the large Impala CNN."""

    def __init__(self, depth: int, norm_func):
        super().__init__()
        self.conv_0 = norm_func(nn.Conv2d(depth, depth, 3, stride=1, padding=1))
        self.conv_1 = norm_func(nn.Conv2d(depth, depth, 3, stride=1, padding=1))

    def forward(self, x):
        # The activation precedes each convolution, so the skip path is linear.
        hidden = self.conv_0(F.relu(x))
        hidden = self.conv_1(F.relu(hidden))
        return x + hidden


class ImpalaBlock(nn.Module):
    """Stem convolution, stride-2 max-pool, then two residual blocks."""

    def __init__(self, depth_in: int, depth_out: int, norm_func):
        super().__init__()
        # BTR spectrally normalises the residual convolutions only; the stem
        # convolution of each block is left unnormalised.
        self.conv = nn.Conv2d(depth_in, depth_out, 3, stride=1, padding=1)
        self.max_pool = nn.MaxPool2d(3, 2, padding=1)
        self.residual_0 = ImpalaResidual(depth_out, norm_func)
        self.residual_1 = ImpalaResidual(depth_out, norm_func)

    def forward(self, x):
        return self.residual_1(self.residual_0(self.max_pool(self.conv(x))))


# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    """Impala-large IQN trunk with a factorised-noisy dueling quantile head.

    Keeps miqn_atari.py's `features` / `quantile_values` / `get_action` surface so
    the Munchausen-IQN training loop is unchanged; only what is behind them
    differs.
    """

    def __init__(
        self,
        env,
        n_cos=64,
        n_policy_taus=8,
        model_size=2,
        maxpool_size=6,
        linear_size=512,
        noisy_std=0.5,
        spectral=True,
    ):
        super().__init__()
        self.n_cos = int(n_cos)
        self.n_policy_taus = int(n_policy_taus)
        self.n = int(env.single_action_space.n)
        # Official BTR indexes the cosine basis from 0; miqn_atari.py follows
        # Dopamine's shipped 1..n_cos.  Keeping BTR's convention here, and
        # building it in float64 before the cast so it is bit-identical to
        # upstream's `[np.pi * i for i in range(n_cos)]`.
        self.register_buffer(
            "cos_multipliers", (math.pi * torch.arange(n_cos, dtype=torch.float64)).float()
        )

        def identity(module):
            return module

        norm_func = nn.utils.parametrizations.spectral_norm if spectral else identity
        self.conv = nn.Sequential(
            ImpalaBlock(4, 16 * model_size, norm_func),  # 84 -> 42
            ImpalaBlock(16 * model_size, 32 * model_size, norm_func),  # 42 -> 21
            ImpalaBlock(32 * model_size, 32 * model_size, norm_func),  # 21 -> 11
            nn.ReLU(),
            # Adaptive pooling decouples the head width from the input
            # resolution and drops 77% of the trunk's parameters.
            nn.AdaptiveMaxPool2d((maxpool_size, maxpool_size)),
            nn.Flatten(),
        )
        self.feature_dim = 32 * model_size * maxpool_size * maxpool_size
        self.cos_embedding = nn.Linear(self.n_cos, self.feature_dim)
        self.value_head = nn.Sequential(
            FactorizedNoisyLinear(self.feature_dim, linear_size, noisy_std),
            nn.ReLU(),
            FactorizedNoisyLinear(linear_size, 1, noisy_std),
        )
        self.advantage_head = nn.Sequential(
            FactorizedNoisyLinear(self.feature_dim, linear_size, noisy_std),
            nn.ReLU(),
            FactorizedNoisyLinear(linear_size, self.n, noisy_std),
        )

    def features(self, x):
        return self.conv(x / 255.0)

    def quantile_values(self, features, taus, advantages_only: bool = False):
        """Quantile values z_tau(x, a) with shape (batch, taus, actions)."""
        cos = torch.cos(taus.unsqueeze(-1) * self.cos_multipliers)
        phi = F.relu(self.cos_embedding(cos))
        h = features.unsqueeze(1) * phi
        advantage = self.advantage_head(h)
        if advantages_only:
            # The value term is action-independent, so argmax is unaffected.
            return advantage
        value = self.value_head(h)
        return value + advantage - advantage.mean(dim=-1, keepdim=True)

    def get_action(self, x, action=None):
        features = self.features(x)
        taus = torch.rand(features.shape[0], self.n_policy_taus, device=features.device)
        quantiles = self.quantile_values(features, taus)
        q_values = quantiles.mean(1)
        if action is None:
            action = torch.argmax(q_values, 1)
        return action, quantiles[torch.arange(quantiles.shape[0], device=quantiles.device), :, action]

    def reset_noise(self):
        for module in self.modules():
            if isinstance(module, FactorizedNoisyLinear):
                module.reset_noise()

    def disable_noise(self):
        for module in self.modules():
            if isinstance(module, FactorizedNoisyLinear):
                module.disable_noise()


def scaled_log_softmax(q_values, tau):
    """tau * log_softmax(q / tau) (official munchausen_rl utils, stable form)."""
    return tau * F.log_softmax(q_values / tau, dim=-1)


def munchausen_target(
    current_q, next_q, next_z, actions, rewards, dones, alpha, tau, clip, gamma_n
):
    """BTR's Munchausen-IQN target.

    Two things distinguish it from plain IQN. The reward carries a Munchausen
    bonus `alpha * clip(tau * ln pi(a|s), clip, 0)` read off the *online*
    network at s (official `q_k_target = self.net.qvals(states)`), and the
    bootstrap is the soft expectation `E_pi[z(s',a) - tau ln pi(a|s')]` rather
    than a max, so the target is an entropy-regularised one.
    """
    tau_log_pi_current = scaled_log_softmax(current_q, tau)
    bonus = alpha * tau_log_pi_current.gather(1, actions).clamp(clip, 0.0)
    tau_log_pi_next = scaled_log_softmax(next_q, tau)
    pi_next = F.softmax(next_q / tau, dim=-1)
    soft_values = (pi_next.unsqueeze(1) * (next_z - tau_log_pi_next.unsqueeze(1))).sum(2)
    return rewards + bonus + gamma_n * soft_values * (1.0 - dones)


def quantile_td_errors(predicted, target):
    """Pairwise errors u[b, i, j] = target_j - predicted_i."""
    return target.unsqueeze(1) - predicted.unsqueeze(2)


def quantile_huber_loss(errors, taus, kappa):
    """Per-sample quantile Huber loss over the pairwise error matrix.

    Upstream reduces as `.sum(dim=1).mean(dim=1)` -- summed over the predicting
    taus, averaged over the target taus. Doing the mean first is the same number
    and keeps the two reductions on their own axes.
    """
    absolute = errors.abs()
    huber = torch.where(absolute <= kappa, 0.5 * errors.pow(2), kappa * (absolute - 0.5 * kappa))
    rho = (taus.unsqueeze(-1) - (errors.detach() < 0).float()).abs() * huber / kappa
    return rho.mean(2).sum(1)


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


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
    if min(args.n_taus, args.n_target_taus, args.n_policy_taus, args.n_cos) < 1:
        raise ValueError("n_taus, n_target_taus, n_policy_taus, and n_cos must be positive")
    if args.kappa <= 0:
        raise ValueError("kappa must be positive")
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

    network_kwargs = dict(
        n_cos=args.n_cos,
        n_policy_taus=args.n_policy_taus,
        model_size=args.model_size,
        maxpool_size=args.maxpool_size,
        linear_size=args.linear_size,
        noisy_std=args.noisy_std,
        spectral=args.spectral_norm,
    )
    q_network = QNetwork(envs, **network_kwargs).to(device)
    # Official BTR: Adam(lr=1e-4, eps=0.005 / batch_size), which is 1.953125e-5
    # at the default batch of 256. It is written as the ratio rather than the
    # rounded constant so it tracks --batch-size exactly as upstream does.
    optimizer = optim.Adam(
        q_network.parameters(), lr=args.learning_rate, eps=args.adam_eps_ratio / args.batch_size
    )
    target_network = QNetwork(envs, **network_kwargs).to(device)
    target_network.load_state_dict(q_network.state_dict())

    # BTR replaces M-IQN's uniform replay with n-step prioritized replay.
    rb = PrioritizedAtariReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        n_envs=args.num_envs,
        n_step=args.n_step,
        gamma=args.gamma,
        alpha=args.prioritized_replay_alpha,
        beta=args.prioritized_replay_beta,
        eps=args.prioritized_replay_eps,
    )
    epsilon_disable_step = args.epsilon_disable_fraction * args.total_timesteps
    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    reset_result = envs.reset(seed=args.seed) if args.env_backend != "envpool" else envs.reset()
    obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result
    rb.initialize(obs)
    global_step = 0
    update_budget = 0.0
    learner_updates = 0
    episode_stats = EpisodeStats(args.solve_window, args.solve_reward)
    next_target_update = args.target_network_frequency
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
        previous_global_step = global_step
        # anneal PER beta to 1
        rb.beta = min(
            1.0,
            args.prioritized_replay_beta
            + previous_global_step * (1.0 - args.prioritized_replay_beta) / args.total_timesteps,
        )
        # ALGO LOGIC: put action logic here
        # BTR anneals epsilon over a fixed transition budget rather than a
        # fraction of training, and switches epsilon-greedy off entirely once
        # half the frames are spent, leaving the noisy layers as the only source
        # of exploration.  (The official code decays epsilon geometrically,
        # `eps -= (eps - eps_final) / steps`; the paper describes a linear decay
        # and this keeps the parent's linear schedule.)
        if previous_global_step >= epsilon_disable_step:
            epsilon = 0.0
        else:
            epsilon = linear_schedule(
                args.start_e, args.end_e, args.epsilon_decay_steps, previous_global_step
            )
        if previous_global_step < args.learning_starts:
            actions = torch.randint(
                envs.single_action_space.n, (args.num_envs,), device=device
            )
        else:
            # Resample the factorised noise once per acting step, as BTR does at
            # the top of `choose_action`.
            q_network.reset_noise()
            with torch.no_grad():
                features = q_network.features(to_tensor(obs, device))
                policy_taus = torch.rand(features.shape[0], q_network.n_policy_taus, device=device)
                if args.interact == "stochastic":
                    q_values = q_network.quantile_values(features, policy_taus).mean(1)
                    # Gumbel-max sampling from softmax(Q / tau)
                    uniform = torch.rand_like(q_values).clamp_min(1e-10)
                    gumbel = -torch.log((-torch.log(uniform)).clamp_min(1e-10))
                    greedy_actions = torch.argmax(q_values / args.munchausen_tau + gumbel, dim=1)
                else:
                    # Only the advantage branch is needed for an argmax.
                    advantages = q_network.quantile_values(
                        features, policy_taus, advantages_only=True
                    ).mean(1)
                    greedy_actions = torch.argmax(advantages, dim=1)
            random_actions = torch.randint(
                envs.single_action_space.n, (args.num_envs,), device=device
            )
            explore = torch.rand(args.num_envs, device=device) < epsilon
            actions = torch.where(explore, random_actions, greedy_actions)

        # TRY NOT TO MODIFY: execute the game and log data.
        step_result = step_env(envs, actions)
        if len(step_result) == 5:
            next_obs, rewards, terminations, truncations, infos = step_result
        else:
            next_obs, rewards, terminations, infos = step_result
            truncations = np.zeros_like(np.asarray(terminations), dtype=bool)
        transition_dones = done_tensor(terminations, truncations, device).bool()
        global_step += args.num_envs

        # TRY NOT TO MODIFY: record rewards for plotting purposes
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
            gamma_n = args.gamma**args.n_step
            for _ in range(num_updates):
                # BTR resamples the target network's noise once per learner call.
                target_network.reset_noise()
                data = rb.sample(args.batch_size)
                batch_indices = torch.arange(args.batch_size, device=device)
                with torch.no_grad():
                    next_features = target_network.features(data.next_observations)
                    # One target draw serves both the bootstrap quantiles and the
                    # target policy: BTR takes `q_t_n = Q_targets_next.mean(1)`
                    # rather than drawing separate policy taus as M-IQN does.
                    next_z = target_network.quantile_values(
                        next_features,
                        torch.rand(args.batch_size, args.n_target_taus, device=device),
                    )
                    next_q = next_z.mean(1)
                    # Munchausen bonus: alpha * clip(tau * ln pi(a|s), l0, 0).
                    # BTR reads the log-policy off the ONLINE network at s_t
                    # (official `q_k_target = self.net.qvals(states)`), unlike
                    # M-IQN which uses the target network.
                    current_q = q_network.quantile_values(
                        q_network.features(data.observations),
                        torch.rand(args.batch_size, args.n_policy_taus, device=device),
                    ).mean(1)
                    target_quantiles = munchausen_target(
                        current_q, next_q, next_z, data.actions, data.rewards, data.dones,
                        args.munchausen_alpha, args.munchausen_tau, args.munchausen_clip, gamma_n,
                    )

                features = q_network.features(data.observations)
                taus = torch.rand(args.batch_size, args.n_taus, device=device)
                z = q_network.quantile_values(features, taus)
                old_quantiles = z[batch_indices, :, data.actions.flatten()]

                u = quantile_td_errors(old_quantiles, target_quantiles)
                loss_per_sample = quantile_huber_loss(u, taus, args.kappa)
                loss = (loss_per_sample * data.weights.squeeze()).mean()

                # PER priority is the summed-then-averaged absolute TD error
                # (official `loss_v = |td_error|.sum(dim=1).mean(dim=1)`).
                rb.update_priorities(data.indices, u.abs().mean(2).sum(1).detach().cpu().numpy())

                # optimize the model
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(q_network.parameters(), args.max_grad_norm)
                optimizer.step()
            learner_updates += num_updates

            # update target network
            if learner_updates >= next_target_update:
                target_network.load_state_dict(q_network.state_dict())
                next_target_update = (
                    learner_updates // args.target_network_frequency + 1
                ) * args.target_network_frequency

            if not args.benchmark and global_step >= next_log_step and num_updates:
                old_val = old_quantiles.mean(1)
                sps = int(global_step / (time.time() - start_time))
                writer.add_scalar("losses/loss", loss.item(), global_step)
                writer.add_scalar("losses/q_values", old_val.mean().item(), global_step)
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
            "algorithm": "btr",
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
            "peak_cuda_memory_mb": (
                torch.cuda.max_memory_allocated(device) / (1024**2) if device.type == "cuda" else 0.0
            ),
            "process_seconds": benchmark_end - process_start,
            "replay_backend": "numpy_frame_efficient_per",
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
            "model_weights": q_network.state_dict(),
            "args": vars(args),
        }
        torch.save(model_data, model_path)
        print(f"model saved to {model_path}")
        from cleanrl_utils.evals.distributional_eval import evaluate

        episodic_returns = evaluate(
            model_path,
            make_env,
            args.env_id,
            eval_episodes=10,
            run_name=f"{run_name}-eval",
            Model=QNetwork,
            model_kwargs_keys=("n_cos", "n_policy_taus"),
            device=device,
            epsilon=args.end_e,
        )
        for idx, episodic_return in enumerate(episodic_returns):
            writer.add_scalar("eval/episodic_return", episodic_return, idx)

        if args.upload_model:
            from cleanrl_utils.huggingface import push_to_hub

            repo_name = f"{args.env_id}-{args.exp_name}-seed{args.seed}"
            repo_id = f"{args.hf_entity}/{repo_name}" if args.hf_entity else repo_name
            push_to_hub(args, episodic_returns, repo_id, "M-IQN", f"runs/{run_name}", f"videos/{run_name}-eval")

    envs.close()
    if writer is not None:
        writer.close()
