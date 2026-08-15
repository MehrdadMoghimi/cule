# PPO + RND (Random Network Distillation) for Atari.
#
# Burda et al., ICLR 2019, "Exploration by Random Network Distillation"
# (https://arxiv.org/abs/1810.12894). Ported from the authors' implementation,
# openai/random-network-distillation (MIT), `policies/cnn_policy_param_matched.py`
# and `ppo_agent.py`, with the PPO scaffolding following CleanRL's
# `ppo_rnd_envpool.py`.
#
# RND is the exploration bonus that first solved Montezuma's Revenge without
# demonstrations. The idea is almost embarrassingly simple: freeze a randomly
# initialised network `f`, train a predictor `f_hat` to match it on observed
# states, and pay the agent the prediction error. On a state it has seen often
# the predictor has already fitted `f`, so the bonus is near zero; on a novel
# state the error is large. The bonus is therefore a *learned* novelty measure
# that needs no density model, no counts, and no dynamics model — and, unlike
# forward-dynamics bonuses, it cannot be farmed by finding a noisy TV, because
# `f` is deterministic.
#
# Every algorithm currently in this repository scores ~0 on Montezuma's Revenge,
# Pitfall and Private Eye. This is the file that is supposed to change that, and
# `--env-id MontezumaRevenge-v5` is what it is tuned for.
#
# The pieces that make it work, all of which are easy to leave out:
#
#   * **Two value heads and two discounts.** Extrinsic returns are episodic at
#     gamma=0.999; intrinsic returns are *non-episodic* at gamma=0.99 — the
#     intrinsic GAE never masks on `done`, deliberately, so that dying does not
#     look like a way to escape a low-novelty region. The advantages are
#     combined as `ext_coef * A_ext + int_coef * A_int` with (2.0, 1.0).
#   * **Observation normalisation for the RND input only.** The predictor sees a
#     single frame, whitened by a running mean/var and clipped to +/-5. The
#     policy still sees raw `obs / 255`. The running statistics are seeded by
#     `--num-iterations-obs-norm-init` rollouts of a *random* agent before
#     training starts, because an unnormalised first batch makes the bonus
#     meaningless.
#   * **Intrinsic reward normalisation.** Divided by the running standard
#     deviation of the discounted intrinsic return (`RewardForwardFilter`), not
#     of the reward itself.
#   * **Predictor subsampling.** Only `--update-proportion` (0.25) of each
#     minibatch contributes to the predictor loss, so the predictor learns more
#     slowly than the policy collects — which is what keeps the bonus alive.
#
# One place the reference and CleanRL disagree: the official intrinsic reward is
# `mean_j (f_j - f_hat_j)^2` (`cnn_policy_param_matched.py:167`), while CleanRL
# uses `sum_j (...)^2 / 2`. This port follows the reference. The two differ by a
# constant factor of `feature_dim / 2`, which the reward normalisation divides
# straight back out — `tests/test_rnd_equivalence.py` proves the normalised
# bonuses are proportional, so the disagreement is real but inert.
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
import torch.nn.functional as F
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
    env_backend: str = "envpool"
    """environment backend: `gymnasium`, `cule`, or `envpool`"""
    cule_device: str = "auto"
    """device for the CuLE backend: `auto`, `cpu`, or a CUDA device string"""

    # Algorithm specific arguments
    env_id: str = "MontezumaRevenge-v5"
    """the id of the environment"""
    total_timesteps: int = 2000000000
    """total timesteps of the experiments"""
    learning_rate: float = 1e-4
    """the learning rate of the optimizer"""
    num_envs: int = 128
    """the number of parallel game environments"""
    num_steps: int = 128
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.999
    """the extrinsic discount factor"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 4
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.1
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function"""
    ent_coef: float = 0.001
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

    # RND arguments
    update_proportion: float = 0.25
    """fraction of each minibatch used for the predictor update"""
    int_coef: float = 1.0
    """weight on the intrinsic advantage"""
    ext_coef: float = 2.0
    """weight on the extrinsic advantage"""
    int_gamma: float = 0.99
    """intrinsic discount factor"""
    num_iterations_obs_norm_init: int = 50
    """random-agent rollouts used to seed the observation normalization"""
    obs_norm_clip: float = 5.0
    """the RND input is clipped to +/- this after whitening"""
    intrinsic_reward_reduction: str = "mean"
    """`mean` (the reference) or `sum_half` (CleanRL's variant); see the header"""
    rnd_feature_dim: int = 512
    """width of the random target features"""

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
# running statistics
# ---------------------------------------------------------------------------

class RunningMeanStd:
    """Chan et al.'s parallel variance update, as in `gym.wrappers.normalize`.

    Kept here rather than imported so the trainer does not depend on which gym
    version happens to expose it, and so the tests can drive it directly.
    """

    def __init__(self, epsilon=1e-4, shape=(), dtype=np.float64):
        self.mean = np.zeros(shape, dtype=dtype)
        self.var = np.ones(shape, dtype=dtype)
        self.count = epsilon

    def update(self, x):
        self.update_from_moments(np.mean(x, axis=0), np.var(x, axis=0), x.shape[0])

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        squared = m_a + m_b + np.square(delta) * self.count * batch_count / total_count
        self.mean = new_mean
        self.var = squared / total_count
        self.count = total_count


class RewardForwardFilter:
    """The running *discounted return*, not the running reward.

    RND normalises the intrinsic reward by `std(sum_k gamma^k r_int)`, which is
    what keeps the bonus scale stable as the predictor converges. Note there is
    no `done` masking: the intrinsic stream is non-episodic by construction.
    """

    def __init__(self, gamma):
        self.rewems = None
        self.gamma = gamma

    def update(self, rews):
        if self.rewems is None:
            self.rewems = rews
        else:
            self.rewems = self.rewems * self.gamma + rews
        return self.rewems


def normalize_rnd_obs(frame, mean, var, clip):
    """Whiten a single frame for the RND networks and clip it.

    `(x - mean) / sqrt(var)`, clipped to `+/- clip`. Note the reference divides
    by `sqrt(var)` with no epsilon; `var` starts at 1 and only grows, so this is
    safe, but it is *not* the usual `std + 1e-8`.
    """
    return ((frame - mean) / torch.sqrt(var)).clamp(-clip, clip)


# ---------------------------------------------------------------------------
# networks
# ---------------------------------------------------------------------------

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    """The reference's "param matched" policy: 448-wide trunk, two value heads.

    `extra_layer` and the `features + hidden` residual come straight from
    `cnn_policy_param_matched.py`; they are not decoration, they are what the
    name "param matched" refers to.
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
            layer_init(nn.Linear(64 * 7 * 7, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, 448)),
            nn.ReLU(),
        )
        self.extra_layer = nn.Sequential(layer_init(nn.Linear(448, 448), std=0.1), nn.ReLU())
        self.actor = nn.Sequential(
            layer_init(nn.Linear(448, 448), std=0.01),
            nn.ReLU(),
            layer_init(nn.Linear(448, envs.single_action_space.n), std=0.01),
        )
        self.critic_ext = layer_init(nn.Linear(448, 1), std=0.01)
        self.critic_int = layer_init(nn.Linear(448, 1), std=0.01)

    def get_action_and_value(self, x, action=None):
        hidden = self.network(x / 255.0)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        features = self.extra_layer(hidden)
        if action is None:
            action = probs.sample()
        return (
            action,
            probs.log_prob(action),
            probs.entropy(),
            self.critic_ext(features + hidden),
            self.critic_int(features + hidden),
        )

    def get_value(self, x):
        hidden = self.network(x / 255.0)
        features = self.extra_layer(hidden)
        return self.critic_ext(features + hidden), self.critic_int(features + hidden)


class RNDModel(nn.Module):
    """A frozen random target and a trainable predictor over one 84x84 frame.

    The predictor is deeper than the target on purpose: the target is a *fixed*
    random function, so making the predictor strictly more expressive means any
    residual error is attributable to lack of data rather than lack of capacity.
    LeakyReLU in the convolutions, ReLU in the head — the reference's choice.
    """

    def __init__(self, feature_dim=512):
        super().__init__()
        conv_output = 7 * 7 * 64

        self.predictor = nn.Sequential(
            layer_init(nn.Conv2d(1, 32, 8, stride=4)),
            nn.LeakyReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.LeakyReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.LeakyReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(conv_output, feature_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(feature_dim, feature_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(feature_dim, feature_dim)),
        )
        self.target = nn.Sequential(
            layer_init(nn.Conv2d(1, 32, 8, stride=4)),
            nn.LeakyReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.LeakyReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.LeakyReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(conv_output, feature_dim)),
        )
        for parameter in self.target.parameters():
            parameter.requires_grad = False

    def forward(self, next_obs):
        return self.predictor(next_obs), self.target(next_obs)


def intrinsic_reward(predict_feature, target_feature, reduction="mean"):
    """The novelty bonus.

    `mean` is `cnn_policy_param_matched.py:167`. `sum_half` is CleanRL's
    `sum(1) / 2`; it differs by a constant `feature_dim / 2` that the running
    normalisation removes.
    """
    squared_error = (target_feature - predict_feature).pow(2)
    if reduction == "mean":
        return squared_error.mean(-1)
    if reduction == "sum_half":
        return squared_error.sum(-1) / 2
    raise ValueError(f"unsupported intrinsic_reward_reduction: {reduction}")


def rnd_predictor_loss(predict_feature, target_feature, mask):
    """`sum(mask * per_sample_mse) / max(sum(mask), 1)`.

    The mask subsamples the minibatch. Dividing by `sum(mask)` rather than by
    the batch size keeps the *gradient scale* independent of
    `update_proportion`, so lowering it slows the predictor down without also
    shrinking its steps.
    """
    per_sample = F.mse_loss(predict_feature, target_feature.detach(), reduction="none").mean(-1)
    denominator = torch.clamp(mask.sum(), min=1.0)
    return (per_sample * mask).sum() / denominator


def dual_gae(rewards, curiosity_rewards, ext_values, int_values, dones, next_done,
             next_value_ext, next_value_int, gamma, int_gamma, gae_lambda):
    """Two GAEs: extrinsic episodic, intrinsic **non-episodic**.

    The intrinsic stream uses `nextnonterminal = 1` unconditionally. That is the
    reference's behaviour and it is deliberate: if dying reset the intrinsic
    return, an agent stuck in a well-explored region could farm the bonus by
    dying, and near a lethal novelty it would be taught that death is cheap.
    """
    num_steps = rewards.shape[0]
    ext_advantages = torch.zeros_like(rewards)
    int_advantages = torch.zeros_like(curiosity_rewards)
    ext_lastgaelam = 0
    int_lastgaelam = 0
    for t in reversed(range(num_steps)):
        if t == num_steps - 1:
            ext_nextnonterminal = 1.0 - next_done
            ext_nextvalues = next_value_ext
            int_nextvalues = next_value_int
        else:
            ext_nextnonterminal = 1.0 - dones[t + 1]
            ext_nextvalues = ext_values[t + 1]
            int_nextvalues = int_values[t + 1]
        int_nextnonterminal = 1.0

        ext_delta = rewards[t] + gamma * ext_nextvalues * ext_nextnonterminal - ext_values[t]
        int_delta = curiosity_rewards[t] + int_gamma * int_nextvalues * int_nextnonterminal - int_values[t]
        ext_advantages[t] = ext_lastgaelam = (
            ext_delta + gamma * gae_lambda * ext_nextnonterminal * ext_lastgaelam
        )
        int_advantages[t] = int_lastgaelam = (
            int_delta + int_gamma * gae_lambda * int_nextnonterminal * int_lastgaelam
        )
    return ext_advantages, int_advantages


if __name__ == "__main__":
    process_start = time.perf_counter()
    args = tyro.cli(Args)
    if args.num_envs < 1:
        raise ValueError("num_envs must be positive")
    if args.num_steps < 1:
        raise ValueError("num_steps must be positive")
    if args.num_minibatches < 1:
        raise ValueError("num_minibatches must be positive")
    if not 0.0 < args.update_proportion <= 1.0:
        raise ValueError("update_proportion must be in (0, 1]")
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
            [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)],
        )
    else:
        raise ValueError(f"unsupported environment backend: {args.env_backend}")
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    agent = Agent(envs).to(device)
    rnd_model = RNDModel(args.rnd_feature_dim).to(device)
    # One optimiser over the policy and the predictor. The frozen target is
    # excluded, so a stray `requires_grad` would show up as a shape mismatch
    # rather than silently training the target into the predictor.
    combined_parameters = list(agent.parameters()) + list(rnd_model.predictor.parameters())
    optimizer = optim.Adam(combined_parameters, lr=args.learning_rate, eps=1e-5)

    reward_rms = RunningMeanStd()
    obs_rms = RunningMeanStd(shape=(1, 1, 84, 84))
    discounted_reward = RewardForwardFilter(args.int_gamma)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs), dtype=torch.long).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    curiosity_rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    ext_values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    int_values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = to_tensor(next_obs, device)
    next_done = torch.zeros(args.num_envs).to(device)
    stats = EpisodeStats(args.solve_window)
    benchmark_start = None
    benchmark_start_step = None

    # Seed the observation statistics with a random agent. Skipped under
    # `--benchmark`, which measures the training loop, not this.
    if args.num_iterations_obs_norm_init > 0 and not args.benchmark:
        print("initializing observation normalization from a random agent...")
        collected = []
        for _ in range(args.num_steps * args.num_iterations_obs_norm_init):
            random_actions = torch.randint(
                0, int(envs.single_action_space.n), (args.num_envs,), device=device)
            random_obs, _, terminations, truncations, _ = step_env(envs, random_actions)
            random_obs = to_tensor(random_obs, device)
            collected.append(to_numpy(random_obs[:, 3, :, :]).reshape(-1, 1, 84, 84))
            if len(collected) * args.num_envs >= args.num_steps * args.num_envs:
                obs_rms.update(np.concatenate(collected, axis=0))
                collected = []
        if collected:
            obs_rms.update(np.concatenate(collected, axis=0))
        next_obs, _ = envs.reset(seed=args.seed)
        next_obs = to_tensor(next_obs, device)
        print("done.")

    obs_rms_mean = torch.from_numpy(obs_rms.mean).to(device, torch.float32)
    obs_rms_var = torch.from_numpy(obs_rms.var).to(device, torch.float32)

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
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value_ext, value_int = agent.get_action_and_value(next_obs)
                ext_values[step], int_values[step] = value_ext.flatten(), value_int.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = step_env(envs, action)
            next_done = done_tensor(terminations, truncations, device)
            rewards[step] = to_tensor(reward, device).view(-1)
            next_obs = to_tensor(next_obs, device)

            # The bonus is paid for the state we *arrived in*.
            with torch.no_grad():
                rnd_next_obs = normalize_rnd_obs(
                    next_obs[:, 3, :, :].reshape(args.num_envs, 1, 84, 84),
                    obs_rms_mean, obs_rms_var, args.obs_norm_clip).float()
                predict_feature, target_feature = rnd_model(rnd_next_obs)
                curiosity_rewards[step] = intrinsic_reward(
                    predict_feature, target_feature, args.intrinsic_reward_reduction)

            stats.update(completed_episode_infos(infos, next_done), global_step, writer)

        # Normalise the bonus by the running std of the *discounted* bonus.
        curiosity_per_env = np.array(
            [discounted_reward.update(step_rewards)
             for step_rewards in to_numpy(curiosity_rewards).T]
        )
        reward_rms.update_from_moments(
            np.mean(curiosity_per_env), np.std(curiosity_per_env) ** 2, len(curiosity_per_env))
        curiosity_rewards /= np.sqrt(reward_rms.var)

        with torch.no_grad():
            next_value_ext, next_value_int = agent.get_value(next_obs)
            ext_advantages, int_advantages = dual_gae(
                rewards, curiosity_rewards, ext_values, int_values, dones, next_done,
                next_value_ext.reshape(-1), next_value_int.reshape(-1),
                args.gamma, args.int_gamma, args.gae_lambda)
            ext_returns = ext_advantages + ext_values
            int_returns = int_advantages + int_values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape(-1)
        b_ext_advantages = ext_advantages.reshape(-1)
        b_int_advantages = int_advantages.reshape(-1)
        b_ext_returns = ext_returns.reshape(-1)
        b_int_returns = int_returns.reshape(-1)
        b_ext_values = ext_values.reshape(-1)
        b_advantages = b_int_advantages * args.int_coef + b_ext_advantages * args.ext_coef

        # Fold this rollout into the observation statistics, then rebuild the
        # RND inputs with the *updated* statistics -- the reference updates
        # before the epochs, not after.
        obs_rms.update(to_numpy(b_obs[:, 3, :, :]).reshape(-1, 1, 84, 84))
        obs_rms_mean = torch.from_numpy(obs_rms.mean).to(device, torch.float32)
        obs_rms_var = torch.from_numpy(obs_rms.var).to(device, torch.float32)
        rnd_next_obs = normalize_rnd_obs(
            b_obs[:, 3, :, :].reshape(-1, 1, 84, 84),
            obs_rms_mean, obs_rms_var, args.obs_norm_clip).float()

        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                predict_feature, target_feature = rnd_model(rnd_next_obs[mb_inds])
                mask = (torch.rand(len(mb_inds), device=device) < args.update_proportion).float()
                forward_loss = rnd_predictor_loss(predict_feature, target_feature, mask)

                _, newlogprob, entropy, new_ext_values, new_int_values = agent.get_action_and_value(
                    b_obs[mb_inds], b_actions[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                new_ext_values, new_int_values = new_ext_values.view(-1), new_int_values.view(-1)
                if args.clip_vloss:
                    ext_v_loss_unclipped = (new_ext_values - b_ext_returns[mb_inds]) ** 2
                    ext_v_clipped = b_ext_values[mb_inds] + torch.clamp(
                        new_ext_values - b_ext_values[mb_inds], -args.clip_coef, args.clip_coef)
                    ext_v_loss_clipped = (ext_v_clipped - b_ext_returns[mb_inds]) ** 2
                    ext_v_loss = 0.5 * torch.max(ext_v_loss_unclipped, ext_v_loss_clipped).mean()
                else:
                    ext_v_loss = 0.5 * ((new_ext_values - b_ext_returns[mb_inds]) ** 2).mean()
                # The intrinsic value loss is never clipped: its target scale
                # moves as the predictor learns, so old-value clipping would
                # anchor it to a scale that no longer exists.
                int_v_loss = 0.5 * ((new_int_values - b_int_returns[mb_inds]) ** 2).mean()
                v_loss = ext_v_loss + int_v_loss

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef + forward_loss

                optimizer.zero_grad()
                loss.backward()
                if args.max_grad_norm:
                    nn.utils.clip_grad_norm_(combined_parameters, args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        if writer is not None:
            writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
            writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
            writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
            writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
            writer.add_scalar("losses/rnd_forward_loss", forward_loss.item(), global_step)
            writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
            writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
            writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
            writer.add_scalar("charts/mean_curiosity_reward", curiosity_rewards.mean().item(), global_step)
            writer.add_scalar("charts/max_curiosity_reward", curiosity_rewards.max().item(), global_step)
            print("SPS:", int(global_step / (time.time() - start_time)))
            writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    if args.benchmark:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        benchmark_end = time.perf_counter()
        measured_steps = global_step - benchmark_start_step
        measured_seconds = benchmark_end - benchmark_start
        result = {
            "algorithm": "ppo_rnd",
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

    envs.close()
    if writer is not None:
        writer.close()
