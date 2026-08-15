# PPG (Phasic Policy Gradient) for Atari.
#
# Cobbe et al., ICML 2021, "Phasic Policy Gradient"
# (https://arxiv.org/abs/2009.04416). Ported from the authors' implementation,
# openai/phasic-policy-gradient (MIT), files `phasic_policy_gradient/ppg.py`,
# `ppo.py` and `train.py`.
#
# PPG's observation is that PPO forces one network to serve two masters. A
# shared torso trained on `pg_loss + vf_coef * vf_loss` lets value-fitting
# gradients perturb the policy, so you cannot train the value function harder
# than the policy can tolerate — even though the value function usually wants
# many more epochs than the policy does. PPG separates them in *time* rather
# than only in space:
#
#   Policy phase   N_pi = 32 iterations of ordinary PPO, one epoch each, on
#                  *separate* policy and value networks (`--arch dual`). Every
#                  rollout is kept.
#   Auxiliary phase  E_aux = 6 epochs over all 32 stored rollouts, minimising
#                  `beta_clone * KL(pi_old || pi) + 0.5 (V_aux - vtarg)^2
#                   + 0.5 (V_true - vtarg)^2`.
#
# The auxiliary phase is where the trick lives. `V_aux` is a value head grafted
# onto the *policy* encoder, so fitting it distils value features into the
# policy torso; the KL term simultaneously pins the policy where the policy
# phase left it. The policy gets the representation without the interference.
#
# Deviations from the reference, all forced by the domain and all flagged here:
#
#   * The paper is Procgen (64x64x3, IMPALA encoder, gamma 0.999). This is
#     Atari, so the default encoder is the Nature CNN and gamma is 0.99.
#     `--encoder impala` restores the paper's torso.
#   * The reference keeps `n_pi * num_envs * nstep` observations at
#     64 envs x 256 steps x 32 iterations. At Atari's 4x84x84 that buffer would
#     be ~118 GB, so the defaults here are 32 envs x 128 steps and the buffer is
#     stored as **uint8 on the CPU** (exact for Atari, whose frames are
#     integers) — 3.7 GB at the paper's `n_pi = 32`.
#
# Details from the reference that a from-the-paper implementation would miss,
# each pinned by `tests/test_ppg_equivalence.py`:
#
#   * minibatches are taken over *environments*, not shuffled transitions, so
#     each one carries whole 128-step trajectories;
#   * the advantage is normalised once over the entire rollout, before
#     minibatching, not per minibatch;
#   * the value loss is `vfcoef * (vpred - vtarg)^2` with **no leading 1/2** and
#     no PPO-style value clipping;
#   * heads are `NormedLinear(scale=0.1)`: rows rescaled to L2 norm 0.1, bias 0;
#   * the auxiliary phase runs on its own persistent Adam, which is *not* reset
#     between phases, and the policy phase's optimiser is likewise persistent.
import json
import math
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
    env_backend: str = "gymnasium"
    """environment backend: `gymnasium`, `cule`, or `envpool`"""
    cule_device: str = "auto"
    """device for the CuLE backend: `auto`, `cpu`, or a CUDA device string"""

    # Algorithm specific arguments
    env_id: str = "BreakoutNoFrameskip-v4"
    """the id of the environment"""
    total_timesteps: int = 10000000
    """total timesteps of the experiments"""
    learning_rate: float = 5e-4
    """policy-phase Adam learning rate (reference default)"""
    aux_learning_rate: float = 5e-4
    """auxiliary-phase Adam learning rate (reference default)"""
    num_envs: int = 32
    """the number of parallel game environments"""
    num_steps: int = 128
    """rollout length per environment (the reference uses 256 on Procgen)"""
    anneal_lr: bool = False
    """the reference does not anneal; kept off by default to match it"""
    gamma: float = 0.99
    """the discount factor gamma (the reference uses 0.999 on Procgen)"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 8
    """environment-wise minibatches per policy-phase epoch (`nminibatch`)"""
    n_epoch_pi: int = 1
    """policy epochs per policy-phase iteration"""
    n_epoch_vf: int = 1
    """value epochs per policy-phase iteration"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient (`clip_param`)"""
    ent_coef: float = 0.01
    """coefficient of the entropy (`entcoef`)"""
    vf_coef: float = 0.5
    """coefficient of the value function (`vfcoef`)"""
    kl_penalty: float = 0.0
    """coefficient on `0.5 * logratio^2` in the policy loss"""
    max_grad_norm: float = 0.0
    """gradient clipping; 0 disables it, as the reference does"""

    n_pi: int = 32
    """policy-phase iterations between auxiliary phases"""
    n_aux_epochs: int = 6
    """epochs over the auxiliary buffer per auxiliary phase"""
    aux_mbsize: int = 4
    """whole trajectories per auxiliary minibatch"""
    beta_clone: float = 1.0
    """weight on the KL that pins the policy during the auxiliary phase"""
    vf_true_weight: float = 1.0
    """weight on the true value head's auxiliary-phase loss"""
    arch: str = "dual"
    """`dual` (separate encoders), `shared`, or `detach`"""
    encoder: str = "nature"
    """`nature` (Atari default) or `impala` (the reference's Procgen torso)"""
    aux_buffer_device: str = "cpu"
    """where the auxiliary observation buffer lives; `cpu` keeps VRAM free"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
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
# network
# ---------------------------------------------------------------------------

def normed_linear(in_features, out_features, scale=1.0, bias=True):
    """`torch_util.NormedLinear`: rescale each output row to L2 norm `scale`.

    Not orthogonal init and not the usual `sqrt(2)` gain — every head in PPG is
    built this way at `scale=0.1`, which starts the policy near-uniform and the
    value heads near-zero.
    """
    layer = nn.Linear(in_features, out_features, bias=bias)
    with torch.no_grad():
        layer.weight.data *= scale / layer.weight.norm(dim=1, p=2, keepdim=True)
        if bias:
            layer.bias.data *= 0
    return layer


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv0 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        inputs = x
        x = F.relu(x)
        x = self.conv0(x)
        x = F.relu(x)
        x = self.conv1(x)
        return x + inputs


class ConvSequence(nn.Module):
    def __init__(self, input_shape, out_channels):
        super().__init__()
        self._input_shape = input_shape
        self._out_channels = out_channels
        self.conv = nn.Conv2d(input_shape[0], out_channels, 3, padding=1)
        self.res_block0 = ResidualBlock(out_channels)
        self.res_block1 = ResidualBlock(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        x = self.res_block0(x)
        return self.res_block1(x)

    def get_output_shape(self):
        _c, h, w = self._input_shape
        return (self._out_channels, (h + 1) // 2, (w + 1) // 2)


def build_encoder(obs_shape, encoder, hidden=512):
    if encoder == "nature":
        return nn.Sequential(
            layer_init(nn.Conv2d(obs_shape[0], 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 7 * 7, hidden)),
            nn.ReLU(),
        )
    if encoder == "impala":
        shape = obs_shape
        layers = []
        for out_channels in (16, 32, 32):
            sequence = ConvSequence(shape, out_channels)
            shape = sequence.get_output_shape()
            layers.append(sequence)
        layers += [nn.ReLU(), nn.Flatten(), layer_init(nn.Linear(int(np.prod(shape)), hidden)), nn.ReLU()]
        return nn.Sequential(*layers)
    raise ValueError(f"unsupported encoder: {encoder}")


class PhasicValueAgent(nn.Module):
    """`ppg.PhasicValueModel`, without the recurrent/tree plumbing.

    `dual`   two encoders; GAE uses the value network's head.
    `shared` one encoder; the true value head sits on it.
    `detach` one encoder; the true value head's gradient is stopped, so the
             policy phase cannot push value error into the torso at all.

    In every arch the auxiliary head `aux_vf_head` hangs off the *policy*
    encoder — that is the channel the auxiliary phase distils through.
    """

    def __init__(self, envs, arch="dual", encoder="nature", hidden=512):
        super().__init__()
        if arch not in ("dual", "shared", "detach"):
            raise ValueError(f"unsupported arch: {arch}")
        self.arch = arch
        self.detach_value_head = arch == "detach"
        obs_shape = envs.single_observation_space.shape
        num_actions = int(envs.single_action_space.n)

        self.pi_encoder = build_encoder(obs_shape, encoder, hidden)
        self.vf_encoder = build_encoder(obs_shape, encoder, hidden) if arch == "dual" else None

        self.pi_head = normed_linear(hidden, num_actions, scale=0.1)
        self.aux_vf_head = normed_linear(hidden, 1, scale=0.1)
        self.vf_head = normed_linear(hidden, 1, scale=0.1)

    def _encode(self, x):
        x = x / 255.0
        pi_latent = self.pi_encoder(x)
        vf_latent = self.vf_encoder(x) if self.arch == "dual" else pi_latent
        return pi_latent, vf_latent

    def forward(self, x):
        """Returns `(logits, vpred_true, vpred_aux)`."""
        pi_latent, vf_latent = self._encode(x)
        if self.detach_value_head:
            vf_latent = vf_latent.detach()
        logits = self.pi_head(pi_latent)
        vpred_true = self.vf_head(vf_latent).squeeze(-1)
        vpred_aux = self.aux_vf_head(pi_latent).squeeze(-1)
        return logits, vpred_true, vpred_aux

    def get_value(self, x):
        return self.forward(x)[1]

    def get_action_and_value(self, x, action=None):
        logits, vpred_true, _ = self.forward(x)
        distribution = Categorical(logits=logits)
        if action is None:
            action = distribution.sample()
        return action, distribution.log_prob(action), distribution.entropy(), vpred_true


# ---------------------------------------------------------------------------
# advantages and losses
# ---------------------------------------------------------------------------

def compute_gae(vpred, reward, first, gamma, gae_lambda):
    """`ppo.compute_gae`, in the reference's `[nenv, nstep]` layout.

    `vpred` and `first` are `[nenv, nstep + 1]`; `reward` is `[nenv, nstep]`.
    `first[:, t]` marks that timestep `t` *begins* an episode, so
    `notlast = 1 - first[:, t + 1]` is the reference's spelling of
    "the transition out of step t did not end the episode".
    """
    nenv, nstep = reward.shape
    assert vpred.shape == first.shape == (nenv, nstep + 1)
    advantages = torch.zeros(nenv, nstep, dtype=vpred.dtype, device=vpred.device)
    lastgaelam = 0
    for t in reversed(range(nstep)):
        notlast = 1.0 - first[:, t + 1]
        nextvalue = vpred[:, t + 1]
        delta = reward[:, t] + notlast * gamma * nextvalue - vpred[:, t]
        advantages[:, t] = lastgaelam = delta + notlast * gamma * gae_lambda * lastgaelam
    vtarg = vpred[:, :-1] + advantages
    return advantages, vtarg


def normalize_advantage(advantages):
    """Whitening over the *whole* rollout, before minibatching.

    `(adv - mean) / (sqrt(var) + 1e-8)` — note the reference divides by
    `sqrt(var) + eps`, not `std + eps`; they coincide, but the unbiased/biased
    variance choice does not. `torch.var` defaults to unbiased, which is what
    `mpi_moments` computes.
    """
    mean = advantages.mean()
    variance = advantages.var(unbiased=True)
    return (advantages - mean) / (math.sqrt(float(variance)) + 1e-8)


def ppo_losses(logits, vpred, actions, old_logprobs, advantages, vtarg,
               clip_coef, ent_coef, vf_coef, kl_penalty):
    """`ppo.compute_losses`, returning `(pi_loss, vf_loss, diagnostics)`.

    Note `vf_loss = vf_coef * (vpred - vtarg)^2` — no leading 1/2, and no PPO
    value clipping. The customary 1/2 is not folded into `vf_coef` here either:
    the reference really does use `0.5 * mse * 2`, i.e. plain `mse * 0.5`.
    """
    distribution = Categorical(logits=logits)
    new_logprobs = distribution.log_prob(actions)
    logratio = new_logprobs - old_logprobs
    ratio = logratio.exp()

    if clip_coef > 0:
        pg_losses = torch.max(
            -advantages * ratio,
            -advantages * torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef),
        )
    else:
        pg_losses = -advantages * ratio

    entropy = distribution.entropy().mean()
    pg_loss = pg_losses.mean()
    kl_loss = kl_penalty * 0.5 * (logratio**2).mean()
    pi_loss = -entropy * ent_coef + pg_loss + kl_loss
    vf_loss = vf_coef * ((vpred - vtarg) ** 2).mean()

    with torch.no_grad():
        diagnostics = {
            "entropy": entropy.detach(),
            "pg_loss": pg_loss.detach(),
            "approxkl": 0.5 * (logratio**2).mean(),
            "clipfrac": ((ratio - 1).abs() > clip_coef).float().mean(),
        }
    return pi_loss, vf_loss, diagnostics


def aux_losses(logits, vpred_true, vpred_aux, old_logits, vtarg,
               beta_clone, vf_true_weight):
    """`ppg.aux_train` + `PhasicValueModel.compute_aux_loss`.

    `KL(pi_old || pi)`, the *forward* KL with the frozen policy first — it
    penalises the new policy for putting low probability where the old one put
    high probability, which is what "pin the policy in place" needs.
    """
    old_distribution = Categorical(logits=old_logits)
    new_distribution = Categorical(logits=logits)
    pol_distance = torch.distributions.kl_divergence(old_distribution, new_distribution).mean()
    vf_aux = 0.5 * ((vpred_aux - vtarg) ** 2).mean()
    vf_true = 0.5 * ((vpred_true - vtarg) ** 2).mean()
    loss = beta_clone * pol_distance + vf_aux + vf_true_weight * vf_true
    return loss, pol_distance.detach(), vf_aux.detach(), vf_true.detach()


if __name__ == "__main__":
    process_start = time.perf_counter()
    args = tyro.cli(Args)
    if args.num_envs < 1:
        raise ValueError("num_envs must be positive")
    if args.num_steps < 1:
        raise ValueError("num_steps must be positive")
    if args.num_minibatches < 1 or args.num_envs % args.num_minibatches:
        raise ValueError("num_envs must be a positive multiple of num_minibatches")
    if args.n_pi < 1:
        raise ValueError("n_pi must be positive")
    if args.aux_mbsize < 1:
        raise ValueError("aux_mbsize must be positive")
    if args.benchmark_warmup_iterations < 0:
        raise ValueError("benchmark_warmup_iterations cannot be negative")
    if args.benchmark_measure_iterations < 1:
        raise ValueError("benchmark_measure_iterations must be positive")
    args.batch_size = int(args.num_envs * args.num_steps)
    args.num_iterations = args.total_timesteps // args.batch_size
    if args.benchmark:
        args.num_iterations = args.benchmark_warmup_iterations + args.benchmark_measure_iterations
    envs_per_minibatch = args.num_envs // args.num_minibatches
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
    aux_device = torch.device(args.aux_buffer_device)
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
    num_actions = int(envs.single_action_space.n)
    obs_shape = envs.single_observation_space.shape

    agent = PhasicValueAgent(envs, args.arch, args.encoder).to(device)
    # Two persistent optimisers, exactly as the reference: `ppo.learn` keeps its
    # `opts` in `learn_state` across iterations and `ppg.learn` holds `aux_state`
    # across auxiliary phases. Recreating either would throw away Adam moments.
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate)
    aux_optimizer = optim.Adam(agent.parameters(), lr=args.aux_learning_rate)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + obs_shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs), dtype=torch.long).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    next_dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # The auxiliary buffer: `n_pi` rollouts of observations and value targets.
    # uint8 because Atari frames are integers, and on the CPU because at the
    # paper's `n_pi = 32` this is several GB.
    aux_obs = torch.zeros((args.n_pi, args.num_steps, args.num_envs) + obs_shape,
                          dtype=torch.uint8, device=aux_device)
    aux_vtarg = torch.zeros((args.n_pi, args.num_steps, args.num_envs),
                            dtype=torch.float32, device=aux_device)

    # TRY NOT TO MODIFY: start the game
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
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value
            actions[step] = action
            logprobs[step] = logprob

            next_obs, reward, terminations, truncations, infos = step_env(envs, action)
            next_done = done_tensor(terminations, truncations, device)
            rewards[step] = to_tensor(reward, device).view(-1)
            next_dones[step] = next_done
            next_obs = to_tensor(next_obs, device)

            stats.update(completed_episode_infos(infos, next_done), global_step, writer)

        # GAE in the reference's [nenv, nstep] layout. `first[:, t + 1]` is the
        # done produced by step t, and `first[:, 0]` never enters the recursion.
        with torch.no_grad():
            bootstrap_value = agent.get_value(next_obs)
            vpred = torch.cat([values, bootstrap_value.unsqueeze(0)], dim=0).transpose(0, 1)
            first = torch.cat([torch.zeros_like(next_dones[:1]), next_dones], dim=0).transpose(0, 1)
            advantages, vtarg = compute_gae(
                vpred, rewards.transpose(0, 1), first, args.gamma, args.gae_lambda)
            advantages = normalize_advantage(advantages)
            advantages = advantages.transpose(0, 1)  # back to [nstep, nenv]
            vtarg = vtarg.transpose(0, 1)

        aux_slot = (iteration - 1) % args.n_pi
        aux_obs[aux_slot] = obs.to(torch.uint8).to(aux_device)
        aux_vtarg[aux_slot] = vtarg.to(aux_device)

        # --- policy phase -------------------------------------------------
        # Minibatches are over environments, so each carries whole trajectories.
        for _ in range(max(args.n_epoch_pi, args.n_epoch_vf)):
            permutation = torch.randperm(args.num_envs, device=device)
            for start in range(0, args.num_envs, envs_per_minibatch):
                env_indices = permutation[start : start + envs_per_minibatch]
                mb_obs = obs[:, env_indices].flatten(0, 1)
                logits, vpred_true, _ = agent(mb_obs)
                pi_loss, vf_loss, diagnostics = ppo_losses(
                    logits,
                    vpred_true,
                    actions[:, env_indices].flatten(),
                    logprobs[:, env_indices].flatten(),
                    advantages[:, env_indices].flatten(),
                    vtarg[:, env_indices].flatten(),
                    args.clip_coef,
                    args.ent_coef,
                    args.vf_coef,
                    args.kl_penalty,
                )
                optimizer.zero_grad()
                (pi_loss + vf_loss).backward()
                if args.max_grad_norm > 0:
                    nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

        # --- auxiliary phase ----------------------------------------------
        if iteration % args.n_pi == 0:
            filled = min(iteration, args.n_pi)
            # `compute_presleep_outputs`: freeze the policy the phase must
            # preserve. Recomputed here rather than reused from the rollout,
            # because the policy phase has moved since those logits were made.
            old_logits = torch.zeros(
                (filled, args.num_steps, args.num_envs, num_actions),
                dtype=torch.float32, device=aux_device)
            with torch.no_grad():
                for slot in range(filled):
                    for env_start in range(0, args.num_envs, envs_per_minibatch):
                        env_slice = slice(env_start, env_start + envs_per_minibatch)
                        batch = aux_obs[slot, :, env_slice].to(device, torch.float32).flatten(0, 1)
                        logits, _, _ = agent(batch)
                        old_logits[slot, :, env_slice] = logits.view(
                            args.num_steps, -1, num_actions).to(aux_device)

            # Minibatches are (rollout, environment) pairs, each a whole
            # trajectory -- the reference's `make_minibatches`.
            pairs = torch.cartesian_prod(torch.arange(filled), torch.arange(args.num_envs))
            for _ in range(args.n_aux_epochs):
                for chunk in torch.randperm(len(pairs)).split(args.aux_mbsize):
                    selected = pairs[chunk]
                    slot_index, env_index = selected[:, 0], selected[:, 1]
                    mb_obs = aux_obs[slot_index, :, env_index].to(device, torch.float32)
                    mb_vtarg = aux_vtarg[slot_index, :, env_index].to(device)
                    mb_old_logits = old_logits[slot_index, :, env_index].to(device)

                    logits, vpred_true, vpred_aux = agent(mb_obs.flatten(0, 1))
                    loss, pol_distance, vf_aux, vf_true = aux_losses(
                        logits,
                        vpred_true,
                        vpred_aux,
                        mb_old_logits.flatten(0, 1),
                        mb_vtarg.flatten(),
                        args.beta_clone,
                        args.vf_true_weight,
                    )
                    aux_optimizer.zero_grad()
                    loss.backward()
                    if args.max_grad_norm > 0:
                        nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                    aux_optimizer.step()

            if writer is not None:
                writer.add_scalar("losses/aux_pol_distance", pol_distance.item(), global_step)
                writer.add_scalar("losses/aux_vf_aux", vf_aux.item(), global_step)
                writer.add_scalar("losses/aux_vf_true", vf_true.item(), global_step)

        if writer is not None:
            writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
            writer.add_scalar("losses/value_loss", vf_loss.item(), global_step)
            writer.add_scalar("losses/policy_loss", diagnostics["pg_loss"].item(), global_step)
            writer.add_scalar("losses/entropy", diagnostics["entropy"].item(), global_step)
            writer.add_scalar("losses/approx_kl", diagnostics["approxkl"].item(), global_step)
            writer.add_scalar("losses/clipfrac", diagnostics["clipfrac"].item(), global_step)
            print("SPS:", int(global_step / (time.time() - start_time)))
            writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    if args.benchmark:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        benchmark_end = time.perf_counter()
        measured_steps = global_step - benchmark_start_step
        measured_seconds = benchmark_end - benchmark_start
        result = {
            "algorithm": "ppg",
            "arch": args.arch,
            "backend": args.env_backend,
            "batch_size": args.batch_size,
            "benchmark": "full_training_loop",
            "compile": False,
            "encoder": args.encoder,
            "env_id": args.env_id,
            "measure_iterations": args.benchmark_measure_iterations,
            "measured_seconds": measured_seconds,
            "measured_steps": measured_steps,
            "n_aux_epochs": args.n_aux_epochs,
            "n_pi": args.n_pi,
            "num_envs": args.num_envs,
            "num_minibatches": args.num_minibatches,
            "num_steps": args.num_steps,
            "peak_cuda_memory_mb": (
                torch.cuda.max_memory_allocated(device) / (1024**2) if device.type == "cuda" else 0.0
            ),
            "process_seconds": benchmark_end - process_start,
            "schema_version": 1,
            "sps": measured_steps / measured_seconds,
            "warmup_iterations": args.benchmark_warmup_iterations,
        }
        print(f"BENCHMARK_RESULT {json.dumps(result, sort_keys=True)}", flush=True)

    envs.close()
    if writer is not None:
        writer.close()
