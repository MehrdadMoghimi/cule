# PPO scaled to many parallel environments, for Atari.
#
# Beukman, Khetarpal, Zheng, Dabney, Foerster, Dennis and Lyle, RLJ/RLC 2026,
# "Preventing Learning Stagnation in PPO by Scaling to 1 Million Parallel
# Environments" (https://rlj.cs.umass.edu/2026/papers/Paper19.html).
#
# The paper's claim is about *why* PPO plateaus, not about a new loss. Model the
# outer loop -- rollout, then K epochs, then a new rollout -- as plain stochastic
# optimisation. Then two quantities decide whether training converges or thrashes:
#
#     outer step size    how far the policy is allowed to move per iteration,
#                        set by the regularisation towards the behaviour policy
#     update noise       how well the sampled batch's loss proxies the true
#                        objective, set by how much data is collected per update
#
# A plateau in this regime is the RL twin of SGD with too large a learning rate:
# the loss stops falling while the gradient norm and the update magnitude stay
# large, because the iterate bounces around a local optimum instead of settling.
# The paper's Figure 3 makes the diagnosis testable -- raise the regularisation
# mid-run on a plateaued agent and it immediately resumes learning; lower it again
# and it drops straight back to the plateau.
#
# Three separable mechanisms come out of that, and this file implements all three.
#
# 1. PPO-EWMA (`--proximal-com`), from Hilton, Cobbe and Schulman, 2022,
#    "Batch size-invariance for policy optimization" (arXiv:2110.00641), which the
#    paper adopts as its analysis instrument. PPO's behaviour policy does two
#    unrelated jobs: it is the importance-sampling denominator, *and* it is the
#    reference the clip regularises towards. Splitting them lets the second be set
#    independently of the data collection:
#
#        L = E[ (pi_prox / pi_behaviour) * min(r_prox, clip(r_prox, 1-e, 1+e)) * A ]
#        r_prox = pi_theta / pi_prox
#
#    where `pi_prox` is an EWMA of the current weights, refreshed after every
#    minibatch step. Its centre of mass, `COM = 1/(1 - beta) - 1`, is the age of
#    the regularisation target in minibatch steps and is the direct handle on the
#    outer step size -- larger COM means an older, stronger reference. At `e = inf`
#    the two ratios multiply back to the ordinary IS ratio, which is why this is a
#    decomposition of PPO rather than a different algorithm.
#
# 2. The scaling recipe (Section 5), which is the paper's practical
#    recommendation and what `--minibatch-size` exists to express. Adding parallel
#    environments grows the rollout, and that data has to be partitioned somehow.
#    Of the three options -- more minibatches, bigger minibatches, or bigger
#    minibatches with a sqrt-scaled learning rate -- the paper finds that holding
#    the *inner* optimisation fixed wins: keep the learning rate and the minibatch
#    size where they were and let the number of minibatches grow. So this file is
#    parameterised by minibatch **size**, not minibatch count, and the count is
#    derived. `--num-minibatches` is still accepted, and pins the count instead, to
#    reproduce the losing arm of Figure 9.
#
# 3. DDR (Section 4.4), the Data to Divergence Ratio: data points per unit KL
#    from the behaviour policy, unifying the two levers above into one number.
#    Low DDR predicts an early plateau, high DDR predicts learning too slowly to
#    reach the ceiling in budget, and the optimum *moves up* as the interaction
#    budget grows -- which is the paper's argument for why a configuration tuned
#    at 100M steps stagnates at 10B. It is logged every iteration as
#    `charts/ddr`, free of charge, off the KL estimator PPO already computes.
#
# At the defaults this file is `ppo_atari.py`: `--proximal-com` unset means no
# EWMA and the ordinary clipped surrogate, and `--minibatch-size 256` at 8 envs
# and 128 steps is the parent's 4 minibatches. That reduction is checked
# bit-exactly in `tests/test_ppo_scale_equivalence.py`, because a paper about
# avoiding plateaus is only measurable against a baseline it can reproduce.
#
# Two honest notes.
#
# * The paper's headline is 1M parallel environments on Kinetix across 128 GPUs,
#   sustained to a trillion transitions. Nothing here reaches that; Atari on one
#   GPU with CuLE gets to the low thousands of environments. What transfers is the
#   mechanism and the recipe, and `--proximal-com` is what makes the plateau
#   diagnosis runnable at this scale -- Figure 3's raise-the-COM intervention needs
#   no extra hardware at all.
# * DDR's exact normalisation is not written down. The paper defines it in words
#   as "the number of data points per unit KL divergence from the behaviour
#   policy" and never gives the formula, and Figure 8's absolute range does not
#   pin it down either. This file takes the whole rollout as the data and the
#   end-of-inner-loop KL as the divergence, so `ddr = batch_size / kl_behaviour`,
#   and logs `losses/kl_behaviour` alongside it so any other normalisation can be
#   recovered from the logs. That is a reading, not a transcription.
import json
import math
import os
import random
import sys
import time
from copy import deepcopy
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
    """the learning rate of the optimizer; the paper's recipe holds this fixed as `num_envs` grows"""
    num_envs: int = 8
    """the number of parallel game environments -- the axis the paper scales"""
    num_steps: int = 128
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    minibatch_size: int = 256
    """the mini-batch size, held fixed as `num_envs` grows (the paper's recipe)"""
    num_minibatches: int = None
    """pin the number of mini-batches instead, letting the size scale (the losing arm of Figure 9)"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.1
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function"""
    ent_coef: float = 0.01
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

    # PPO-EWMA arguments (Hilton et al. 2022, the paper's analysis instrument)
    proximal_com: float = None
    """centre of mass of the proximal policy in minibatch steps; unset means plain PPO"""
    imp_samp_max: float = 100.0
    """clip the importance-sampling ratio at this value for numerical stability; 0 disables"""

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


def com_to_decay(com):
    """Convert a centre of mass in minibatch steps to an EWMA decay.

    `COM = 1/(1 - beta) - 1`, so `beta = COM / (COM + 1)`. The paper's worked
    example: a COM of 1024 is `beta = 1024/1025 ~ 0.99902`, which it calls
    comparable to standard PPO at 256 minibatches and 8 epochs, since PPO
    refreshes its behaviour policy every 2048 minibatch steps and so regularises
    towards a target of average age 1024.
    """
    if com < 0:
        raise ValueError("proximal_com cannot be negative")
    return com / (com + 1.0)


def decay_to_com(decay):
    """Inverse of `com_to_decay`."""
    if not 0.0 <= decay < 1.0:
        raise ValueError("ewma decay must be in [0, 1)")
    return 1.0 / (1.0 - decay) - 1.0


class ProximalPolicy(nn.Module):
    """An EWMA-lagged copy of the agent: the reference the clip regularises towards.

    This is `EwmaModel` from openai/ppo-ewma (MIT), and the bias correction it
    carries is the part worth spelling out. The naive recursion
    `p_ewma <- beta*p_ewma + (1 - beta)*p` is anchored to whatever the weights
    happened to be at initialisation, and at the COMs this paper cares about
    (1024, 4096) that anchor takes tens of thousands of minibatch steps to decay
    out -- the early training the plateau analysis is about would be measured
    against noise. Tracking the accumulated weight instead makes the EWMA an exact
    weighted mean of the weights seen *so far* from the very first step:

        w' = decay*w + 1
        p_ewma <- p_ewma * (decay*w / w') + p / w'

    `w` rises to `1/(1 - decay)`, at which point the update is the naive one, so
    this differs from the paper's stated recursion only in the transient -- which
    is exactly where it matters. Same idea as Adam's bias correction.

    The buffers are cloned from the agent, not re-initialised, so `pi_prox` starts
    equal to `pi_theta` and the first update is already a valid average.
    """

    def __init__(self, model, decay):
        super().__init__()
        if not 0.0 <= decay < 1.0:
            raise ValueError("ewma decay must be in [0, 1)")
        self.decay = decay
        self.model = deepcopy(model)
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.total_weight = 1.0

    @torch.no_grad()
    def update(self, model, decay=None):
        decay = self.decay if decay is None else decay
        new_total_weight = decay * self.total_weight + 1.0
        decayed_weight_ratio = decay * self.total_weight / new_total_weight
        for target, source in zip(self.model.parameters(), model.parameters()):
            target.mul_(decayed_weight_ratio).add_(source.detach() / new_total_weight)
        self.total_weight = new_total_weight

    @torch.no_grad()
    def reset(self, model):
        """Snap the proximal policy back onto the current one."""
        self.update(model, decay=0.0)

    @torch.no_grad()
    def log_prob(self, x, action):
        return Categorical(logits=self.model.actor(self.model.network(x / 255.0))).log_prob(action)


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    """The Nature CNN with policy and value heads; identical to `ppo_atari.py`."""

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
        self.critic = layer_init(nn.Linear(512, 1), std=1)

    def get_value(self, x):
        return self.critic(self.network(x / 255.0))

    def get_action_and_value(self, x, action=None):
        hidden = self.network(x / 255.0)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)


def resolve_minibatching(batch_size, minibatch_size, num_minibatches):
    """Partition a rollout, by fixed minibatch size or by fixed minibatch count.

    The paper's recipe is the first: hold the learning rate and the minibatch
    size where they were and let the *number* of minibatches grow with the number
    of parallel environments, so extra data buys extra optimisation steps rather
    than bigger ones. Section 5 finds this beats both alternatives, and Appendix D
    shows the cost of getting it wrong -- Kinetix at 1M environments keeping the
    default 32 minibatches is substantially worse than scaling the count, and the
    largest setting runs out of memory outright.

    Passing `num_minibatches` pins the count instead and lets the size scale,
    which is the "Fix LR | Scale Minibatch Size" arm of Figure 9. It is supported
    so the comparison is reproducible, not because it is recommended.

    Returns `(minibatch_size, num_minibatches)`; the product need not equal
    `batch_size`, since the loop's final minibatch is short when it does not
    divide evenly.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if num_minibatches is not None:
        if num_minibatches < 1:
            raise ValueError("num_minibatches must be positive")
        if num_minibatches > batch_size:
            raise ValueError("num_minibatches cannot exceed num_envs * num_steps")
        return int(batch_size // num_minibatches), int(num_minibatches)
    if minibatch_size < 1:
        raise ValueError("minibatch_size must be positive")
    if minibatch_size > batch_size:
        raise ValueError(
            f"minibatch_size ({minibatch_size}) exceeds the rollout ({batch_size}); "
            "raise --num-envs or --num-steps, or lower --minibatch-size"
        )
    # Ceil, because a short trailing minibatch is still a gradient step: the loop
    # strides the shuffled indices by `minibatch_size` and takes what is left.
    return int(minibatch_size), int(-(-batch_size // minibatch_size))


def data_to_divergence_ratio(batch_size, kl_behaviour):
    """DDR (Section 4.4): data points per unit KL from the behaviour policy.

    The single number the paper's analysis reduces to. Both plateau modes live on
    this axis -- too low and the outer step is large relative to the noise, so the
    policy thrashes and stalls early; too high and it converges too slowly to
    reach its ceiling in budget. The optimum is not fixed: Figure 8 shows it
    shifting right as the interaction budget grows, which is why a configuration
    tuned at 100M steps stagnates at 10B, and why adding environments (which
    raises DDR from both sides at once) is the paper's recommended fix.

    Returns NaN for a non-positive KL rather than dividing: an exactly-zero KL
    means the policy did not move at all, which is not an infinitely good ratio.
    """
    if not kl_behaviour > 0 or not math.isfinite(kl_behaviour):
        return float("nan")
    return batch_size / kl_behaviour


def policy_surrogate(advantages, newlogp, behaviour_logp, proximal_logp, clip_coef, imp_samp_max):
    """The clipped surrogate, decoupled when a proximal policy is supplied.

    With `proximal_logp` unset this is `ppo_atari.py`'s expression, character for
    character, so the reduction to PPO is exact rather than approximate.

    With it set, this is Equation 4 -- the clip and the KL are measured against
    `pi_prox` while the importance weight stays against `pi_behaviour`:

        min( r_is, clip(r_prox, 1-e, 1+e) * pi_prox/pi_behaviour ) * A

    Two things are load-bearing and both come from the reference implementation.
    The ratios are formed in log space and only exponentiated once at the end,
    because a stale behaviour policy makes `pi_theta/pi_behaviour` large enough to
    overflow if built by division. And `imp_samp_max` bounds that ratio by lifting
    the *denominator* -- `logp_adj = max(logp, newlogp - log(max))` -- which caps
    the weight without touching its gradient path through `newlogp`.

    Returns `(pg_loss, logratio)`, where `logratio` is measured against whichever
    reference the clip used, since that is the KL the regulariser controls.
    """
    if proximal_logp is None:
        logratio = newlogp - behaviour_logp
        ratio = logratio.exp()
        surrogate = torch.max(
            -advantages * ratio,
            -advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef),
        )
        return surrogate.mean(), logratio

    logratio = newlogp - proximal_logp
    logp_adj = behaviour_logp
    if imp_samp_max > 0:
        logp_adj = torch.max(behaviour_logp, newlogp.detach() - math.log(imp_samp_max))
    surrogate = -advantages * torch.exp(newlogp - logp_adj)
    if clip_coef > 0:
        clipped_logratio = torch.clamp(logratio, math.log(1.0 - clip_coef), math.log(1.0 + clip_coef))
        surrogate = torch.max(surrogate, -advantages * torch.exp(clipped_logratio + proximal_logp - logp_adj))
    return surrogate.mean(), logratio


if __name__ == "__main__":
    process_start = time.perf_counter()
    args = tyro.cli(Args)
    if args.num_envs < 1 or args.num_steps < 1:
        raise ValueError("num_envs and num_steps must be positive")
    if args.benchmark_warmup_iterations < 0 or args.benchmark_measure_iterations < 1:
        raise ValueError("invalid benchmark window")
    if args.imp_samp_max < 0:
        raise ValueError("imp_samp_max cannot be negative")
    if args.imp_samp_max > 0 and args.imp_samp_max < 1:
        raise ValueError("imp_samp_max below 1 would clip the on-policy ratio itself")
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size, args.num_minibatches = resolve_minibatching(
        args.batch_size, args.minibatch_size, args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    if args.benchmark:
        args.num_iterations = args.benchmark_warmup_iterations + args.benchmark_measure_iterations
    ewma_decay = None if args.proximal_com is None else com_to_decay(args.proximal_com)
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
    proximal = None if ewma_decay is None else ProximalPolicy(agent, ewma_decay).to(device)
    if proximal is not None:
        print(
            f"PPO-EWMA: proximal COM {args.proximal_com} minibatch steps "
            f"(decay {ewma_decay:.8f}), refreshed every minibatch"
        )
    print(
        f"rollout {args.batch_size} = {args.num_envs} envs x {args.num_steps} steps -> "
        f"{args.num_minibatches} minibatches of {args.minibatch_size} x {args.update_epochs} epochs"
    )

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs), dtype=torch.long).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

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
            dones[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            next_obs, reward, terminations, truncations, infos = step_env(envs, action)
            next_done = done_tensor(terminations, truncations, device)
            rewards[step] = to_tensor(reward, device).view(-1)
            next_obs = to_tensor(next_obs, device)

            stats.update(completed_episode_infos(infos, next_done), global_step, writer)

        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        b_inds = np.arange(args.batch_size)
        clipfracs = []
        # DDR's divergence term. Accumulated over the final epoch only, where each
        # minibatch is visited exactly once, so this averages the behaviour-policy
        # KL over the whole rollout at the end of the inner loop -- which is the
        # deviation the outer step size actually produced -- for no extra passes.
        kl_behaviour_sum = 0.0
        kl_behaviour_count = 0
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            final_epoch = epoch == args.update_epochs - 1
            for start in range(0, args.batch_size, args.minibatch_size):
                mb_inds = b_inds[start : start + args.minibatch_size]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                mb_behaviour_logp = b_logprobs[mb_inds]
                # Recomputed every minibatch, because the proximal policy moves
                # every minibatch -- caching it per iteration would silently turn
                # the COM into the rollout length.
                mb_proximal_logp = None if proximal is None else proximal.log_prob(b_obs[mb_inds], b_actions[mb_inds])

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                pg_loss, logratio = policy_surrogate(
                    mb_advantages, newlogprob, mb_behaviour_logp, mb_proximal_logp,
                    args.clip_coef, args.imp_samp_max,
                )

                with torch.no_grad():
                    # KL against the clip's own reference, which is what
                    # `--clip-coef` and `--proximal-com` jointly control.
                    ratio = logratio.exp()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]
                    if final_epoch:
                        # DDR is defined against the behaviour policy, which is a
                        # different reference once the EWMA is on.
                        behaviour_logratio = newlogprob - mb_behaviour_logp
                        kl_behaviour_sum += float(
                            ((behaviour_logratio.exp() - 1) - behaviour_logratio).mean()
                        )
                        kl_behaviour_count += 1

                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds], -args.clip_coef, args.clip_coef)
                    v_loss = 0.5 * torch.max(v_loss_unclipped, (v_clipped - b_returns[mb_inds]) ** 2).mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()
                if proximal is not None:
                    # After the step, matching the reference: the proximal policy
                    # is an average of post-update weights.
                    proximal.update(agent)

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        if writer is not None:
            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
            kl_behaviour = kl_behaviour_sum / kl_behaviour_count if kl_behaviour_count else float("nan")

            writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
            writer.add_scalar("charts/num_minibatches", args.num_minibatches, global_step)
            writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
            writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
            writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
            writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
            writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
            writer.add_scalar("losses/explained_variance", explained_var, global_step)
            # The plateau diagnostics of Figure 2: a large gradient norm and a
            # large behaviour KL alongside a flat return is the signature of an
            # outer step size that is too big, as opposed to a converged agent.
            writer.add_scalar("losses/grad_norm", float(grad_norm), global_step)
            writer.add_scalar("losses/kl_behaviour", kl_behaviour, global_step)
            ddr = data_to_divergence_ratio(args.batch_size, kl_behaviour)
            if math.isfinite(ddr):
                writer.add_scalar("charts/ddr", ddr, global_step)
            print("SPS:", int(global_step / (time.time() - start_time)))
            writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    if args.benchmark:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        benchmark_end = time.perf_counter()
        measured_steps = global_step - benchmark_start_step
        measured_seconds = benchmark_end - benchmark_start
        result = {
            "algorithm": "ppo_scale",
            "backend": args.env_backend,
            "batch_size": args.batch_size,
            "benchmark": "full_training_loop",
            "compile": False,
            "env_id": args.env_id,
            "measure_iterations": args.benchmark_measure_iterations,
            "measured_seconds": measured_seconds,
            "measured_steps": measured_steps,
            "minibatch_size": args.minibatch_size,
            "num_envs": args.num_envs,
            "num_minibatches": args.num_minibatches,
            "num_steps": args.num_steps,
            "peak_cuda_memory_mb": (
                torch.cuda.max_memory_allocated(device) / (1024**2) if device.type == "cuda" else 0.0
            ),
            "process_seconds": benchmark_end - process_start,
            "proximal_com": args.proximal_com,
            "schema_version": 1,
            "sps": measured_steps / measured_seconds,
            "update_epochs": args.update_epochs,
            "warmup_iterations": args.benchmark_warmup_iterations,
        }
        print(f"BENCHMARK_RESULT {json.dumps(result, sort_keys=True)}", flush=True)

    envs.close()
    if writer is not None:
        writer.close()
