# A2C + SIL (Self-Imitation Learning) for Atari.
#
# Oh, Guo, Lee, Lewis and Singh, ICML 2018, "Self-Imitation Learning"
# (https://arxiv.org/abs/1806.05635). Ported from the authors' implementation,
# junhyukoh/self-imitation-learning (MIT), `baselines/common/self_imitation.py`
# and `baselines/a2c/a2c_sil.py`.
#
# The idea is one line long: **when the agent did better than it expected, do
# that again.** Store past transitions with their Monte-Carlo return `R`, and
# train off-policy on the ones where `R > V(s)` — the agent's own good luck,
# treated as demonstrations. The clipped advantage `(R - V)_+` is both the
# policy-gradient weight and the replay priority, so the buffer preferentially
# resurfaces exactly the experiences the critic still underrates.
#
# Sitting on top of `a2c_atari.py`: everything about the on-policy update is
# inherited unchanged, and each A2C step is followed by `--sil-update` (4)
# gradient steps on batches of 512 drawn from a prioritized buffer.
#
# The reference has five details that are individually easy to get wrong and
# jointly are most of why it works. All five are pinned by
# `tests/test_sil_equivalence.py`:
#
#   1. **Only episodes containing a positive reward are stored**
#      (`update_buffer`). On Atari with sign-clipped rewards that means the
#      buffer holds nothing until the agent scores at all — SIL amplifies
#      success, it does not manufacture it.
#   2. **The normaliser is `max(#{R > V}, min_batch_size)`, not the batch size.**
#      When only three samples in a 512-batch beat their value estimate, the
#      loss is divided by 64, not by 3 and not by 512. Dividing by the count of
#      valid samples would make a batch with one lucky transition as loud as a
#      batch with four hundred.
#   3. **The value "loss" is written as `sum(W * V * stop_grad(delta))`.** Its
#      *value* is meaningless; only its gradient is intended, and that gradient
#      is the gradient of `0.5 (V - R)^2` restricted to `R > V` and clipped to
#      magnitude `clip`. Transcribing it as an actual squared error changes the
#      update as soon as `|V - R| > 1`.
#   4. **`nlogp` is value-clipped but gradient-transparent**:
#      `stop_grad(min(nlogp, 5) - nlogp) + nlogp` reports at most 5 while
#      passing the *unclipped* gradient through. It bounds the reported loss
#      without bounding the learning signal.
#   5. **Priorities are the clipped advantage itself**, floored at 1e-6 so a
#      transition the critic has caught up with is never sampled to zero
#      probability but is effectively retired.
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
from torch.optim.optimizer import Optimizer
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

    # Algorithm specific arguments (A2C; see a2c_atari.py)
    env_id: str = "BreakoutNoFrameskip-v4"
    """the id of the environment"""
    total_timesteps: int = 10000000
    """total timesteps of the experiments"""
    learning_rate: float = 7e-4
    """the learning rate of the optimizer"""
    num_envs: int = 16
    """the number of parallel game environments"""
    num_steps: int = 5
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = False
    """the SIL paper's Atari runs use `lrschedule='constant'`"""
    gamma: float = 0.99
    """the discount factor gamma"""
    ent_coef: float = 0.01
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    optimizer: str = "rmsprop-tf"
    """`rmsprop-tf` (baselines' TensorFlow RMSProp), `rmsprop` (torch), or `adam`"""
    rmsprop_alpha: float = 0.99
    """RMSProp decay rate"""
    rmsprop_eps: float = 1e-5
    """RMSProp epsilon"""

    # SIL arguments
    sil_update: int = 4
    """SIL gradient steps per A2C update (`--sil-update`)"""
    sil_batch_size: int = 512
    """transitions per SIL gradient step"""
    sil_beta: float = 0.1
    """importance-sampling exponent for the prioritized replay (`--sil-beta`)"""
    sil_alpha: float = 0.6
    """prioritization exponent"""
    sil_capacity: int = 100000
    """maximum transitions held in the SIL buffer"""
    sil_w_value: float = 0.01
    """weight on the SIL value term"""
    sil_w_entropy: float = 0.01
    """weight on the SIL entropy term"""
    sil_clip: float = 1.0
    """upper bound on the clipped advantage `(R - V)_+`"""
    sil_max_nlogp: float = 5.0
    """value-clip on `-log pi(a|s)`; the gradient is left unclipped"""
    sil_min_batch_size: int = 64
    """floor on the loss normaliser"""
    sil_min_buffer: int = 100
    """transitions required before SIL starts training"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""

    solve_window: int = 100
    """episodes averaged when reporting the running return"""

    benchmark: bool = False
    """run a fixed warmup/measurement window and print a JSON benchmark result"""
    benchmark_warmup_iterations: int = 20
    """full training iterations excluded from benchmark timing"""
    benchmark_measure_iterations: int = 100
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
# A2C pieces, shared with a2c_atari.py
# ---------------------------------------------------------------------------

class RMSpropTFLike(Optimizer):
    """RMSProp with TensorFlow 1.x semantics; see `a2c_atari.py` for why."""

    def __init__(self, params, lr=1e-2, alpha=0.99, eps=1e-10, weight_decay=0.0, momentum=0.0, centered=False):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        defaults = dict(lr=lr, momentum=momentum, alpha=alpha, eps=eps,
                        centered=centered, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for param in group["params"]:
                if param.grad is None:
                    continue
                grad = param.grad
                state = self.state[param]
                if len(state) == 0:
                    state["step"] = 0
                    state["square_avg"] = torch.ones_like(param, memory_format=torch.preserve_format)
                    if group["momentum"] > 0:
                        state["momentum_buffer"] = torch.zeros_like(param, memory_format=torch.preserve_format)
                square_avg = state["square_avg"]
                state["step"] += 1
                if group["weight_decay"] != 0:
                    grad = grad.add(param, alpha=group["weight_decay"])
                square_avg.mul_(group["alpha"]).addcmul_(grad, grad, value=1 - group["alpha"])
                avg = square_avg.add(group["eps"]).sqrt_()  # epsilon inside the sqrt
                if group["momentum"] > 0:
                    buf = state["momentum_buffer"]
                    buf.mul_(group["momentum"]).addcdiv_(grad, avg)
                    param.add_(buf, alpha=-group["lr"])
                else:
                    param.addcdiv_(grad, avg, value=-group["lr"])
        return loss


def discount_with_dones(rewards, dones, gamma):
    """`baselines/a2c/utils.py::discount_with_dones`."""
    discounted = []
    running = 0.0
    for reward, done in zip(rewards[::-1], dones[::-1]):
        running = reward + gamma * running * (1.0 - done)
        discounted.append(running)
    return discounted[::-1]


def nstep_returns(rewards, next_dones, last_values, gamma):
    """Bootstrapped n-step returns; see `a2c_atari.py`."""
    num_steps = rewards.shape[0]
    returns = torch.zeros_like(rewards)
    running = last_values
    for t in reversed(range(num_steps)):
        running = rewards[t] + gamma * running * (1.0 - next_dones[t])
        returns[t] = running
    return returns


def a2c_losses(logprobs, entropies, values, returns, behaviour_values, ent_coef, vf_coef):
    """baselines' A2C objective; see `a2c_atari.py`."""
    advantages = returns - behaviour_values
    pg_loss = (advantages * -logprobs).mean()
    value_loss = ((values - returns) ** 2).mean()
    entropy = entropies.mean()
    loss = pg_loss - entropy * ent_coef + value_loss * vf_coef
    return loss, pg_loss, value_loss, entropy


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    """The Nature CNN with policy and value heads, as in `a2c_atari.py`."""

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


# ---------------------------------------------------------------------------
# prioritized replay
# ---------------------------------------------------------------------------

class SumMinSegmentTree:
    """The sum/min segment trees baselines' `PrioritizedReplayBuffer` uses.

    Kept as one class over two arrays because SIL always updates both together.
    Capacity is rounded up to a power of two, exactly as upstream does.
    """

    def __init__(self, capacity):
        size = 1
        while size < capacity:
            size *= 2
        self.size = size
        self.sum_tree = np.zeros(2 * size, dtype=np.float64)
        self.min_tree = np.full(2 * size, np.inf, dtype=np.float64)

    def __setitem__(self, index, value):
        index += self.size
        self.sum_tree[index] = value
        self.min_tree[index] = value
        index //= 2
        while index >= 1:
            self.sum_tree[index] = self.sum_tree[2 * index] + self.sum_tree[2 * index + 1]
            self.min_tree[index] = min(self.min_tree[2 * index], self.min_tree[2 * index + 1])
            index //= 2

    def __getitem__(self, index):
        return self.sum_tree[index + self.size]

    def sum(self):
        return self.sum_tree[1]

    def min(self):
        return self.min_tree[1]

    def find_prefixsum_idx(self, prefixsum):
        """Smallest `i` with `sum(tree[:i+1]) > prefixsum`, upstream's semantics."""
        index = 1
        while index < self.size:
            if self.sum_tree[2 * index] > prefixsum:
                index = 2 * index
            else:
                prefixsum -= self.sum_tree[2 * index]
                index = 2 * index + 1
        return index - self.size


class PrioritizedReplayBuffer:
    """`self_imitation.py::PrioritizedReplayBuffer`, storing `(obs, action, R)`.

    Note what is *not* here: no next observation, no done, no reward. SIL only
    ever needs the Monte-Carlo return, which is computed once when the episode
    ends and never bootstrapped again.
    """

    def __init__(self, capacity, alpha, obs_shape, obs_dtype=np.uint8):
        self.capacity = capacity
        self.alpha = alpha
        self.observations = np.zeros((capacity,) + tuple(obs_shape), dtype=obs_dtype)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.returns = np.zeros(capacity, dtype=np.float32)
        self.tree = SumMinSegmentTree(capacity)
        self.next_index = 0
        self.size = 0
        self.max_priority = 1.0

    def __len__(self):
        return self.size

    def add(self, observation, action, episode_return):
        index = self.next_index
        self.observations[index] = observation
        self.actions[index] = action
        self.returns[index] = episode_return
        self.tree[index] = self.max_priority**self.alpha
        self.next_index = (index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, beta, generator=None):
        """Proportional sampling with importance weights normalised by the max.

        Upstream draws `mass = rand() * sum(0, len - 1)` — note the exclusive
        upper index, a quirk that makes the newest element unreachable in the
        very rare case where it holds the whole mass. Reproduced.
        """
        if self.size == 0:
            return None
        random_source = generator if generator is not None else np.random
        indices = np.empty(batch_size, dtype=np.int64)
        total = self.tree.sum()
        for i in range(batch_size):
            mass = random_source.random() * total
            indices[i] = min(self.tree.find_prefixsum_idx(mass), self.size - 1)

        p_min = self.tree.min() / total
        max_weight = (p_min * self.size) ** (-beta)
        probabilities = np.array([self.tree[int(i)] for i in indices]) / total
        weights = ((probabilities * self.size) ** (-beta)) / max_weight
        return (
            self.observations[indices],
            self.actions[indices],
            self.returns[indices],
            weights.astype(np.float32),
            indices,
        )

    def update_priorities(self, indices, priorities):
        for index, priority in zip(indices, priorities):
            priority = max(float(priority), 1e-6)  # upstream's floor
            self.tree[int(index)] = priority**self.alpha
            self.max_priority = max(self.max_priority, priority)


class SelfImitation:
    """Episode bookkeeping plus the SIL objective.

    `step` accumulates per-environment trajectories; on `done` the episode is
    scored and, **only if it contained a positive reward**, discounted and
    written to the buffer.
    """

    def __init__(self, num_envs, obs_shape, capacity=100000, alpha=0.6, gamma=0.99,
                 obs_dtype=np.uint8):
        self.num_envs = num_envs
        self.gamma = gamma
        self.buffer = PrioritizedReplayBuffer(capacity, alpha, obs_shape, obs_dtype)
        self.running_episodes = [[] for _ in range(num_envs)]
        self.episodes_stored = 0

    def step(self, observations, actions, rewards, dones):
        for env in range(self.num_envs):
            self.running_episodes[env].append(
                (observations[env], int(actions[env]), float(rewards[env])))
        for env, done in enumerate(dones):
            if done:
                self.update_buffer(self.running_episodes[env])
                self.running_episodes[env] = []

    def update_buffer(self, trajectory):
        if not trajectory:
            return False
        # "Only episodes with a positive reward": the gate that makes SIL
        # amplify success rather than reinforce whatever happened first.
        if not any(reward > 0 for _, _, reward in trajectory):
            return False
        rewards = [reward for _, _, reward in trajectory]
        dones = [False] * len(trajectory)
        dones[-1] = True
        returns = discount_with_dones(rewards, dones, self.gamma)
        for (observation, action, _), episode_return in zip(trajectory, returns):
            self.buffer.add(observation, action, episode_return)
        self.episodes_stored += 1
        return True


def clipped_neg_logp(neg_logp, max_nlogp):
    """`stop_gradient(min(nlogp, max) - nlogp) + nlogp`.

    Value-clipped, gradient-transparent: the forward value is
    `min(nlogp, max_nlogp)` but the backward pass sees `d nlogp`, unmodified.
    Writing `clamp(nlogp, max=max_nlogp)` instead would zero the gradient
    wherever the clip bites — which is precisely on the rare, high-surprise
    actions SIL most wants to learn from.
    """
    return (torch.minimum(neg_logp, torch.as_tensor(max_nlogp, dtype=neg_logp.dtype,
                                                    device=neg_logp.device)) - neg_logp).detach() + neg_logp


def sil_losses(logprobs, entropies, values, returns, weights,
               clip=1.0, max_nlogp=5.0, min_batch_size=64,
               w_value=0.01, w_entropy=0.01):
    """`SelfImitation.build_loss_op`.

    Returns `(loss, clipped_advantage, num_valid)`. The clipped advantage is
    also the new replay priority.

    The value term is upstream's surrogate `sum(W * V * stop_grad(delta))` with
    `delta = clip(V - R, -clip, 0) * mask`. Its numerical value is not a
    squared error and should not be read as one; its *gradient* is
    `W * delta * dV/dtheta`, i.e. the gradient of `0.5 (V - R)^2` restricted to
    `R > V` and clipped at `clip`.
    """
    neg_logp = -logprobs
    mask = (returns - values > 0).to(values.dtype)
    num_valid = mask.sum()
    num_samples = torch.clamp(num_valid, min=float(min_batch_size))

    clipped = clipped_neg_logp(neg_logp, max_nlogp)
    advantage = torch.clamp(returns - values, min=0.0, max=clip).detach()
    pg_loss = (weights * advantage * clipped).sum() / num_samples
    entropy_term = (weights * entropies * mask).sum() / num_samples
    loss = pg_loss - entropy_term * w_entropy

    delta = torch.clamp(values - returns, min=-clip, max=0.0) * mask
    value_surrogate = (weights * values * delta.detach()).sum() / num_samples
    loss = loss + 0.5 * w_value * value_surrogate
    return loss, advantage, num_valid


def build_optimizer(name, parameters, args):
    if name == "rmsprop-tf":
        return RMSpropTFLike(parameters, lr=args.learning_rate, alpha=args.rmsprop_alpha, eps=args.rmsprop_eps)
    if name == "rmsprop":
        return optim.RMSprop(parameters, lr=args.learning_rate, alpha=args.rmsprop_alpha, eps=args.rmsprop_eps)
    if name == "adam":
        return optim.Adam(parameters, lr=args.learning_rate, eps=1e-5)
    raise ValueError(f"unsupported optimizer: {name}")


if __name__ == "__main__":
    process_start = time.perf_counter()
    args = tyro.cli(Args)
    if args.num_envs < 1:
        raise ValueError("num_envs must be positive")
    if args.num_steps < 1:
        raise ValueError("num_steps must be positive")
    if args.sil_update < 0:
        raise ValueError("sil_update cannot be negative")
    if args.benchmark_warmup_iterations < 0:
        raise ValueError("benchmark_warmup_iterations cannot be negative")
    if args.benchmark_measure_iterations < 1:
        raise ValueError("benchmark_measure_iterations must be positive")
    args.batch_size = int(args.num_envs * args.num_steps)
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
    optimizer = build_optimizer(args.optimizer, agent.parameters(), args)
    self_imitation = SelfImitation(
        args.num_envs,
        envs.single_observation_space.shape,
        capacity=args.sil_capacity,
        alpha=args.sil_alpha,
        gamma=args.gamma,
    )

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs), dtype=torch.long).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    next_dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = to_tensor(next_obs, device)
    next_done = torch.zeros(args.num_envs).to(device)
    stats = EpisodeStats(args.solve_window)
    benchmark_start = None
    benchmark_start_step = None
    sil_valid_samples = 0.0

    for iteration in range(1, args.num_iterations + 1):
        if args.benchmark and iteration == args.benchmark_warmup_iterations + 1:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            benchmark_start = time.perf_counter()
            benchmark_start_step = global_step
        if args.anneal_lr:
            frac = max(0.0, 1.0 - (iteration * args.batch_size - 1) / args.total_timesteps)
            optimizer.param_groups[0]["lr"] = frac * args.learning_rate

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs

            with torch.no_grad():
                action, _, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action

            step_obs = to_numpy(next_obs).astype(np.uint8)
            next_obs, reward, terminations, truncations, infos = step_env(envs, action)
            next_done = done_tensor(terminations, truncations, device)
            rewards[step] = to_tensor(reward, device).view(-1)
            next_dones[step] = next_done
            next_obs = to_tensor(next_obs, device)

            # The SIL buffer sees the state the action was taken *in*, the
            # action, and the (already sign-clipped) reward it produced.
            self_imitation.step(
                step_obs,
                to_numpy(action),
                to_numpy(rewards[step]),
                to_numpy(next_done).astype(bool),
            )

            stats.update(completed_episode_infos(infos, next_done), global_step, writer)

        # --- the ordinary A2C update ---------------------------------------
        with torch.no_grad():
            last_value = agent.get_value(next_obs).reshape(-1)
            returns = nstep_returns(rewards, next_dones, last_value, args.gamma)

        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_actions = actions.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs, b_actions)
        loss, pg_loss, v_loss, entropy_loss = a2c_losses(
            newlogprob, entropy, newvalue.view(-1), b_returns, b_values, args.ent_coef, args.vf_coef)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
        optimizer.step()

        # --- the SIL updates ------------------------------------------------
        sil_loss_value = 0.0
        if args.sil_update > 0 and len(self_imitation.buffer) > args.sil_min_buffer:
            for _ in range(args.sil_update):
                batch_size = min(args.sil_batch_size, len(self_imitation.buffer))
                sample = self_imitation.buffer.sample(batch_size, args.sil_beta)
                if sample is None:
                    break
                sample_obs, sample_actions, sample_returns, sample_weights, sample_indices = sample

                sil_obs = torch.as_tensor(sample_obs, device=device, dtype=torch.float32)
                sil_actions = torch.as_tensor(sample_actions, device=device)
                sil_returns = torch.as_tensor(sample_returns, device=device, dtype=torch.float32)
                sil_weights = torch.as_tensor(sample_weights, device=device, dtype=torch.float32)

                _, sil_logprob, sil_entropy, sil_value = agent.get_action_and_value(sil_obs, sil_actions)
                sil_loss, advantage, num_valid = sil_losses(
                    sil_logprob,
                    sil_entropy,
                    sil_value.view(-1),
                    sil_returns,
                    sil_weights,
                    clip=args.sil_clip,
                    max_nlogp=args.sil_max_nlogp,
                    min_batch_size=args.sil_min_batch_size,
                    w_value=args.sil_w_value,
                    w_entropy=args.sil_w_entropy,
                )

                optimizer.zero_grad()
                sil_loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

                self_imitation.buffer.update_priorities(sample_indices, to_numpy(advantage))
                sil_loss_value = sil_loss.item()
                sil_valid_samples = float(num_valid.item())

        if writer is not None and iteration % 100 == 0:
            writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
            writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
            writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
            writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
            writer.add_scalar("sil/loss", sil_loss_value, global_step)
            writer.add_scalar("sil/buffer_size", len(self_imitation.buffer), global_step)
            writer.add_scalar("sil/episodes_stored", self_imitation.episodes_stored, global_step)
            writer.add_scalar("sil/valid_samples", sil_valid_samples, global_step)
            print("SPS:", int(global_step / (time.time() - start_time)))
            writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    if args.benchmark:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        benchmark_end = time.perf_counter()
        measured_steps = global_step - benchmark_start_step
        measured_seconds = benchmark_end - benchmark_start
        result = {
            "algorithm": "a2c_sil",
            "backend": args.env_backend,
            "batch_size": args.batch_size,
            "benchmark": "full_training_loop",
            "compile": False,
            "env_id": args.env_id,
            "measure_iterations": args.benchmark_measure_iterations,
            "measured_seconds": measured_seconds,
            "measured_steps": measured_steps,
            "num_envs": args.num_envs,
            "num_steps": args.num_steps,
            "peak_cuda_memory_mb": (
                torch.cuda.max_memory_allocated(device) / (1024**2) if device.type == "cuda" else 0.0
            ),
            "process_seconds": benchmark_end - process_start,
            "schema_version": 1,
            "sil_batch_size": args.sil_batch_size,
            "sil_update": args.sil_update,
            "sps": measured_steps / measured_seconds,
            "warmup_iterations": args.benchmark_warmup_iterations,
        }
        print(f"BENCHMARK_RESULT {json.dumps(result, sort_keys=True)}", flush=True)

    envs.close()
    if writer is not None:
        writer.close()
