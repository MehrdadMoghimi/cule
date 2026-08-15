# A2C (Advantage Actor-Critic) for Atari.
#
# The synchronous variant of A3C (Mnih et al., 2016, "Asynchronous Methods for
# Deep Reinforcement Learning", https://arxiv.org/abs/1602.01783). A3C's
# contribution is an *architecture* — many asynchronous workers each holding a
# copy of the network and applying Hogwild! updates to a shared parameter
# server. A2C is the observation, first published in OpenAI's baselines, that
# the asynchrony buys nothing once you can batch: run the same N workers
# synchronously, average their gradients, and you get the same algorithm with
# better GPU utilisation and reproducible seeding. This repo's vectorised
# rollout loop *is* that synchronous worker pool, so this file is A2C, and
# `--num-envs 16` reproduces the standard A3C-paper worker count.
#
# Ported from the reference implementation in openai/baselines
# (https://github.com/openai/baselines, MIT; `baselines/a2c/a2c.py` and
# `baselines/a2c/runner.py`). Structurally this is `ppo_atari.py` with the PPO
# machinery removed:
#
#   * GAE(lambda)                 -> plain n-step discounted returns
#   * clipped surrogate objective -> vanilla policy gradient, `A * -log pi(a|s)`
#   * K epochs over M minibatches -> exactly one gradient step on the full batch
#   * advantage normalisation     -> none
#   * clipped value loss          -> plain MSE
#   * Adam                        -> RMSProp, with TensorFlow's semantics
#
# That last one is not cosmetic. `torch.optim.RMSprop` is *not* the optimiser
# baselines trained with, in two separate ways, and at baselines' `eps=1e-5`
# both matter enormously:
#
#   1. TensorFlow initialises the mean-square accumulator to **ones**; PyTorch
#      initialises it to zeros. With zeros, the first update is
#      `lr * g / (sqrt(0.01 g^2) + eps) ~= 10 * lr * sign(g)` — a step ten times
#      the learning rate, in the sign direction only.
#   2. TensorFlow puts epsilon **inside** the square root, `g / sqrt(ms + eps)`;
#      PyTorch puts it outside, `g / (sqrt(ms) + eps)`.
#
# `RMSpropTFLike` below implements TensorFlow's version and is the default.
# Stable-Baselines3 ships the same optimiser for the same reason; the port is
# diffed against it in `tests/test_a2c_equivalence.py`.
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

    # Algorithm specific arguments
    env_id: str = "BreakoutNoFrameskip-v4"
    """the id of the environment"""
    total_timesteps: int = 10000000
    """total timesteps of the experiments"""
    learning_rate: float = 7e-4
    """the learning rate of the optimizer (baselines' A2C default)"""
    num_envs: int = 16
    """the number of parallel game environments (baselines' A2C default)"""
    num_steps: int = 5
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle linear learning rate annealing, as in baselines' `lrschedule='linear'`"""
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
    """RMSProp decay rate (baselines' `alpha`)"""
    rmsprop_eps: float = 1e-5
    """RMSProp epsilon (baselines' `epsilon`)"""

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


class RMSpropTFLike(Optimizer):
    """RMSProp with TensorFlow 1.x semantics, as `tf.train.RMSPropOptimizer`.

    Two deliberate differences from `torch.optim.RMSprop`, both of which change
    A2C's trajectory at baselines' `eps=1e-5`:

      * the mean-square accumulator starts at **one**, not zero, so the first
        update is `lr * g` rather than `~10 * lr * sign(g)`;
      * epsilon goes **inside** the square root, `g / sqrt(ms + eps)`, rather
        than outside, `g / (sqrt(ms) + eps)`.

    Equivalent to Stable-Baselines3's `RMSpropTFLike`, which exists for the same
    reason; `tests/test_a2c_equivalence.py` diffs the two update-for-update.
    """

    def __init__(self, params, lr=1e-2, alpha=0.99, eps=1e-10, weight_decay=0.0, momentum=0.0, centered=False):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if not 0.0 <= alpha:
            raise ValueError(f"Invalid alpha value: {alpha}")
        defaults = dict(
            lr=lr, momentum=momentum, alpha=alpha, eps=eps, centered=centered, weight_decay=weight_decay
        )
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("momentum", 0)
            group.setdefault("centered", False)

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
                if grad.is_sparse:
                    raise RuntimeError("RMSpropTFLike does not support sparse gradients")
                state = self.state[param]

                if len(state) == 0:
                    state["step"] = 0
                    # TensorFlow initialises the "rms" slot with ones.
                    state["square_avg"] = torch.ones_like(param, memory_format=torch.preserve_format)
                    if group["momentum"] > 0:
                        state["momentum_buffer"] = torch.zeros_like(param, memory_format=torch.preserve_format)
                    if group["centered"]:
                        state["grad_avg"] = torch.zeros_like(param, memory_format=torch.preserve_format)

                square_avg = state["square_avg"]
                alpha = group["alpha"]
                state["step"] += 1

                if group["weight_decay"] != 0:
                    grad = grad.add(param, alpha=group["weight_decay"])

                square_avg.mul_(alpha).addcmul_(grad, grad, value=1 - alpha)

                if group["centered"]:
                    grad_avg = state["grad_avg"]
                    grad_avg.mul_(alpha).add_(grad, alpha=1 - alpha)
                    # epsilon inside the square root
                    avg = square_avg.addcmul(grad_avg, grad_avg, value=-1).add_(group["eps"]).sqrt_()
                else:
                    # epsilon inside the square root
                    avg = square_avg.add(group["eps"]).sqrt_()

                if group["momentum"] > 0:
                    buf = state["momentum_buffer"]
                    buf.mul_(group["momentum"]).addcdiv_(grad, avg)
                    param.add_(buf, alpha=-group["lr"])
                else:
                    param.addcdiv_(grad, avg, value=-group["lr"])

        return loss


def discount_with_dones(rewards, dones, gamma):
    """Reference transcription of `baselines/a2c/utils.py::discount_with_dones`.

    Kept as an explicit, list-based function so the vectorised
    `nstep_returns` below has something to be diffed against.
    """
    discounted = []
    running = 0.0
    for reward, done in zip(rewards[::-1], dones[::-1]):
        running = reward + gamma * running * (1.0 - done)
        discounted.append(running)
    return discounted[::-1]


def nstep_returns(rewards, next_dones, last_values, gamma):
    """Bootstrapped n-step discounted returns, vectorised over environments.

    `next_dones[t]` is the done flag *produced by* step `t` — baselines'
    `mb_dones[:, 1:]`, not the `mb_masks` used for recurrent state resets.

    baselines branches on whether the final step ended an episode, bootstrapping
    only when it did not. The branch is redundant: when `next_dones[-1] == 1`
    the `(1 - done)` factor already zeroes the bootstrap, so both branches agree.
    This function always bootstraps; `tests/test_a2c_equivalence.py` pins the
    equivalence against `discount_with_dones` on both cases.
    """
    num_steps = rewards.shape[0]
    returns = torch.zeros_like(rewards)
    running = last_values
    for t in reversed(range(num_steps)):
        running = rewards[t] + gamma * running * (1.0 - next_dones[t])
        returns[t] = running
    return returns


def baselines_lr_fraction(iteration, batch_size, total_timesteps):
    """baselines' linear schedule, off-by-one included.

    `Scheduler.value()` is called once per element of the batch inside
    `Model.train`, and the *last* of those calls supplies the learning rate used
    for the update. So at 1-indexed iteration `u` the counter has reached
    `u * batch_size - 1`, not `u * batch_size`. Preserved exactly rather than
    rounded off, because it is the kind of detail that silently splits a port
    from its reference.
    """
    consumed = iteration * batch_size - 1
    return max(0.0, 1.0 - consumed / total_timesteps)


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    """The Nature-CNN torso with separate policy and value heads.

    Identical to `ppo_atari.py`'s agent, which is itself identical to the
    `cnn` policy baselines' A2C builds: orthogonal init at `sqrt(2)` for the
    torso, `0.01` for the policy head and `1.0` for the value head.
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


def a2c_losses(logprobs, entropies, values, returns, behaviour_values, ent_coef, vf_coef):
    """baselines' A2C objective.

    `pg_loss = mean(ADV * neglogp)` with `ADV = returns - behaviour_values`
    supplied as data, so no gradient flows through the advantage. The value loss
    is TensorFlow's `losses.mean_squared_error`, i.e. a plain mean of squares
    with no leading 1/2 — the customary 1/2 lives in `vf_coef = 0.5`.
    """
    advantages = returns - behaviour_values
    pg_loss = (advantages * -logprobs).mean()
    value_loss = ((values - returns) ** 2).mean()
    entropy = entropies.mean()
    loss = pg_loss - entropy * ent_coef + value_loss * vf_coef
    return loss, pg_loss, value_loss, entropy


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

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    # `next_dones[t]` is the flag produced *by* step t (baselines' mb_dones[:, 1:]).
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

    for iteration in range(1, args.num_iterations + 1):
        if args.benchmark and iteration == args.benchmark_warmup_iterations + 1:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            benchmark_start = time.perf_counter()
            benchmark_start_step = global_step
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = baselines_lr_fraction(iteration, args.batch_size, args.total_timesteps)
            optimizer.param_groups[0]["lr"] = frac * args.learning_rate

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, _, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = step_env(envs, action)
            next_done = done_tensor(terminations, truncations, device)
            rewards[step] = to_tensor(reward, device).view(-1)
            next_dones[step] = next_done
            next_obs = to_tensor(next_obs, device)

            stats.update(completed_episode_infos(infos, next_done), global_step, writer)

        # n-step returns, bootstrapped off the value of the state we stopped in
        with torch.no_grad():
            last_value = agent.get_value(next_obs).reshape(-1)
            returns = nstep_returns(rewards, next_dones, last_value, args.gamma)

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Exactly one gradient step on the whole batch: no epochs, no minibatches.
        _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs, b_actions.long())
        loss, pg_loss, v_loss, entropy_loss = a2c_losses(
            newlogprob,
            entropy,
            newvalue.view(-1),
            b_returns,
            b_values,
            args.ent_coef,
            args.vf_coef,
        )

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
        optimizer.step()

        if writer is not None and iteration % 100 == 0:
            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

            # TRY NOT TO MODIFY: record rewards for plotting purposes
            writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
            writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
            writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
            writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
            writer.add_scalar("losses/explained_variance", explained_var, global_step)
            print("SPS:", int(global_step / (time.time() - start_time)))
            writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    if args.benchmark:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        benchmark_end = time.perf_counter()
        measured_steps = global_step - benchmark_start_step
        measured_seconds = benchmark_end - benchmark_start
        result = {
            "algorithm": "a2c",
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
            "optimizer": args.optimizer,
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
