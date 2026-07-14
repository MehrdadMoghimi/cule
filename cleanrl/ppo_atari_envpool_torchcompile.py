# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_atari_envpoolpy
import os
import random
import sys
import time
from collections import deque
from dataclasses import dataclass

os.environ["TORCHDYNAMO_INLINE_INBUILT_NN_MODULES"] = "1"

try:
    import envpool
except ImportError:
    envpool = None
import gymnasium as gym
import numpy as np
import tensordict
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
import tyro
import wandb
from tensordict import from_module
from tensordict.nn import CudaGraphModule
from torch.distributions.categorical import Categorical, Distribution

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from cule_env import (
    done_tensor,
    make_cule_env,
    resolve_cule_device,
    step_env,
    to_numpy,
    to_tensor,
)

Distribution.set_default_validate_args(False)

torch.set_float32_matmul_precision("high")


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
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    env_backend: str = "cule"
    """environment backend: `cule` or `envpool`"""
    cule_device: str = "auto"
    """CuLE device; auto uses CUDA for 32+ envs and CPU for smaller batches"""
    track: bool = False
    """if toggled, track the experiment with Weights and Biases"""

    # Algorithm specific arguments
    env_id: str = "Breakout-v5"
    """the id of the environment"""
    total_timesteps: int = 10000000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 256
    """the number of parallel game environments"""
    num_steps: int = 32
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
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
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.01
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float | None = None
    """the target KL divergence threshold"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""

    measure_burnin: int = 3
    """Number of burn-in iterations for speed measure."""

    compile: bool = False
    """whether to use torch.compile."""
    cudagraphs: bool = False
    """whether to use cudagraphs on top of compile."""


class RecordEpisodeStatistics(gym.Wrapper):
    def __init__(self, env, deque_size=100):
        super().__init__(env)
        self.num_envs = getattr(env, "num_envs", 1)
        self.single_action_space = getattr(env, "single_action_space", env.action_space)
        self.single_observation_space = getattr(env, "single_observation_space", env.observation_space)
        self.episode_returns = None
        self.episode_lengths = None

    def reset(self, **kwargs):
        observations = super().reset(**kwargs)
        self.episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        self.lives = np.zeros(self.num_envs, dtype=np.int32)
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
        self.episode_returns *= 1 - infos["terminated"]
        self.episode_lengths *= 1 - infos["terminated"]
        infos["r"] = self.returned_episode_returns
        infos["l"] = self.returned_episode_lengths
        if len(result) == 5:
            return observations, rewards, terminations, truncations, infos
        return observations, rewards, dones, infos


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs, device=None):
        super().__init__()
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(4, 32, 8, stride=4, device=device)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2, device=device)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1, device=device)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 7 * 7, 512, device=device)),
            nn.ReLU(),
        )
        self.actor = layer_init(nn.Linear(512, envs.single_action_space.n, device=device), std=0.01)
        self.critic = layer_init(nn.Linear(512, 1, device=device), std=1)

    def get_value(self, x):
        return self.critic(self.network(x / 255.0))

    def get_action_and_value(self, obs, action=None):
        hidden = self.network(obs / 255.0)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)


def gae(next_obs, next_done, container):
    # bootstrap value if not done
    next_value = get_value(next_obs).reshape(-1)
    lastgaelam = 0
    nextnonterminals = (~container["dones"]).float().unbind(0)
    vals = container["vals"]
    vals_unbind = vals.unbind(0)
    rewards = container["rewards"].unbind(0)

    advantages = []
    nextnonterminal = (~next_done).float()
    nextvalues = next_value
    for t in range(args.num_steps - 1, -1, -1):
        cur_val = vals_unbind[t]
        delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - cur_val
        advantages.append(delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam)
        lastgaelam = advantages[-1]

        nextnonterminal = nextnonterminals[t]
        nextvalues = cur_val

    advantages = container["advantages"] = torch.stack(list(reversed(advantages)))
    container["returns"] = advantages + vals
    return container


def rollout(obs, done, avg_returns):
    ts = []
    for step in range(args.num_steps):
        torch.compiler.cudagraph_mark_step_begin()
        with torch.no_grad():
            action, logprob, _, value = policy(obs=obs)
            # reduce-overhead compilation may return CUDA-graph-owned buffers
            # that are overwritten by the next policy call.
            action = action.clone()
            logprob = logprob.clone()
            value = value.clone()
        # CuLE mutates and reuses its observation tensor during step().
        rollout_obs = obs.clone()
        step_result = step_env(envs, action)
        if len(step_result) == 5:
            next_obs_raw, reward, terminations, truncations, info = step_result
            next_done = done_tensor(terminations, truncations, device).bool()
        else:
            next_obs_raw, reward, next_done_raw, info = step_result
            next_done = to_tensor(next_done_raw, device, torch.bool)

        if "final_info" in info:
            for final_info in info["final_info"]:
                if final_info and "episode" in final_info:
                    avg_returns.append(float(final_info["episode"]["r"]))
        elif "r" in info:
            game_overs = to_numpy(next_done).astype(bool) & (np.asarray(info["lives"]) == 0)
            avg_returns.extend(np.asarray(info["r"])[game_overs].tolist())

        next_obs = to_tensor(next_obs_raw, device)
        reward = to_tensor(reward, device, torch.float32)

        ts.append(
            tensordict.TensorDict._new_unsafe(
                obs=rollout_obs,
                # cleanrl ppo examples associate the done with the previous obs (not the done resulting from action)
                dones=done,
                vals=value.flatten(),
                actions=action,
                logprobs=logprob,
                rewards=reward,
                batch_size=(args.num_envs,),
            )
        )

        obs = next_obs
        done = next_done

    container = torch.stack(ts, 0).to(device)
    return next_obs, done, container


def update(obs, actions, logprobs, advantages, returns, vals):
    optimizer.zero_grad()
    _, newlogprob, entropy, newvalue = agent.get_action_and_value(obs, actions)
    logratio = newlogprob - logprobs
    ratio = logratio.exp()

    with torch.no_grad():
        # calculate approx_kl http://joschu.net/blog/kl-approx.html
        old_approx_kl = (-logratio).mean()
        approx_kl = ((ratio - 1) - logratio).mean()
        clipfrac = ((ratio - 1.0).abs() > args.clip_coef).float().mean()

    if args.norm_adv:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # Policy loss
    pg_loss1 = -advantages * ratio
    pg_loss2 = -advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

    # Value loss
    newvalue = newvalue.view(-1)
    if args.clip_vloss:
        v_loss_unclipped = (newvalue - returns) ** 2
        v_clipped = vals + torch.clamp(
            newvalue - vals,
            -args.clip_coef,
            args.clip_coef,
        )
        v_loss_clipped = (v_clipped - returns) ** 2
        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
        v_loss = 0.5 * v_loss_max.mean()
    else:
        v_loss = 0.5 * ((newvalue - returns) ** 2).mean()

    entropy_loss = entropy.mean()
    loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

    loss.backward()
    gn = nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
    optimizer.step()

    return approx_kl, v_loss.detach(), pg_loss.detach(), entropy_loss.detach(), old_approx_kl, clipfrac, gn


update = tensordict.nn.TensorDictModule(
    update,
    in_keys=["obs", "actions", "logprobs", "advantages", "returns", "vals"],
    out_keys=["approx_kl", "v_loss", "pg_loss", "entropy_loss", "old_approx_kl", "clipfrac", "gn"],
)

if __name__ == "__main__":
    args = tyro.cli(Args)

    if args.num_envs < 1:
        raise ValueError("num_envs must be positive")
    if args.num_steps < 1:
        raise ValueError("num_steps must be positive")
    if args.num_minibatches < 1:
        raise ValueError("num_minibatches must be positive")

    batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = batch_size // args.num_minibatches
    if args.minibatch_size < 1:
        raise ValueError("num_minibatches cannot exceed num_envs * num_steps")
    args.batch_size = args.num_minibatches * args.minibatch_size
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{args.compile}__{args.cudagraphs}"

    if args.track:
        wandb.init(
            project="ppo_atari",
            name=f"{os.path.splitext(os.path.basename(__file__))[0]}-{run_name}",
            config=vars(args),
            save_code=True,
        )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    if args.cudagraphs and device.type != "cuda":
        raise ValueError("cudagraphs requires CUDA")

    ####### Environment setup #######
    if args.env_backend == "cule":
        env_device = resolve_cule_device(args.cule_device, device, args.num_envs)
        envs = make_cule_env(args.env_id, args.num_envs, env_device, args.seed, args.capture_video)
    elif args.env_backend == "envpool":
        if envpool is None:
            raise ImportError("EnvPool backend requested; install envpool or pass --env-backend cule")
        envs = envpool.make(
            args.env_id,
            env_type="gym",
            num_envs=args.num_envs,
            episodic_life=True,
            reward_clip=True,
            seed=args.seed,
        )
        envs = RecordEpisodeStatistics(envs)
    else:
        raise ValueError(f"unsupported environment backend: {args.env_backend}")
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    # def step_func(action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    #     next_obs_np, reward, next_done, info = envs.step(action.cpu().numpy())
    #     return torch.as_tensor(next_obs_np), torch.as_tensor(reward), torch.as_tensor(next_done), info

    ####### Agent #######
    agent = Agent(envs, device=device)
    # Make a version of agent with detached params
    agent_inference = Agent(envs, device=device)
    agent_inference_p = from_module(agent).detach()
    agent_inference_p.to_module(agent_inference, preserve_module_state=True)

    ####### Optimizer #######
    optimizer = optim.Adam(
        agent.parameters(),
        lr=torch.tensor(args.learning_rate, device=device),
        eps=1e-5,
        capturable=args.cudagraphs and not args.compile,
    )

    ####### Executables #######
    # Define networks: wrapping the policy in a TensorDictModule allows us to use CudaGraphModule
    policy = agent_inference.get_action_and_value
    get_value = agent_inference.get_value

    # Compile policy
    if args.compile:
        mode = "reduce-overhead" if not args.cudagraphs else None
        policy = torch.compile(policy, mode=mode)
        # GAE results live across several compiled optimizer calls; implicit
        # reduce-overhead CUDA graphs would reuse and overwrite those outputs.
        gae = torch.compile(gae, fullgraph=True, mode=None)
        update = torch.compile(update, mode=mode)

    if args.cudagraphs:
        policy = CudaGraphModule(policy, warmup=20)
        #gae = CudaGraphModule(gae, warmup=20)
        update = CudaGraphModule(update, warmup=20)

    avg_returns = deque(maxlen=20)
    global_step = 0
    container_local = None
    reset_result = envs.reset(seed=args.seed) if args.env_backend == "cule" else envs.reset()
    reset_obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result
    next_obs = to_tensor(reset_obs, device, torch.uint8)
    next_done = torch.zeros(args.num_envs, device=device, dtype=torch.bool)
    pbar = tqdm.tqdm(range(1, args.num_iterations + 1))
    global_step_burnin = None
    for iteration in pbar:
        if iteration == args.measure_burnin:
            global_step_burnin = global_step
            start_time = time.time()

        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"].copy_(lrnow)

        torch.compiler.cudagraph_mark_step_begin()
        next_obs, next_done, container = rollout(next_obs, next_done, avg_returns=avg_returns)
        global_step += container.numel()

        torch.compiler.cudagraph_mark_step_begin()
        with torch.no_grad():
            container = gae(next_obs, next_done, container)
        container_flat = container.view(-1)

        # Optimizing the policy and value network
        clipfracs = []
        for epoch in range(args.update_epochs):
            b_inds = torch.randperm(container_flat.shape[0], device=device).split(args.minibatch_size)
            for b in b_inds:
                container_local = container_flat[b]

                torch.compiler.cudagraph_mark_step_begin()
                out = update(container_local, tensordict_out=tensordict.TensorDict())
                if args.target_kl is not None and out["approx_kl"] > args.target_kl:
                    break
            else:
                continue
            break

        if global_step_burnin is not None and iteration % 10 == 0:
            cur_time = time.time()
            speed = (global_step - global_step_burnin) / (cur_time - start_time)
            global_step_burnin = global_step
            start_time = cur_time

            r = container["rewards"].mean()
            r_max = container["rewards"].max()
            avg_returns_t = torch.tensor(avg_returns).mean() if avg_returns else torch.tensor(float("nan"))

            with torch.no_grad():
                logs = {
                    "episode_return": np.array(avg_returns).mean(),
                    "logprobs": container["logprobs"].mean(),
                    "advantages": container["advantages"].mean(),
                    "returns": container["returns"].mean(),
                    "vals": container["vals"].mean(),
                    "gn": out["gn"].mean(),
                }

            lr = optimizer.param_groups[0]["lr"]
            pbar.set_description(
                f"speed: {speed: 4.1f} sps, "
                f"reward avg: {r :4.2f}, "
                f"reward max: {r_max:4.2f}, "
                f"returns: {avg_returns_t: 4.2f},"
                f"lr: {lr: 4.2f}"
            )
            if args.track:
                wandb.log(
                    {"speed": speed, "episode_return": avg_returns_t, "r": r, "r_max": r_max, "lr": lr, **logs},
                    step=global_step,
                )

    envs.close()
