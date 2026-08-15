# PPO with Transformer-XL episodic memory, for Atari.
#
# Pleines et al., JMLR 2025, "Memory Gym: Towards Endless Tasks to Benchmark
# Memory Capabilities of Agents" (https://arxiv.org/abs/2309.17207), whose
# TrXL-PPO baseline is the reference implementation here. Ported from
# MarcoMeter/episodic-transformer-memory-ppo and its CleanRL contribution
# `cleanrl/ppo_trxl/ppo_trxl.py` (MIT).
#
# This is the attention counterpart to `ppo_atari_lstm.py`. Instead of carrying
# one recurrent state forward, the agent keeps the *hidden state of every
# transformer layer at every past timestep of the current episode* and, at each
# step, attends over a sliding window of the most recent `--trxl-memory-length`
# of them. Nothing is compressed into a fixed-size vector, so a fact observed
# 100 steps ago is still available verbatim.
#
# ## The Atari adaptation, stated plainly
#
# The reference allocates memory for a whole episode:
# `[num_envs, max_episode_steps, num_layers, dim]`. Memory Gym episodes are a
# few hundred steps, so that tensor is tens of megabytes. An Atari episode runs
# to 108,000 frames — 27,000 agent steps at frame-skip 4 — and the same tensor
# at 3 layers and dim 384 is **4.0 GB at 32 envs and 16 GB at the 128 envs this
# repo normally runs**, on top of the per-rollout window buffer. That is the
# concrete reason no one publishes Atari numbers for this architecture.
#
# So the episode is not the memory horizon here: `--trxl-max-episode-steps`
# (default 1024) is, which brings the same tensor to 151 MB at 32 envs. The
# agent's memory is cleared every 1024 steps *and* at every true episode
# boundary, whichever comes first. At the default `--trxl-memory-length 119`
# that costs at most 119 steps of context once every 1024 — about 11% of steps
# run with a partly-empty window. The bound is a deliberate, load-bearing
# deviation, not an implementation detail.
#
# The other memory cost does not depend on the horizon at all: the stored
# attention windows are `num_steps * num_envs * memory_length * num_layers * dim`
# floats, 2.2 GB at 128 steps, 32 envs and the default widths. That is why
# `--num-steps` defaults to 128 here rather than the reference's 512.
#
# Two smaller adaptations: the encoder takes this repo's 4-channel stacked
# grayscale rather than the reference's 3-channel RGB (the frame stack is
# redundant with the memory but every backend here produces it), and the
# multi-discrete action head is collapsed to a single `Categorical`, Atari
# having one action dimension.
#
# ## Details from the reference that matter
#
#   * The attention mask is `tril(ones(L, L), diagonal=-1)` indexed by the
#     *current episode step*, which is what stops a young episode attending to
#     memory slots that have not been written yet.
#   * Query normalisation and key/value normalisation are **separate**
#     LayerNorms, applied pre-attention; `key = value` after norming, so it is
#     genuine self-attention over the memory with the current step as query.
#   * The energy is scaled by `sqrt(embed_dim)`, not `sqrt(head_size)` — a
#     deviation from Vaswani et al. that the reference makes and that this port
#     reproduces, because changing it changes the trained model.
#   * Masked positions are filled with `-1e20`, not `-inf`: an all-masked row
#     would otherwise softmax to NaN.
#   * The stored memory for layer `i` is the layer's *input*, detached.
#   * The learning rate and the entropy coefficient are both linearly annealed
#     from an initial to a final value over `--anneal-steps`, and then held.
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
    init_lr: float = 2.75e-4
    """the initial learning rate"""
    final_lr: float = 1.0e-5
    """the learning rate after annealing"""
    num_envs: int = 32
    """the number of parallel game environments"""
    num_steps: int = 128
    """rollout length per environment (the reference uses 512 on Memory Gym)"""
    anneal_steps: int = 10000000
    """steps over which the learning rate and entropy coefficient are annealed"""
    gamma: float = 0.995
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 8
    """the number of mini-batches"""
    update_epochs: int = 3
    """the K epochs to update the policy"""
    norm_adv: bool = False
    """Toggles advantages normalization (the reference leaves this off)"""
    clip_coef: float = 0.1
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function"""
    init_ent_coef: float = 0.001
    """initial coefficient of the entropy bonus"""
    final_ent_coef: float = 0.000001
    """final coefficient of the entropy bonus after annealing"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.25
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

    # Transformer-XL specific arguments
    trxl_num_layers: int = 3
    """the number of transformer layers"""
    trxl_num_heads: int = 4
    """the number of attention heads"""
    trxl_dim: int = 384
    """the transformer's embedding dimension"""
    trxl_memory_length: int = 119
    """the length of the sliding memory window attended over"""
    trxl_max_episode_steps: int = 1024
    """the memory horizon; memory is cleared every this many steps. See the header"""
    trxl_positional_encoding: str = "absolute"
    """`absolute`, `learned`, or `` for none"""

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
    benchmark_warmup_iterations: int = 2
    """full training iterations excluded from benchmark timing"""
    benchmark_measure_iterations: int = 5
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
# Transformer-XL
# ---------------------------------------------------------------------------

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    if layer.bias is not None:
        torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def batched_index_select(source, dim, index):
    """Gather a per-row window out of a batched tensor.

    `source` is `[N, T, ...]` and `index` is `[N, L]`; the result is `[N, L, ...]`
    where row `n` takes `source[n, index[n]]`. Straight from the reference.
    """
    for axis in range(1, len(source.shape)):
        if axis != dim:
            index = index.unsqueeze(axis)
    expanse = list(source.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.expand(expanse)
    return torch.gather(source, dim, index)


class PositionalEncoding(nn.Module):
    """The reference's sinusoidal encoding, indexed backwards from the present.

    `seq = arange(seq_len - 1, -1, -1)`, so position 0 of the table is the
    *oldest* slot. The frequency ladder uses `arange(0, dim, min_timescale=2.0)`,
    which yields `dim / 2` frequencies and hence a `dim`-wide sin/cos concat.
    """

    def __init__(self, dim, min_timescale=2.0, max_timescale=1e4):
        super().__init__()
        freqs = torch.arange(0, dim, min_timescale)
        inv_freqs = max_timescale ** (-freqs / dim)
        self.register_buffer("inv_freqs", inv_freqs)

    def forward(self, seq_len):
        seq = torch.arange(seq_len - 1, -1, -1.0, device=self.inv_freqs.device)
        sinusoidal = seq.unsqueeze(-1) * self.inv_freqs.unsqueeze(0)
        return torch.cat((sinusoidal.sin(), sinusoidal.cos()), dim=-1)


class MultiHeadAttention(nn.Module):
    """Multi-head attention, without dropout, in the reference's exact shape.

    Two things differ from the textbook version and both are preserved:
    the value/key/query projections act on the *per-head* slice
    (`head_size -> head_size`) rather than on the full embedding, and the
    energy is divided by `sqrt(embed_dim)` rather than `sqrt(head_size)`.
    """

    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_size = embed_dim // num_heads
        if self.head_size * num_heads != embed_dim:
            raise ValueError("trxl_dim must be divisible by trxl_num_heads")

        self.values = nn.Linear(self.head_size, self.head_size, bias=False)
        self.keys = nn.Linear(self.head_size, self.head_size, bias=False)
        self.queries = nn.Linear(self.head_size, self.head_size, bias=False)
        self.fc_out = nn.Linear(num_heads * self.head_size, embed_dim)

    def forward(self, values, keys, query, mask):
        batch = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        values = values.reshape(batch, value_len, self.num_heads, self.head_size)
        keys = keys.reshape(batch, key_len, self.num_heads, self.head_size)
        query = query.reshape(batch, query_len, self.num_heads, self.head_size)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(query)

        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        if mask is not None:
            # -1e20 rather than -inf: a fully masked row would softmax to NaN.
            energy = energy.masked_fill(mask.unsqueeze(1).unsqueeze(1) == 0, float("-1e20"))
        attention = torch.softmax(energy / (self.embed_dim ** (1 / 2)), dim=3)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values])
        out = out.reshape(batch, query_len, self.num_heads * self.head_size)
        return self.fc_out(out), attention


class TransformerLayer(nn.Module):
    """Pre-norm attention block with two residual connections."""

    def __init__(self, dim, num_heads):
        super().__init__()
        self.attention = MultiHeadAttention(dim, num_heads)
        self.layer_norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.layer_norm_attn = nn.LayerNorm(dim)
        self.fc_projection = nn.Sequential(nn.Linear(dim, dim), nn.ReLU())

    def forward(self, value, key, query, mask):
        query_ = self.layer_norm_q(query)
        value = self.norm_kv(value)
        key = value  # K = V: self-attention over the memory window
        attention, attention_weights = self.attention(value, key, query_, mask)
        x = attention + query
        x_ = self.layer_norm_attn(x)
        out = self.fc_projection(x_) + x
        return out, attention_weights


class Transformer(nn.Module):
    """Stacked `TransformerLayer`s over a per-step memory window.

    Returns the new embedding and the per-layer memories to be written back —
    memory `i` is the *input* to layer `i`, detached, so gradients never flow
    across timesteps through the memory.
    """

    def __init__(self, num_layers, dim, num_heads, max_episode_steps, positional_encoding):
        super().__init__()
        self.max_episode_steps = max_episode_steps
        self.positional_encoding = positional_encoding
        if positional_encoding == "absolute":
            self.pos_embedding = PositionalEncoding(dim)
        elif positional_encoding == "learned":
            self.pos_embedding = nn.Parameter(torch.randn(max_episode_steps, dim))
        elif positional_encoding:
            raise ValueError(f"unsupported positional encoding: {positional_encoding}")
        self.transformer_layers = nn.ModuleList(
            [TransformerLayer(dim, num_heads) for _ in range(num_layers)])

    def forward(self, x, memories, mask, memory_indices):
        if self.positional_encoding == "absolute":
            pos_embedding = self.pos_embedding(self.max_episode_steps)[memory_indices]
            memories = memories + pos_embedding.unsqueeze(2)
        elif self.positional_encoding == "learned":
            memories = memories + self.pos_embedding[memory_indices].unsqueeze(2)

        out_memories = []
        for index, layer in enumerate(self.transformer_layers):
            out_memories.append(x.detach())
            x, _ = layer(memories[:, :, index], memories[:, :, index], x.unsqueeze(1), mask)
            x = x.squeeze(1)
        return x, torch.stack(out_memories, dim=1)


class Agent(nn.Module):
    def __init__(self, envs, args, max_episode_steps):
        super().__init__()
        channels = envs.single_observation_space.shape[0]
        self.encoder = nn.Sequential(
            layer_init(nn.Conv2d(channels, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 7 * 7, args.trxl_dim)),
            nn.ReLU(),
        )
        self.transformer = Transformer(
            args.trxl_num_layers, args.trxl_dim, args.trxl_num_heads,
            max_episode_steps, args.trxl_positional_encoding)
        self.hidden_post_trxl = nn.Sequential(
            layer_init(nn.Linear(args.trxl_dim, args.trxl_dim)),
            nn.ReLU(),
        )
        self.actor = layer_init(nn.Linear(args.trxl_dim, envs.single_action_space.n), std=np.sqrt(0.01))
        self.critic = layer_init(nn.Linear(args.trxl_dim, 1), std=1)

    def _trunk(self, x, memory, memory_mask, memory_indices):
        x = self.encoder(x / 255.0)
        x, memory = self.transformer(x, memory, memory_mask, memory_indices)
        return self.hidden_post_trxl(x), memory

    def get_value(self, x, memory, memory_mask, memory_indices):
        hidden, _ = self._trunk(x, memory, memory_mask, memory_indices)
        return self.critic(hidden).flatten()

    def get_action_and_value(self, x, memory, memory_mask, memory_indices, action=None):
        hidden, memory = self._trunk(x, memory, memory_mask, memory_indices)
        probs = Categorical(logits=self.actor(hidden))
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden).flatten(), memory


def build_memory_index_table(max_episode_steps, memory_length, device=None):
    """The per-step memory window, precomputed for every episode step.

    For steps beyond the warm-up the window is `[t - L + 1, t]`. For the first
    `L - 1` steps there is no such window, so the reference repeats
    `arange(0, L)`: a young episode looks at the same `L` slots every step and
    relies on the mask to hide the ones it has not written yet.
    """
    tail = torch.stack([torch.arange(i, i + memory_length)
                        for i in range(max_episode_steps - memory_length + 1)])
    warmup = tail[0].repeat(memory_length - 1, 1)
    return torch.cat((warmup, tail)).to(device)


def build_memory_mask(memory_length, device=None):
    """`tril(ones(L, L), diagonal=-1)`, indexed by the current episode step.

    Row `t` has `t` ones, so at episode step `t < L` exactly the `t` slots
    already written are visible. Row 0 is all zeros — the very first step of an
    episode attends to nothing, and `-1e20` rather than `-inf` is what keeps
    that from becoming NaN.
    """
    return torch.tril(torch.ones((memory_length, memory_length), device=device), diagonal=-1)


def linear_anneal(initial, final, consumed_steps, anneal_steps):
    """Linear from `initial` to `final` over `anneal_steps`, then held."""
    if anneal_steps <= 0:
        return final
    fraction = min(1.0, consumed_steps / anneal_steps)
    return initial + fraction * (final - initial)


if __name__ == "__main__":
    process_start = time.perf_counter()
    args = tyro.cli(Args)
    if args.num_envs < 1 or args.num_steps < 1:
        raise ValueError("num_envs and num_steps must be positive")
    if args.num_minibatches < 1:
        raise ValueError("num_minibatches must be positive")
    if args.trxl_memory_length < 1:
        raise ValueError("trxl_memory_length must be positive")
    if args.trxl_max_episode_steps < args.trxl_memory_length:
        raise ValueError("trxl_max_episode_steps must be at least trxl_memory_length")
    if args.benchmark_warmup_iterations < 0 or args.benchmark_measure_iterations < 1:
        raise ValueError("invalid benchmark window")
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
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

    max_episode_steps = args.trxl_max_episode_steps
    agent = Agent(envs, args, max_episode_steps).to(device)
    optimizer = optim.AdamW(agent.parameters(), lr=args.init_lr)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs), dtype=torch.long).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    stored_memory_masks = torch.zeros(
        (args.num_steps, args.num_envs, args.trxl_memory_length), dtype=torch.bool, device=device)
    stored_memory_indices = torch.zeros(
        (args.num_steps, args.num_envs, args.trxl_memory_length), dtype=torch.long, device=device)
    # The memory windows actually attended over, saved so the learner replays
    # the same context the actor saw.
    stored_memory_windows = torch.zeros(
        (args.num_steps, args.num_envs, args.trxl_memory_length, args.trxl_num_layers, args.trxl_dim),
        device=device)

    # The live per-episode memory, one slot per step of the memory horizon.
    next_memory = torch.zeros(
        (args.num_envs, max_episode_steps, args.trxl_num_layers, args.trxl_dim), device=device)
    memory_mask = build_memory_mask(args.trxl_memory_length, device)
    memory_indices = build_memory_index_table(max_episode_steps, args.trxl_memory_length, device)
    episode_step = torch.zeros(args.num_envs, dtype=torch.long, device=device)

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

        learning_rate = linear_anneal(args.init_lr, args.final_lr, global_step, args.anneal_steps)
        ent_coef = linear_anneal(args.init_ent_coef, args.final_ent_coef, global_step, args.anneal_steps)
        optimizer.param_groups[0]["lr"] = learning_rate

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            stored_memory_masks[step] = memory_mask[
                torch.clip(episode_step, 0, args.trxl_memory_length - 1)].bool()
            stored_memory_indices[step] = memory_indices[episode_step]
            window = batched_index_select(next_memory, 1, stored_memory_indices[step])
            stored_memory_windows[step] = window

            with torch.no_grad():
                action, logprob, _, value, new_memory = agent.get_action_and_value(
                    next_obs, window, stored_memory_masks[step], stored_memory_indices[step])
                values[step] = value
            actions[step] = action
            logprobs[step] = logprob
            next_memory[torch.arange(args.num_envs, device=device), episode_step] = new_memory

            next_obs, reward, terminations, truncations, infos = step_env(envs, action)
            next_done = done_tensor(terminations, truncations, device)
            rewards[step] = to_tensor(reward, device).view(-1)
            next_obs = to_tensor(next_obs, device)

            # Clear the memory at an episode boundary *or* at the memory
            # horizon, whichever comes first. See the header.
            episode_step = episode_step + 1
            expired = (next_done.bool()) | (episode_step >= max_episode_steps)
            if expired.any():
                next_memory[expired] = 0.0
                episode_step = torch.where(expired, torch.zeros_like(episode_step), episode_step)

            stats.update(completed_episode_infos(infos, next_done), global_step, writer)

        with torch.no_grad():
            bootstrap_window = batched_index_select(
                next_memory, 1, memory_indices[episode_step])
            next_value = agent.get_value(
                next_obs, bootstrap_window,
                memory_mask[torch.clip(episode_step, 0, args.trxl_memory_length - 1)].bool(),
                memory_indices[episode_step])
            advantages = torch.zeros_like(rewards)
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
        b_actions = actions.reshape(-1)
        b_logprobs = logprobs.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        b_memory_windows = stored_memory_windows.reshape(
            -1, args.trxl_memory_length, args.trxl_num_layers, args.trxl_dim)
        b_memory_masks = stored_memory_masks.reshape(-1, args.trxl_memory_length)
        b_memory_indices = stored_memory_indices.reshape(-1, args.trxl_memory_length)

        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                mb_inds = b_inds[start : start + args.minibatch_size]

                _, newlogprob, entropy, newvalue, _ = agent.get_action_and_value(
                    b_obs[mb_inds], b_memory_windows[mb_inds], b_memory_masks[mb_inds],
                    b_memory_indices[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                pg_loss = torch.max(
                    -mb_advantages * ratio,
                    -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef),
                ).mean()

                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds], -args.clip_coef, args.clip_coef)
                    v_loss = 0.5 * torch.max(v_loss_unclipped, (v_clipped - b_returns[mb_inds]) ** 2).mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        if writer is not None:
            writer.add_scalar("charts/learning_rate", learning_rate, global_step)
            writer.add_scalar("charts/entropy_coefficient", ent_coef, global_step)
            writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
            writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
            writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
            writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
            writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
            print("SPS:", int(global_step / (time.time() - start_time)))
            writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    if args.benchmark:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        benchmark_end = time.perf_counter()
        measured_steps = global_step - benchmark_start_step
        measured_seconds = benchmark_end - benchmark_start
        result = {
            "algorithm": "ppo_trxl",
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
            "trxl_dim": args.trxl_dim,
            "trxl_max_episode_steps": args.trxl_max_episode_steps,
            "trxl_memory_length": args.trxl_memory_length,
            "trxl_num_layers": args.trxl_num_layers,
            "update_epochs": args.update_epochs,
            "warmup_iterations": args.benchmark_warmup_iterations,
        }
        print(f"BENCHMARK_RESULT {json.dumps(result, sort_keys=True)}", flush=True)

    envs.close()
    if writer is not None:
        writer.close()
