# DreamerV3: learn a world model from replayed experience, then train an
# actor-critic entirely inside that model's imagination.
#
# "Mastering diverse control tasks through world models", Hafner, Pasukonis,
# Ba & Lillicrap, Nature 2025 (arXiv:2301.04104). Official JAX implementation:
# github.com/danijar/dreamerv3. This port follows the widely used PyTorch
# reproduction github.com/NM512/dreamerv3-torch, which is what
# tests/crosscheck/check_dreamer.py diffs against component by component.
#
# Structure follows this repository's CleanRL convention, but the observation
# pipeline is DreamerV3's, not the Atari DQN one used everywhere else here:
# 64x64 RGB, action repeat 4, NO frame stacking (the recurrent state carries
# history), no reward clipping, no episodic-life resets. Those are load-bearing
# -- the encoder halves resolution until it reaches 4x4, so 84 would not divide.
#
# CuLE is not a supported backend for this file. Its wrapper is hardwired to
# 4x84x84 grayscale, and Atari-100K is replay-bound anyway (one gradient step
# per environment step at batch 16x64), so the environment is not the
# bottleneck a faster simulator would relieve.
#
# The defaults below are the `atari100k` block of upstream's configs.yaml
# merged over its `defaults` block.
import json
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
import torch.distributions as torchd
import torch.nn as nn
import torch.nn.functional as F
import tyro
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

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
    env_backend: str = "envpool"
    """environment backend: `envpool` or `gymnasium` (CuLE cannot emit 64x64 RGB)"""

    # Algorithm specific arguments
    env_id: str = "Pong-v5"
    """the id of the environment"""
    total_timesteps: int = 400000
    """total environment frames; Atari-100K is 400k frames = 100k policy steps"""
    num_envs: int = 1
    """the number of parallel game environments"""
    action_repeat: int = 4
    """frames each action is held for"""
    buffer_size: int = 1000000
    """replay capacity in policy steps"""
    prefill: int = 2500
    """policy steps of random actions before training starts"""
    batch_size: int = 16
    """sequences per world-model batch"""
    batch_length: int = 64
    """timesteps per replayed sequence"""
    train_ratio: int = 1024
    """replayed steps per policy step; 1024 with a 16x64 batch is one update per step"""

    # World model
    dyn_stoch: int = 32
    """categorical variables in the stochastic state"""
    dyn_discrete: int = 32
    """classes per categorical variable"""
    dyn_deter: int = 512
    """size of the deterministic recurrent state"""
    dyn_hidden: int = 512
    """hidden width inside the RSSM"""
    cnn_depth: int = 32
    """channels in the first encoder stage; doubles per stage"""
    units: int = 512
    """hidden width of the MLP heads"""
    model_lr: float = 1e-4
    """world-model learning rate"""
    opt_eps: float = 1e-8
    """Adam epsilon for the world model"""
    grad_clip: float = 1000.0
    """world-model gradient clipping"""
    kl_free: float = 1.0
    """free nats below which the KL terms stop pulling"""
    dyn_scale: float = 0.5
    """weight on the dynamics KL (prior towards posterior)"""
    rep_scale: float = 0.1
    """weight on the representation KL (posterior towards prior)"""
    unimix_ratio: float = 0.01
    """uniform mixture folded into every categorical"""

    # Actor critic in imagination
    imag_horizon: int = 15
    """imagination rollout length"""
    discount: float = 0.997
    """discount used inside imagination"""
    discount_lambda: float = 0.95
    """lambda for the imagination return"""
    actor_lr: float = 3e-5
    """actor learning rate"""
    critic_lr: float = 3e-5
    """critic learning rate"""
    ac_opt_eps: float = 1e-5
    """Adam epsilon for actor and critic"""
    ac_grad_clip: float = 100.0
    """actor/critic gradient clipping"""
    actor_entropy: float = 3e-4
    """entropy bonus on the imagined policy"""
    imag_gradient: str = "reinforce"
    """`reinforce` (the Atari-100K setting) or `dynamics` (straight-through)"""
    slow_target_fraction: float = 0.02
    """Polyak rate of the slow critic"""
    reward_ema: bool = True
    """normalise imagination returns by a 5th/95th percentile EMA"""

    max_training_seconds: float = 0.0
    """wall-clock training limit; zero disables it"""
    solve_reward: float | None = None
    """stop when the moving episodic return reaches this value"""
    solve_window: int = 20
    """number of completed episodes in the solve moving average"""
    benchmark: bool = False
    """run a fixed warmup/measurement window and print a JSON benchmark result"""
    benchmark_warmup_iterations: int = 5
    """policy steps excluded from benchmark timing"""
    benchmark_measure_iterations: int = 20
    """policy steps included in benchmark timing"""


# ---------------------------------------------------------------------------
# Distributions and transforms
# ---------------------------------------------------------------------------


def symlog(x):
    """Compresses large magnitudes without a tuned reward scale (Section 3)."""
    return torch.sign(x) * torch.log(torch.abs(x) + 1.0)


def symexp(x):
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1.0)


class TwoHotDist:
    """255-bin categorical over symlog space, fit with a two-hot target.

    Rewards and values are predicted as a distribution over fixed bins rather
    than regressed, which is what lets one hyperparameter set span reward scales
    from 0.01 to 100000 without clipping.
    """

    def __init__(self, logits, low=-20.0, high=20.0, device="cpu"):
        self.logits = logits
        self.probs = torch.softmax(logits, -1)
        self.buckets = torch.linspace(low, high, steps=255, device=logits.device)

    def mean(self):
        return symexp(torch.sum(self.probs * self.buckets, dim=-1, keepdim=True))

    def mode(self):
        return self.mean()

    def log_prob(self, x):
        x = symlog(x)
        below = torch.sum((self.buckets <= x[..., None]).to(torch.int32), dim=-1) - 1
        above = len(self.buckets) - torch.sum(
            (self.buckets > x[..., None]).to(torch.int32), dim=-1
        )
        below = torch.clip(below, 0, len(self.buckets) - 1)
        above = torch.clip(above, 0, len(self.buckets) - 1)
        equal = below == above
        distance_below = torch.where(equal, 1, torch.abs(self.buckets[below] - x))
        distance_above = torch.where(equal, 1, torch.abs(self.buckets[above] - x))
        total = distance_below + distance_above
        weight_below = distance_above / total
        weight_above = distance_below / total
        target = (
            F.one_hot(below, num_classes=len(self.buckets)) * weight_below[..., None]
            + F.one_hot(above, num_classes=len(self.buckets)) * weight_above[..., None]
        )
        log_prediction = self.logits - torch.logsumexp(self.logits, -1, keepdim=True)
        return (target.squeeze(-2) * log_prediction).sum(-1)


class OneHotDist(torchd.one_hot_categorical.OneHotCategorical):
    """Categorical with a uniform mixture and straight-through gradients."""

    def __init__(self, logits=None, probs=None, unimix_ratio=0.0):
        # validate_args is off deliberately: `sample` returns a straight-through
        # estimate, one-hot plus (probs - probs.detach()), which is one-hot in
        # value but not bit-exactly so, and torch's support check rejects it.
        if logits is not None and unimix_ratio > 0.0:
            probs = F.softmax(logits, dim=-1)
            probs = probs * (1.0 - unimix_ratio) + unimix_ratio / probs.shape[-1]
            logits = torch.log(probs)
            super().__init__(logits=logits, probs=None, validate_args=False)
        else:
            super().__init__(logits=logits, probs=probs, validate_args=False)

    def mode(self):
        mode = F.one_hot(torch.argmax(super().logits, axis=-1), super().logits.shape[-1])
        return mode.detach() + super().logits - super().logits.detach()

    def sample(self, sample_shape=()):
        sample = super().sample(sample_shape).detach()
        probs = super().probs
        while len(probs.shape) < len(sample.shape):
            probs = probs[None]
        return sample + probs - probs.detach()


class MSEDist:
    """Gaussian-with-unit-variance log-likelihood, summed over pixels."""

    def __init__(self, mode):
        self._mode = mode

    def mode(self):
        return self._mode

    def log_prob(self, value):
        distance = (self._mode - value) ** 2
        return -distance.sum(list(range(len(distance.shape)))[2:])


class BernoulliDist:
    """Continuation head: probability the episode has not ended."""

    def __init__(self, logits):
        self.logits = logits
        self._dist = torchd.bernoulli.Bernoulli(logits=logits)

    @property
    def mean(self):
        return self._dist.mean

    def mode(self):
        return torch.round(self._dist.mean)

    def log_prob(self, x):
        log_probs0 = -F.softplus(self.logits)
        log_probs1 = -F.softplus(-self.logits)
        return (log_probs0 * (1 - x) + log_probs1 * x).squeeze(-1)


# ---------------------------------------------------------------------------
# Initialisation, matching upstream's truncated-normal and uniform schemes
# ---------------------------------------------------------------------------

_TRUNC_NORMAL_CORRECTION = 0.87962566103423978


def weight_init(module):
    if isinstance(module, nn.Linear):
        scale = 1.0 / ((module.in_features + module.out_features) / 2.0)
        std = np.sqrt(scale) / _TRUNC_NORMAL_CORRECTION
        nn.init.trunc_normal_(module.weight.data, mean=0.0, std=std, a=-2.0 * std, b=2.0 * std)
        if getattr(module, "bias", None) is not None:
            module.bias.data.fill_(0.0)
    elif isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        space = module.kernel_size[0] * module.kernel_size[1]
        scale = 1.0 / ((space * module.in_channels + space * module.out_channels) / 2.0)
        std = np.sqrt(scale) / _TRUNC_NORMAL_CORRECTION
        nn.init.trunc_normal_(module.weight.data, mean=0.0, std=std, a=-2.0 * std, b=2.0 * std)
        if getattr(module, "bias", None) is not None:
            module.bias.data.fill_(0.0)
    elif isinstance(module, nn.LayerNorm):
        module.weight.data.fill_(1.0)
        if getattr(module, "bias", None) is not None:
            module.bias.data.fill_(0.0)


def uniform_weight_init(given_scale):
    """Used for the distribution heads; `outscale=0` starts a head at zero."""

    def initialise(module):
        if isinstance(module, nn.Linear):
            scale = given_scale / ((module.in_features + module.out_features) / 2.0)
            limit = np.sqrt(3 * scale)
            nn.init.uniform_(module.weight.data, a=-limit, b=limit)
            if getattr(module, "bias", None) is not None:
                module.bias.data.fill_(0.0)
        elif isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            space = module.kernel_size[0] * module.kernel_size[1]
            scale = given_scale / ((space * module.in_channels + space * module.out_channels) / 2.0)
            limit = np.sqrt(3 * scale)
            nn.init.uniform_(module.weight.data, a=-limit, b=limit)
            if getattr(module, "bias", None) is not None:
                module.bias.data.fill_(0.0)
        elif isinstance(module, nn.LayerNorm):
            module.weight.data.fill_(1.0)
            if getattr(module, "bias", None) is not None:
                module.bias.data.fill_(0.0)

    return initialise


# ---------------------------------------------------------------------------
# Convolutional encoder and decoder
# ---------------------------------------------------------------------------


class Conv2dSamePad(nn.Conv2d):
    """TensorFlow-style SAME padding, which upstream inherits from the JAX code."""

    @staticmethod
    def _pad(size, kernel, stride, dilation):
        return max((int(np.ceil(size / stride)) - 1) * stride + (kernel - 1) * dilation + 1 - size, 0)

    def forward(self, x):
        height, width = x.shape[-2:]
        pad_h = self._pad(height, self.kernel_size[0], self.stride[0], self.dilation[0])
        pad_w = self._pad(width, self.kernel_size[1], self.stride[1], self.dilation[1])
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2])
        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class ImgChLayerNorm(nn.Module):
    """LayerNorm over the channel axis of an NCHW tensor."""

    def __init__(self, channels, eps=1e-3):
        super().__init__()
        self.norm = nn.LayerNorm(channels, eps=eps)

    def forward(self, x):
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class ConvEncoder(nn.Module):
    """Halve the resolution until it reaches `minres`, doubling channels."""

    def __init__(self, shape=(3, 64, 64), depth=32, kernel_size=4, minres=4):
        super().__init__()
        channels, height, _ = shape
        stages = int(np.log2(height) - np.log2(minres))
        layers = []
        in_dim, out_dim = channels, depth
        for _ in range(stages):
            layers.append(Conv2dSamePad(in_dim, out_dim, kernel_size, stride=2, bias=False))
            layers.append(ImgChLayerNorm(out_dim))
            layers.append(nn.SiLU())
            in_dim, out_dim = out_dim, out_dim * 2
            height //= 2
        self.outdim = in_dim * height * height
        self.layers = nn.Sequential(*layers)
        self.layers.apply(weight_init)

    def forward(self, observations):
        """(batch, time, C, H, W) in [0, 1] -> (batch, time, outdim)."""
        leading = observations.shape[:-3]
        x = observations.reshape((-1,) + tuple(observations.shape[-3:])) - 0.5
        x = self.layers(x)
        return x.reshape(list(leading) + [int(np.prod(x.shape[1:]))])


class ConvDecoder(nn.Module):
    """Mirror of the encoder, from a latent feature back to a 64x64 image."""

    def __init__(self, feat_size, shape=(3, 64, 64), depth=32, kernel_size=4, minres=4, outscale=1.0):
        super().__init__()
        self._shape = shape
        self._minres = minres
        stages = int(np.log2(shape[1]) - np.log2(minres))
        out_ch = minres**2 * depth * 2 ** (stages - 1)
        self._embed_size = out_ch
        self._linear_layer = nn.Linear(feat_size, out_ch)
        self._linear_layer.apply(uniform_weight_init(outscale))

        in_dim = out_ch // (minres**2)
        out_dim = in_dim // 2
        layers = []
        for index in range(stages):
            bias, activation, normalise = False, True, True
            if index == stages - 1:
                out_dim, bias, activation, normalise = shape[0], True, False, False
            if index != 0:
                in_dim = 2 ** (stages - (index - 1) - 2) * depth
            pad, outpad = self._same_pad(kernel_size, 2, 1)
            layers.append(
                nn.ConvTranspose2d(in_dim, out_dim, kernel_size, 2, padding=pad, output_padding=outpad, bias=bias)
            )
            if normalise:
                layers.append(ImgChLayerNorm(out_dim))
            if activation:
                layers.append(nn.SiLU())
            in_dim, out_dim = out_dim, out_dim // 2
        for module in layers[:-1]:
            module.apply(weight_init)
        layers[-1].apply(uniform_weight_init(outscale))
        self.layers = nn.Sequential(*layers)

    @staticmethod
    def _same_pad(kernel, stride, dilation):
        value = dilation * (kernel - 1) - stride + 1
        pad = int(np.ceil(value / 2))
        return pad, pad * 2 - value

    def forward(self, features):
        x = self._linear_layer(features)
        # The linear output is laid out as (h, w, ch) and then permuted, not as
        # (ch, h, w) directly -- reshaping straight to NCHW transposes the
        # channel and spatial axes and silently produces a different network.
        x = x.reshape([-1, self._minres, self._minres, self._embed_size // self._minres**2])
        x = x.permute(0, 3, 1, 2)
        x = self.layers(x)
        return x.reshape(tuple(features.shape[:-1]) + self._shape) + 0.5


# ---------------------------------------------------------------------------
# Recurrent state-space model
# ---------------------------------------------------------------------------


class GRUCell(nn.Module):
    """Layer-normalised GRU with a -1 update bias, as in the paper's appendix."""

    def __init__(self, input_size, size, update_bias=-1.0):
        super().__init__()
        self._size = size
        self._update_bias = update_bias
        self.layers = nn.Sequential(
            nn.Linear(input_size + size, 3 * size, bias=False),
            nn.LayerNorm(3 * size, eps=1e-3),
        )

    def forward(self, inputs, state):
        parts = self.layers(torch.cat([inputs, state], -1))
        reset, candidate, update = torch.split(parts, [self._size] * 3, -1)
        reset = torch.sigmoid(reset)
        candidate = torch.tanh(reset * candidate)
        update = torch.sigmoid(update + self._update_bias)
        output = update * candidate + (1 - update) * state
        return output


class RSSM(nn.Module):
    """Deterministic GRU path plus a categorical stochastic state.

    Two distributions are produced at each step: the *prior* from the previous
    state and action alone (what the model can imagine), and the *posterior*
    which also sees the current observation embedding. Training pulls them
    together from both sides with separately scaled, separately clipped KLs --
    dyn_scale on the prior, rep_scale on the posterior -- which is what keeps
    the representation from collapsing into whatever the dynamics can predict.
    """

    def __init__(self, stoch=32, deter=512, hidden=512, discrete=32, unimix_ratio=0.01,
                 num_actions=None, embed=None):
        super().__init__()
        self._stoch = stoch
        self._deter = deter
        self._hidden = hidden
        self._discrete = discrete
        self._unimix_ratio = unimix_ratio
        self._num_actions = num_actions

        self._img_in_layers = nn.Sequential(
            nn.Linear(stoch * discrete + num_actions, hidden, bias=False),
            nn.LayerNorm(hidden, eps=1e-3),
            nn.SiLU(),
        )
        self._img_in_layers.apply(weight_init)
        self._cell = GRUCell(hidden, deter)
        self._cell.apply(weight_init)
        self._img_out_layers = nn.Sequential(
            nn.Linear(deter, hidden, bias=False), nn.LayerNorm(hidden, eps=1e-3), nn.SiLU()
        )
        self._img_out_layers.apply(weight_init)
        self._obs_out_layers = nn.Sequential(
            nn.Linear(deter + embed, hidden, bias=False), nn.LayerNorm(hidden, eps=1e-3), nn.SiLU()
        )
        self._obs_out_layers.apply(weight_init)
        self._imgs_stat_layer = nn.Linear(hidden, stoch * discrete)
        self._imgs_stat_layer.apply(uniform_weight_init(1.0))
        self._obs_stat_layer = nn.Linear(hidden, stoch * discrete)
        self._obs_stat_layer.apply(uniform_weight_init(1.0))
        # The initial deterministic state is learned, not zero.
        self.W = nn.Parameter(torch.zeros((1, deter)), requires_grad=True)

    @property
    def feat_size(self):
        return self._stoch * self._discrete + self._deter

    def initial(self, batch_size, device):
        deter = torch.tanh(self.W).repeat(batch_size, 1)
        state = {
            "logit": torch.zeros([batch_size, self._stoch, self._discrete], device=device),
            "stoch": torch.zeros([batch_size, self._stoch, self._discrete], device=device),
            "deter": deter,
        }
        state["stoch"] = self.get_stoch(deter)
        return state

    def get_feat(self, state):
        stoch = state["stoch"]
        stoch = stoch.reshape(list(stoch.shape[:-2]) + [self._stoch * self._discrete])
        return torch.cat([stoch, state["deter"]], -1)

    def get_dist(self, state):
        return torchd.independent.Independent(
            OneHotDist(state["logit"], unimix_ratio=self._unimix_ratio), 1
        )

    def get_stoch(self, deter):
        stats = self._suff_stats_layer("ims", self._img_out_layers(deter))
        return self.get_dist(stats).mode()

    def _suff_stats_layer(self, name, x):
        layer = self._imgs_stat_layer if name == "ims" else self._obs_stat_layer
        logit = layer(x)
        return {"logit": logit.reshape(list(logit.shape[:-1]) + [self._stoch, self._discrete])}

    def obs_step(self, prev_state, prev_action, embed, is_first, sample=True):
        """One filtering step: prior from (state, action), posterior with `embed`."""
        device = embed.device
        if prev_state is None or torch.sum(is_first) == len(is_first):
            prev_state = self.initial(len(is_first), device)
            prev_action = torch.zeros((len(is_first), self._num_actions), device=device)
        elif torch.sum(is_first) > 0:
            is_first = is_first[:, None]
            prev_action = prev_action * (1.0 - is_first)
            initial = self.initial(len(is_first), device)
            # A *new* dict, not an in-place update. `observe` keeps a reference
            # to every posterior it has produced, and the incoming state is the
            # previous timestep's posterior; overwriting it here would rewrite
            # history that has already been recorded, at every episode boundary.
            reset_state = {}
            for key, value in prev_state.items():
                reset = torch.reshape(is_first, is_first.shape + (1,) * (len(value.shape) - 2))
                reset_state[key] = value * (1.0 - reset) + initial[key] * reset
            prev_state = reset_state

        # The prior is always sampled, even when the posterior is taken at its
        # mode; `sample` only controls the posterior draw.
        prior = self.img_step(prev_state, prev_action)
        x = self._obs_out_layers(torch.cat([prior["deter"], embed], -1))
        stats = self._suff_stats_layer("obs", x)
        distribution = self.get_dist(stats)
        stoch = distribution.sample() if sample else distribution.mode()
        return {"stoch": stoch, "deter": prior["deter"], **stats}, prior

    def img_step(self, prev_state, prev_action, sample=True):
        """One imagination step: no observation is used."""
        prev_stoch = prev_state["stoch"]
        prev_stoch = prev_stoch.reshape(
            list(prev_stoch.shape[:-2]) + [self._stoch * self._discrete]
        )
        x = self._img_in_layers(torch.cat([prev_stoch, prev_action], -1))
        deter = self._cell(x, prev_state["deter"])
        stats = self._suff_stats_layer("ims", self._img_out_layers(deter))
        distribution = self.get_dist(stats)
        stoch = distribution.sample() if sample else distribution.mode()
        return {"stoch": stoch, "deter": deter, **stats}

    def observe(self, embed, action, is_first, state=None):
        """Filter a whole (batch, time) sequence; returns posteriors and priors."""
        swap = lambda x: x.permute([1, 0] + list(range(2, len(x.shape))))
        embed, action, is_first = swap(embed), swap(action), swap(is_first)
        posts, priors = [], []
        post = state
        for t in range(embed.shape[0]):
            post, prior = self.obs_step(post, action[t], embed[t], is_first[t])
            posts.append(post)
            priors.append(prior)
        stack = lambda seq: {k: swap(torch.stack([s[k] for s in seq], 0)) for k in seq[0]}
        return stack(posts), stack(priors)

    def kl_loss(self, post, prior, free, dyn_scale, rep_scale):
        detach = lambda state: {k: v.detach() for k, v in state.items()}
        representation = torchd.kl.kl_divergence(self.get_dist(post), self.get_dist(detach(prior)))
        dynamics = torchd.kl.kl_divergence(self.get_dist(detach(post)), self.get_dist(prior))
        value = representation
        representation = torch.clip(representation, min=free)
        dynamics = torch.clip(dynamics, min=free)
        return dyn_scale * dynamics + rep_scale * representation, value, dynamics, representation


class MLP(nn.Module):
    """Two hidden layers, then a distribution head."""

    def __init__(self, in_dim, out_dim, layers, units, dist="symlog_disc", outscale=1.0,
                 unimix_ratio=0.0):
        super().__init__()
        self._dist = dist
        self._unimix_ratio = unimix_ratio
        blocks = []
        width = in_dim
        for _ in range(layers):
            blocks += [nn.Linear(width, units, bias=False), nn.LayerNorm(units, eps=1e-3), nn.SiLU()]
            width = units
        self.layers = nn.Sequential(*blocks)
        self.layers.apply(weight_init)
        self.head = nn.Linear(width, out_dim)
        self.head.apply(uniform_weight_init(outscale))

    def forward(self, features):
        out = self.head(self.layers(features))
        if self._dist == "symlog_disc":
            return TwoHotDist(out)
        if self._dist == "binary":
            return BernoulliDist(out)
        if self._dist == "onehot":
            return OneHotDist(out, unimix_ratio=self._unimix_ratio)
        raise NotImplementedError(self._dist)


# ---------------------------------------------------------------------------
# World model and imagination behaviour
# ---------------------------------------------------------------------------


class WorldModel(nn.Module):
    def __init__(self, num_actions, args, device):
        super().__init__()
        self.args = args
        self.encoder = ConvEncoder(depth=args.cnn_depth)
        self.dynamics = RSSM(
            stoch=args.dyn_stoch, deter=args.dyn_deter, hidden=args.dyn_hidden,
            discrete=args.dyn_discrete, unimix_ratio=args.unimix_ratio,
            num_actions=num_actions, embed=self.encoder.outdim,
        )
        feat_size = self.dynamics.feat_size
        self.decoder = ConvDecoder(feat_size, depth=args.cnn_depth)
        self.reward_head = MLP(feat_size, 255, 2, args.units, "symlog_disc", outscale=0.0)
        self.cont_head = MLP(feat_size, 1, 2, args.units, "binary", outscale=1.0)
        self.to(device)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=args.model_lr, eps=args.opt_eps)

    def loss(self, batch):
        """Reconstruction + reward + continuation + the two KL terms."""
        observations = batch["image"].float() / 255.0
        embed = self.encoder(observations)
        post, prior = self.dynamics.observe(embed, batch["action"], batch["is_first"])
        kl_loss, kl_value, dyn_loss, rep_loss = self.dynamics.kl_loss(
            post, prior, self.args.kl_free, self.args.dyn_scale, self.args.rep_scale
        )
        feat = self.dynamics.get_feat(post)
        image_loss = -MSEDist(self.decoder(feat)).log_prob(observations)
        reward_loss = -self.reward_head(feat).log_prob(batch["reward"].unsqueeze(-1))
        cont_loss = -self.cont_head(feat).log_prob(batch["cont"].unsqueeze(-1))
        model_loss = image_loss + reward_loss + cont_loss + kl_loss
        metrics = {
            "image_loss": image_loss.mean().detach(),
            "reward_loss": reward_loss.mean().detach(),
            "cont_loss": cont_loss.mean().detach(),
            "dyn_loss": dyn_loss.mean().detach(),
            "rep_loss": rep_loss.mean().detach(),
            "kl": kl_value.mean().detach(),
        }
        return model_loss.mean(), {k: v.detach() for k, v in post.items()}, metrics

    def train_step(self, batch):
        loss, post, metrics = self.loss(batch)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), self.args.grad_clip)
        self.optimizer.step()
        metrics["model_loss"] = loss.detach()
        return post, metrics


class RewardEMA:
    """Track the 5th and 95th percentile of imagined returns (Section 3).

    Returns are divided by the spread between them, floored at 1, so that
    scaling only ever shrinks large returns and never inflates small ones.
    """

    def __init__(self, alpha=1e-2):
        self.alpha = alpha

    def __call__(self, x, ema_vals):
        quantiles = torch.quantile(torch.flatten(x.detach()), torch.tensor([0.05, 0.95], device=x.device))
        ema_vals[:] = self.alpha * quantiles + (1 - self.alpha) * ema_vals
        scale = torch.clip(ema_vals[1] - ema_vals[0], min=1.0)
        return ema_vals[0].detach(), scale.detach()


def lambda_return(reward, value, pcont, bootstrap, lambda_):
    """Backwards lambda-return over an imagined rollout (time-major)."""
    next_values = torch.cat([value[1:], bootstrap[None]], 0)
    inputs = reward + pcont * next_values * (1 - lambda_)
    returns = []
    last = bootstrap
    for t in reversed(range(reward.shape[0])):
        last = inputs[t] + pcont[t] * lambda_ * last
        returns.append(last)
    return torch.stack(list(reversed(returns)), 0)


class ImagBehavior(nn.Module):
    def __init__(self, num_actions, world_model, args, device):
        super().__init__()
        self.args = args
        self._world_model = world_model
        feat_size = world_model.dynamics.feat_size
        self.actor = MLP(feat_size, num_actions, 2, args.units, "onehot",
                         outscale=1.0, unimix_ratio=args.unimix_ratio)
        self.critic = MLP(feat_size, 255, 2, args.units, "symlog_disc", outscale=0.0)
        self.slow_critic = MLP(feat_size, 255, 2, args.units, "symlog_disc", outscale=0.0)
        self.slow_critic.load_state_dict(self.critic.state_dict())
        self.register_buffer("ema_vals", torch.zeros(2))
        self.reward_ema = RewardEMA()
        self.to(device)
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=args.actor_lr, eps=args.ac_opt_eps
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=args.critic_lr, eps=args.ac_opt_eps
        )

    def imagine(self, start, horizon):
        """Roll the actor forward through the model, starting from every
        posterior state in the replayed batch flattened into one big batch."""
        dynamics = self._world_model.dynamics
        state = {k: v.reshape([-1] + list(v.shape[2:])) for k, v in start.items()}
        feats, states, actions = [], [], []
        for _ in range(horizon):
            # `feat` belongs to the state the action is taken *from*, so feats,
            # states and actions all have length `horizon` and stay aligned;
            # the last entry is what the lambda-return bootstraps off.
            feat = dynamics.get_feat(state)
            action = self.actor(feat.detach()).sample()
            feats.append(feat)
            states.append(state)
            actions.append(action)
            state = dynamics.img_step(state, action)
        stacked_states = {k: torch.stack([s[k] for s in states], 0) for k in states[0]}
        return torch.stack(feats, 0), stacked_states, torch.stack(actions, 0)

    def compute_target(self, imag_feat, reward):
        continuation = self._world_model.cont_head(imag_feat).mean
        discount = self.args.discount * continuation
        value = self.critic(imag_feat).mode()
        target = lambda_return(
            reward[1:], value[:-1], discount[1:], bootstrap=value[-1],
            lambda_=self.args.discount_lambda,
        )
        weights = torch.cumprod(
            torch.cat([torch.ones_like(discount[:1]), discount[:-1]], 0), 0
        ).detach()
        return target, weights, value[:-1]

    def train_step(self, start):
        self._update_slow_target()
        horizon = self.args.imag_horizon
        imag_feat, imag_state, imag_action = self.imagine(start, horizon)
        reward = self._world_model.reward_head(imag_feat).mode()
        actor_entropy = self.actor(imag_feat).entropy()
        target, weights, base = self.compute_target(imag_feat, reward)

        policy = self.actor(imag_feat.detach())
        if self.args.reward_ema:
            offset, scale = self.reward_ema(target, self.ema_vals)
            advantage = (target - offset) / scale - (base - offset) / scale
        else:
            advantage = target - base
        if self.args.imag_gradient == "reinforce":
            actor_target = (
                policy.log_prob(imag_action)[:-1][:, :, None]
                * (target - self.critic(imag_feat[:-1]).mode()).detach()
            )
        elif self.args.imag_gradient == "dynamics":
            actor_target = advantage
        else:
            raise NotImplementedError(self.args.imag_gradient)
        actor_loss = -weights[:-1] * actor_target
        actor_loss = actor_loss - self.args.actor_entropy * actor_entropy[:-1][..., None]
        actor_loss = actor_loss.mean()

        value_dist = self.critic(imag_feat[:-1].detach())
        value_loss = -value_dist.log_prob(target.detach())
        slow_target = self.slow_critic(imag_feat[:-1].detach()).mode()
        value_loss = value_loss - value_dist.log_prob(slow_target.detach())
        value_loss = torch.mean(weights[:-1] * value_loss[:, :, None])

        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward(retain_graph=True)
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.args.ac_grad_clip)
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad(set_to_none=True)
        value_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.args.ac_grad_clip)
        self.critic_optimizer.step()

        return {
            "actor_loss": actor_loss.detach(),
            "value_loss": value_loss.detach(),
            "actor_entropy": actor_entropy.mean().detach(),
            "imag_reward": reward.mean().detach(),
            "imag_target": target.mean().detach(),
        }

    def _update_slow_target(self):
        mix = self.args.slow_target_fraction
        with torch.no_grad():
            for source, destination in zip(self.critic.parameters(), self.slow_critic.parameters()):
                destination.data = mix * source.data + (1 - mix) * destination.data


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------


class SequenceReplayBuffer:
    """Per-environment ring of transitions, sampled as contiguous windows.

    `is_first` is stored rather than derived: the RSSM resets its state exactly
    where that flag is set, so a sampled window may straddle an episode
    boundary without leaking state across it.
    """

    def __init__(self, capacity, num_envs, num_actions, obs_shape, device):
        self.rows = max(int(np.ceil(capacity / num_envs)), 2)
        self.num_envs = num_envs
        self.num_actions = num_actions
        self.device = device
        self.observations = np.zeros((self.rows, num_envs) + obs_shape, dtype=np.uint8)
        self.actions = np.zeros((self.rows, num_envs), dtype=np.int64)
        self.rewards = np.zeros((self.rows, num_envs), dtype=np.float32)
        self.is_first = np.zeros((self.rows, num_envs), dtype=np.float32)
        self.is_terminal = np.zeros((self.rows, num_envs), dtype=np.float32)
        self.steps = 0

    def add(self, observation, action, reward, is_first, is_terminal):
        row = self.steps % self.rows
        self.observations[row] = observation
        self.actions[row] = action
        self.rewards[row] = reward
        self.is_first[row] = is_first
        self.is_terminal[row] = is_terminal
        self.steps += 1

    def __len__(self):
        return min(self.steps, self.rows) * self.num_envs

    def sample(self, batch_size, length):
        available = min(self.steps, self.rows)
        if available < length + 1:
            raise RuntimeError("not enough transitions stored to sample a sequence")
        newest = self.steps - 1
        oldest = self.steps - available
        starts = np.random.randint(oldest, newest - length + 2, size=batch_size)
        envs = np.random.randint(0, self.num_envs, size=batch_size)
        rows = (starts[:, None] + np.arange(length)[None, :]) % self.rows
        env_index = envs[:, None]
        actions = torch.as_tensor(self.actions[rows, env_index], device=self.device)
        return {
            "image": torch.as_tensor(self.observations[rows, env_index], device=self.device),
            "action": F.one_hot(actions, self.num_actions).float(),
            "reward": torch.as_tensor(self.rewards[rows, env_index], device=self.device),
            "is_first": torch.as_tensor(self.is_first[rows, env_index], device=self.device),
            "cont": 1.0 - torch.as_tensor(self.is_terminal[rows, env_index], device=self.device),
        }


def make_envs(args):
    """64x64 RGB, action repeat 4, single frame, no reward clipping."""
    if args.env_backend == "envpool":
        if envpool is None:
            raise ImportError("EnvPool backend requested; install envpool")
        return envpool.make(
            args.env_id, env_type="gym", num_envs=args.num_envs,
            img_height=64, img_width=64, gray_scale=False, stack_num=1,
            frame_skip=args.action_repeat, noop_max=30,
            repeat_action_probability=0.0, episodic_life=False, reward_clip=False,
            seed=args.seed, max_episode_steps=108000 // args.action_repeat,
        )
    if args.env_backend == "gymnasium":
        def thunk(index):
            def make():
                env = gym.make(args.env_id, frameskip=1, repeat_action_probability=0.0)
                env = gym.wrappers.AtariPreprocessing(
                    env, noop_max=30, frame_skip=args.action_repeat,
                    screen_size=64, grayscale_obs=False, scale_obs=False,
                )
                env = gym.wrappers.TransformObservation(
                    env, lambda obs: np.transpose(obs, (2, 0, 1)),
                    gym.spaces.Box(0, 255, (3, 64, 64), np.uint8),
                )
                env = gym.wrappers.RecordEpisodeStatistics(env)
                env.action_space.seed(args.seed + index)
                return env

            return make

        return gym.vector.SyncVectorEnv([thunk(i) for i in range(args.num_envs)])
    raise ValueError(f"unsupported environment backend: {args.env_backend}")


if __name__ == "__main__":
    process_start = time.perf_counter()
    args = tyro.cli(Args)
    if args.num_envs < 1:
        raise ValueError("num_envs must be positive")
    if args.batch_length < 2:
        raise ValueError("batch_length must be at least 2")
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(project=args.wandb_project_name, entity=args.wandb_entity,
                   sync_tensorboard=True, config=vars(args), name=run_name, save_code=True)
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

    envs = make_envs(args)
    action_space = getattr(envs, "single_action_space", None) or envs.action_space
    num_actions = action_space.n

    world_model = WorldModel(num_actions, args, device)
    behavior = ImagBehavior(num_actions, world_model, args, device)
    replay = SequenceReplayBuffer(args.buffer_size, args.num_envs, num_actions, (3, 64, 64), device)

    # `total_timesteps` counts frames; the agent acts once per action_repeat.
    policy_steps = args.total_timesteps // args.action_repeat
    if args.benchmark:
        policy_steps = args.benchmark_warmup_iterations + args.benchmark_measure_iterations
    train_every = (args.batch_size * args.batch_length) / args.train_ratio

    reset_result = envs.reset()
    observation = reset_result[0] if isinstance(reset_result, tuple) else reset_result
    observation = np.asarray(observation)
    episode_stats = EpisodeStats(args.solve_window, args.solve_reward)
    latent = None
    action = None
    is_first = np.ones(args.num_envs, dtype=np.float32)
    global_step = 0
    updates = 0
    update_budget = 0.0
    start_time = time.time()
    metrics = {}
    benchmark_start = benchmark_start_step = benchmark_start_updates = None

    for step in range(policy_steps):
        if args.benchmark and step == args.benchmark_warmup_iterations:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            benchmark_start = time.perf_counter()
            benchmark_start_step = global_step
            benchmark_start_updates = updates
        if args.max_training_seconds and time.time() - start_time >= args.max_training_seconds:
            break

        # ACT. Before `prefill` the policy is uniform random, as upstream does.
        if global_step < args.prefill:
            action_index = np.random.randint(0, num_actions, size=args.num_envs)
            latent = None
        else:
            with torch.no_grad():
                image = torch.as_tensor(observation, device=device).float().unsqueeze(1) / 255.0
                embed = world_model.encoder(image).squeeze(1)
                previous = action if action is not None else torch.zeros(
                    args.num_envs, num_actions, device=device
                )
                latent, _ = world_model.dynamics.obs_step(
                    latent, previous, embed, torch.as_tensor(is_first, device=device)
                )
                feat = world_model.dynamics.get_feat(latent)
                action = behavior.actor(feat).sample()
                latent = {k: v.detach() for k, v in latent.items()}
            action_index = action.argmax(-1).cpu().numpy()

        step_result = envs.step(action_index)
        if len(step_result) == 5:
            next_observation, reward, terminated, truncated, infos = step_result
        else:
            next_observation, reward, done, infos = step_result
            terminated, truncated = done, np.zeros_like(done)
        next_observation = np.asarray(next_observation)
        done = np.logical_or(terminated, truncated)

        replay.add(observation, action_index, np.asarray(reward, dtype=np.float32),
                   is_first, np.asarray(terminated, dtype=np.float32))
        if global_step >= args.prefill:
            action = F.one_hot(torch.as_tensor(action_index, device=device), num_actions).float()
        observation = next_observation
        is_first = done.astype(np.float32)
        if done.any():
            latent = None
            action = None
        global_step += args.num_envs

        if not args.benchmark:
            episode_stats.update(infos, global_step * args.action_repeat, writer)

        # TRAIN.
        if global_step >= args.prefill and len(replay) > args.batch_length * args.batch_size:
            update_budget += args.num_envs / train_every
            for _ in range(int(update_budget)):
                batch = replay.sample(args.batch_size, args.batch_length)
                post, model_metrics = world_model.train_step(batch)
                behavior_metrics = behavior.train_step(post)
                metrics = {**model_metrics, **behavior_metrics}
                updates += 1
            update_budget -= int(update_budget)

        if writer is not None and step % 500 == 0 and metrics:
            frames = global_step * args.action_repeat
            for name, value in metrics.items():
                writer.add_scalar(f"losses/{name}", float(value), frames)
            sps = int(frames / (time.time() - start_time))
            writer.add_scalar("charts/SPS", sps, frames)
            writer.add_scalar("charts/updates", updates, frames)
            print(f"frames={frames} updates={updates} SPS={sps}")

    if args.benchmark:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        benchmark_end = time.perf_counter()
        measured_steps = (global_step - benchmark_start_step) * args.action_repeat
        measured_seconds = benchmark_end - benchmark_start
        result = {
            "algorithm": "dreamerv3",
            "backend": args.env_backend,
            "batch_size": args.batch_size,
            "batch_length": args.batch_length,
            "benchmark": "full_training_loop",
            "compile": False,
            "env_id": args.env_id,
            "imag_horizon": args.imag_horizon,
            "measure_iterations": args.benchmark_measure_iterations,
            "measured_seconds": measured_seconds,
            "measured_steps": measured_steps,
            "num_envs": args.num_envs,
            "peak_cuda_memory_mb": (
                torch.cuda.max_memory_allocated(device) / (1024**2) if device.type == "cuda" else 0.0
            ),
            "process_seconds": benchmark_end - process_start,
            "schema_version": 1,
            "sps": measured_steps / measured_seconds,
            "updates": updates - benchmark_start_updates,
            "ups": (updates - benchmark_start_updates) / measured_seconds,
            "warmup_iterations": args.benchmark_warmup_iterations,
        }
        print(f"BENCHMARK_RESULT {json.dumps(result, sort_keys=True)}", flush=True)
    else:
        elapsed = time.time() - start_time
        print("frames:", global_step * args.action_repeat)
        print("updates:", updates)
        print("SPS:", int(global_step * args.action_repeat / elapsed))
        episode_stats.print_summary()

    envs.close()
    if writer is not None:
        writer.close()
