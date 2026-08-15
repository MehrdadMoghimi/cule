"""Reference subset of BTR's official `networks.py`, for equivalence testing.

Transcribed from https://github.com/VIPTankz/BTR (commit of the ICML 2025
camera-ready), file `networks.py`, classes `FactorizedNoisyLinear`,
`ImpalaCNNResidual`, `ImpalaCNNBlock`, `Dueling` and `ImpalaCNNLargeIQN`.

    MIT License
    Copyright (c) 2024 VIPTankz

Only the code paths reached by BTR's own configuration are kept (impala arch,
spectral norm on, noisy on, dueling on, maxpool on); the unreachable branches,
device juggling, checkpoint helpers and commented-out debug code are dropped.
The retained lines are otherwise unmodified, so a numerical diff against
`cleanrl/btr_atari.py` is a diff against upstream.
"""

import math
from math import sqrt

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch import nn as nn
from torch.nn import init


class FactorizedNoisyLinear(nn.Module):
    """The factorized Gaussian noise layer for noisy-nets dqn."""

    def __init__(self, in_features: int, out_features: int, sigma_0=0.5) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma_0 = sigma_0

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))

        self.reset_parameters()
        self.reset_noise()
        self.disable_noise()

    @torch.no_grad()
    def reset_parameters(self) -> None:
        scale = 1 / sqrt(self.in_features)
        init.uniform_(self.weight_mu, -scale, scale)
        init.uniform_(self.bias_mu, -scale, scale)
        init.constant_(self.weight_sigma, self.sigma_0 * scale)
        init.constant_(self.bias_sigma, self.sigma_0 * scale)

    @torch.no_grad()
    def _get_noise(self, size: int) -> Tensor:
        noise = torch.randn(size, device=self.weight_mu.device)
        return noise.sign().mul_(noise.abs().sqrt_())

    @torch.no_grad()
    def reset_noise(self) -> None:
        epsilon_in = self._get_noise(self.in_features)
        epsilon_out = self._get_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.outer(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    @torch.no_grad()
    def disable_noise(self) -> None:
        self.weight_epsilon[:] = 0
        self.bias_epsilon[:] = 0

    def forward(self, input: Tensor) -> Tensor:
        return F.linear(input,
                        self.weight_mu + self.weight_sigma * self.weight_epsilon,
                        self.bias_mu + self.bias_sigma * self.bias_epsilon)


class ImpalaCNNResidual(nn.Module):
    """Simple residual block used in the large IMPALA CNN."""

    def __init__(self, depth, norm_func, activation=nn.ReLU):
        super().__init__()
        self.activation = activation()
        self.conv_0 = norm_func(nn.Conv2d(in_channels=depth, out_channels=depth, kernel_size=3, stride=1, padding=1))
        self.conv_1 = norm_func(nn.Conv2d(in_channels=depth, out_channels=depth, kernel_size=3, stride=1, padding=1))

    def forward(self, x):
        x_ = self.conv_0(self.activation(x))
        x_ = self.conv_1(self.activation(x_))
        return x + x_


class ImpalaCNNBlock(nn.Module):
    """Three of these blocks are used in the large IMPALA CNN."""

    def __init__(self, depth_in, depth_out, norm_func, activation=nn.ReLU):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=depth_in, out_channels=depth_out, kernel_size=3, stride=1, padding=1)
        self.max_pool = nn.MaxPool2d(3, 2, padding=1)
        self.residual_0 = ImpalaCNNResidual(depth_out, norm_func=norm_func, activation=activation)
        self.residual_1 = ImpalaCNNResidual(depth_out, norm_func=norm_func, activation=activation)

    def forward(self, x):
        x = self.conv(x)
        x = self.max_pool(x)
        x = self.residual_0(x)
        x = self.residual_1(x)
        return x


class Dueling(nn.Module):
    """The dueling branch used in all nets that use dueling-dqn."""

    def __init__(self, value_branch, advantage_branch):
        super().__init__()
        self.flatten = nn.Flatten()
        self.value_branch = value_branch
        self.advantage_branch = advantage_branch

    def forward(self, x, advantages_only=False):
        x = self.flatten(x)
        advantages = self.advantage_branch(x)
        if advantages_only:
            return advantages

        value = self.value_branch(x)
        return value + (advantages - torch.mean(advantages, dim=1, keepdim=True))


class ImpalaCNNLargeIQN(nn.Module):
    """Implementation of the large variant of the IMPALA CNN, with IQN."""

    def __init__(self, in_depth, actions, model_size=2, spectral=True, device='cpu',
                 noisy=True, maxpool=True, num_tau=8, maxpool_size=6, dueling=True,
                 linear_size=512, ncos=64):
        super().__init__()

        self.model_size = model_size
        self.actions = actions
        self.device = device
        self.noisy = noisy
        self.maxpool = maxpool
        self.in_depth = in_depth
        self.linear_size = linear_size
        self.num_tau = num_tau
        self.maxpool_size = maxpool_size

        activation = nn.ReLU
        conv_activation = nn.ReLU

        self.n_cos = ncos
        self.pis = torch.FloatTensor([np.pi * i for i in range(self.n_cos)]).view(1, 1, self.n_cos).to(device)

        linear_layer = FactorizedNoisyLinear if noisy else nn.Linear

        def identity(p):
            return p

        norm_func = torch.nn.utils.parametrizations.spectral_norm if spectral else identity

        self.conv = nn.Sequential(
            ImpalaCNNBlock(in_depth, int(16 * model_size), norm_func=norm_func, activation=conv_activation),
            ImpalaCNNBlock(int(16 * model_size), int(32 * model_size), norm_func=norm_func, activation=conv_activation),
            ImpalaCNNBlock(int(32 * model_size), int(32 * model_size), norm_func=norm_func, activation=conv_activation),
            nn.ReLU()
        )

        self.pool = torch.nn.AdaptiveMaxPool2d((self.maxpool_size, self.maxpool_size))
        self.conv_out_size = int(1152 * model_size)

        self.cos_embedding = nn.Linear(self.n_cos, self.conv_out_size)

        self.dueling = Dueling(
            nn.Sequential(linear_layer(self.conv_out_size, self.linear_size),
                          activation(),
                          linear_layer(self.linear_size, 1)),
            nn.Sequential(linear_layer(self.conv_out_size, self.linear_size),
                          activation(),
                          linear_layer(self.linear_size, actions))
        )

        self.to(device)

    def forward(self, inputt, advantages_only=False, taus=None):
        """Returns quantiles of shape (batch, num_tau, actions) and taus.

        `taus` is an added test hook: upstream draws them internally with
        `torch.rand`, which cannot be shared across two implementations.  When
        supplied it must have shape (batch, num_tau, 1); the arithmetic is
        otherwise identical to upstream.
        """
        batch_size = inputt.size()[0]
        inputt = inputt.float() / 255

        x = self.conv(inputt)
        if self.maxpool:
            x = self.pool(x)

        x = x.view(batch_size, -1)

        if taus is None:
            taus = torch.rand(batch_size, self.num_tau).to(self.device).unsqueeze(-1)
        cos = torch.cos(taus * self.pis)

        cos = cos.view(batch_size * self.num_tau, self.n_cos)
        cos_x = torch.relu(self.cos_embedding(cos)).view(batch_size, self.num_tau, self.conv_out_size)

        x = (x.unsqueeze(1) * cos_x).view(batch_size * self.num_tau, self.conv_out_size)

        out = self.dueling(x, advantages_only=advantages_only)

        return out.view(batch_size, self.num_tau, self.actions), taus

    def qvals(self, inputs, advantages_only=False, taus=None):
        quantiles, _ = self.forward(inputs, advantages_only, taus=taus)
        return quantiles.mean(dim=1)


def munchausen_iqn_loss(net, tgt_net, states, actions, rewards, next_states, dones, weights,
                        gamma, n, entropy_tau, alpha, lo, num_tau,
                        taus_online, taus_target, taus_policy_cur):
    """Upstream `Agent.learn_call`, branch `self.iqn and self.munchausen`.

    Transcribed from https://github.com/VIPTankz/BTR `Agent.py`.  The only edits
    are (a) hyperparameters and networks are passed in rather than read off
    `self`, and (b) the tau draws are passed in so both implementations can be
    given identical randomness.  Returns (loss, priority).
    """
    batch_size = states.shape[0]

    with torch.no_grad():
        Q_targets_next, _ = tgt_net(next_states, taus=taus_target)

        # (batch, num_tau, actions) -- note upstream reuses this single draw for
        # the target policy rather than sampling fresh taus.
        q_t_n = Q_targets_next.mean(dim=1)

        actions = actions.unsqueeze(1)
        rewards = rewards.unsqueeze(1)
        dones = dones.unsqueeze(1)
        weights = weights.unsqueeze(1)

        logsum = torch.logsumexp(
            (q_t_n - q_t_n.max(1)[0].unsqueeze(-1)) / entropy_tau, 1).unsqueeze(-1)
        tau_log_pi_next = (q_t_n - q_t_n.max(1)[0].unsqueeze(-1) - entropy_tau * logsum).unsqueeze(1)

        pi_target = F.softmax(q_t_n / entropy_tau, dim=1).unsqueeze(1)

        Q_target = (gamma ** n * (
                pi_target * (Q_targets_next - tau_log_pi_next) * (~dones.unsqueeze(-1))).sum(2)).unsqueeze(1)

        q_k_target = net.qvals(states, taus=taus_policy_cur)
        v_k_target = q_k_target.max(1)[0].unsqueeze(-1)
        tau_log_pik = q_k_target - v_k_target - entropy_tau * torch.logsumexp(
            (q_k_target - v_k_target) / entropy_tau, 1).unsqueeze(-1)

        munchausen_addon = tau_log_pik.gather(1, actions)

        munchausen_reward = (
                rewards + alpha * torch.clamp(munchausen_addon, min=lo, max=0)).unsqueeze(-1)
        Q_targets = munchausen_reward + Q_target

    q_k, taus = net(states, taus=taus_online)
    Q_expected = q_k.gather(2, actions.unsqueeze(-1).expand(batch_size, num_tau, 1))

    td_error = Q_targets - Q_expected
    loss_v = torch.abs(td_error).sum(dim=1).mean(dim=1).data
    huber_l = calculate_huber_loss(td_error, 1.0, num_tau)
    quantil_l = abs(taus - (td_error.detach() < 0).float()) * huber_l / 1.0

    loss = quantil_l.sum(dim=1).mean(dim=1, keepdim=True)
    loss = loss * weights
    loss = loss.mean()
    return loss, loss_v


def calculate_huber_loss(td_errors, k=1.0, taus=8):
    """Upstream `Agent.py::calculate_huber_loss`."""
    loss = torch.where(td_errors.abs() <= k, 0.5 * td_errors.pow(2), k * (td_errors.abs() - 0.5 * k))
    return loss
