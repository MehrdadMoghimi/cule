"""DPO: equivalence against luchris429/purejaxrl's reference drift function.

purejaxrl is JAX, so the reference is transcribed rather than executed — but it
is transcribed into NumPy from the exact lines
(`purejaxrl/dpo_continuous_action.py:208-220`) and driven with the paper's
alpha=2.0, beta=0.6.

The rest of the file checks the two properties Mirror Learning actually
requires of a drift function, which is what makes DPO *sound* rather than merely
different: non-negativity everywhere, and a zero value with zero gradient at
ratio = 1.
"""

import numpy as np
import pytest
import torch

from conftest import DiscreteEnvStub, load_trainer

TRAINER = 'dpo_atari'
ALPHA, BETA = 2.0, 0.6


@pytest.fixture(scope='module')
def dpo():
    return load_trainer(TRAINER)


def reference_drift(ratio, log_diff, gae, alpha, beta):
    """NumPy transcription of `dpo_continuous_action.py:213-219`."""
    relu = lambda x: np.maximum(x, 0.0)
    is_pos = (gae >= 0.0).astype(np.float64)
    r1 = ratio - 1.0
    drift1 = relu(r1 * gae - alpha * np.tanh(r1 * gae / alpha))
    drift2 = relu(log_diff * gae - beta * np.tanh(log_diff * gae / beta))
    return drift1 * is_pos + drift2 * (1 - is_pos)


@pytest.mark.parametrize('seed', range(8))
def test_drift_matches_purejaxrl(dpo, seed):
    generator = torch.Generator().manual_seed(seed)
    logratio = torch.randn(4096, generator=generator, dtype=torch.float64) * 0.5
    ratio = logratio.exp()
    advantages = torch.randn(4096, generator=generator, dtype=torch.float64)

    got = dpo.dpo_drift(ratio, logratio, advantages, ALPHA, BETA)
    want = reference_drift(ratio.numpy(), logratio.numpy(), advantages.numpy(), ALPHA, BETA)
    assert np.allclose(got.numpy(), want, rtol=0, atol=1e-14)


@pytest.mark.parametrize('alpha,beta', [(2.0, 0.6), (1.0, 1.0), (0.1, 5.0)])
def test_policy_loss_matches_purejaxrl(dpo, alpha, beta):
    torch.manual_seed(0)
    logratio = torch.randn(2048, dtype=torch.float64) * 0.8
    ratio = logratio.exp()
    advantages = torch.randn(2048, dtype=torch.float64)

    got = dpo.dpo_policy_loss(ratio, logratio, advantages, alpha, beta)
    drift = reference_drift(ratio.numpy(), logratio.numpy(), advantages.numpy(), alpha, beta)
    want = -np.mean(ratio.numpy() * advantages.numpy() - drift)
    assert np.isclose(got.item(), want, rtol=0, atol=1e-13)


def test_drift_is_non_negative(dpo):
    """Mirror Learning requirement 1. `relu` makes it structural, but the *sign*
    of the pre-relu expression is what the theory is about, so sweep widely."""
    logratio = torch.linspace(-4, 4, 400, dtype=torch.float64)
    for advantage in (-10.0, -1.0, -0.01, 0.0, 0.01, 1.0, 10.0):
        advantages = torch.full_like(logratio, advantage)
        drift = dpo.dpo_drift(logratio.exp(), logratio, advantages, ALPHA, BETA)
        assert (drift >= 0).all(), advantage


def test_drift_is_zero_with_zero_gradient_at_ratio_one(dpo):
    """Mirror Learning requirement 2: at `pi = pi_old` the penalty must vanish
    to first order, or the objective is not a policy-improvement operator."""
    logratio = torch.zeros(16, dtype=torch.float64, requires_grad=True)
    advantages = torch.linspace(-3, 3, 16, dtype=torch.float64)
    drift = dpo.dpo_drift(logratio.exp(), logratio, advantages, ALPHA, BETA)

    assert torch.allclose(drift, torch.zeros_like(drift), atol=1e-15)
    drift.sum().backward()
    assert torch.allclose(logratio.grad, torch.zeros_like(logratio.grad), atol=1e-12)


def test_at_ratio_one_the_gradient_is_the_vanilla_policy_gradient(dpo):
    """With the drift silent, `-(ratio * A)` differentiates to `-A`."""
    logratio = torch.zeros(32, dtype=torch.float64, requires_grad=True)
    advantages = torch.randn(32, dtype=torch.float64)
    loss = dpo.dpo_policy_loss(logratio.exp(), logratio, advantages, ALPHA, BETA)
    loss.backward()
    assert torch.allclose(logratio.grad, -advantages / 32, rtol=0, atol=1e-13)


def test_each_branch_penalises_only_one_direction(dpo):
    """`relu(u - c tanh(u/c))` is zero for `u <= 0`, so each branch is one-sided.

    Positive advantage: `u = (ratio - 1) A`, active only for `ratio > 1`.
    Negative advantage: `u = log(ratio) A`, active only for `ratio < 1`.

    So DPO penalises raising the probability of a good action too far, *and*
    penalises lowering the probability of a bad action too far — and does
    nothing in the other two quadrants. That second penalty is the "rollback"
    optimism the paper reports: an action that scored badly is not driven
    towards zero probability, it is held back from being abandoned.
    """
    up = torch.linspace(0.05, 1.5, 40, dtype=torch.float64)
    down = -up
    good = torch.ones(40, dtype=torch.float64)
    bad = -torch.ones(40, dtype=torch.float64)
    zeros = torch.zeros(40, dtype=torch.float64)

    assert (dpo.dpo_drift(up.exp(), up, good, ALPHA, BETA) > 0).all()
    assert torch.allclose(dpo.dpo_drift(down.exp(), down, good, ALPHA, BETA), zeros, atol=1e-15)
    assert (dpo.dpo_drift(down.exp(), down, bad, ALPHA, BETA) > 0).all()
    assert torch.allclose(dpo.dpo_drift(up.exp(), up, bad, ALPHA, BETA), zeros, atol=1e-15)


def test_the_two_branches_are_genuinely_different(dpo):
    """The asymmetry is the finding: `ratio - 1` on one side, `log ratio` on the
    other. Compared where the negative branch is actually active."""
    logratio = torch.linspace(-1.5, -0.05, 50, dtype=torch.float64)
    ratio = logratio.exp()
    negative = torch.full_like(logratio, -1.0)

    actual = dpo.dpo_drift(ratio, logratio, negative, ALPHA, BETA)
    # What the *positive* branch's functional form would have given here.
    r1 = ratio - 1.0
    as_if_positive = torch.relu(r1 * negative - BETA * torch.tanh(r1 * negative / BETA))
    assert (actual > 0).all()
    assert not torch.allclose(actual, as_if_positive, rtol=1e-3, atol=1e-6)


def test_drift_grows_with_distance_from_the_old_policy(dpo):
    """Smoothness with teeth: unlike the clip, the penalty keeps increasing —
    each branch in the direction it actually polices."""
    for advantage, span in ((1.0, (0.0, 2.0)), (-1.0, (0.0, -2.0))):
        logratio = torch.linspace(span[0], span[1], 60, dtype=torch.float64)
        advantages = torch.full_like(logratio, advantage)
        drift = dpo.dpo_drift(logratio.exp(), logratio, advantages, ALPHA, BETA)
        increments = drift[1:] - drift[:-1]
        assert (increments >= -1e-12).all(), advantage
        assert drift[-1] > drift[0] + 1e-6, advantage


def test_unlike_ppo_the_gradient_does_not_switch_off(dpo):
    """PPO's clipped surrogate has exactly zero gradient once outside the trust
    region for a favourable advantage; DPO's does not. This is the practical
    difference the paper is trading on."""
    logratio = torch.tensor([1.5], dtype=torch.float64, requires_grad=True)
    advantages = torch.tensor([1.0], dtype=torch.float64)

    dpo_loss = dpo.dpo_policy_loss(logratio.exp(), logratio, advantages, ALPHA, BETA)
    dpo_loss.backward()
    dpo_grad = logratio.grad.clone()

    ppo_logratio = torch.tensor([1.5], dtype=torch.float64, requires_grad=True)
    ratio = ppo_logratio.exp()
    ppo_loss = torch.max(-advantages * ratio,
                         -advantages * torch.clamp(ratio, 0.9, 1.1)).mean()
    ppo_loss.backward()

    assert ppo_logratio.grad.abs().item() == 0.0
    assert dpo_grad.abs().item() > 1e-6


def test_alpha_and_beta_control_their_own_branches_only(dpo):
    """Each branch is measured where it is active, per the previous test."""
    up = torch.full((8,), 0.7, dtype=torch.float64)
    down = torch.full((8,), -0.7, dtype=torch.float64)
    positive = torch.ones(8, dtype=torch.float64)
    negative = -torch.ones(8, dtype=torch.float64)

    base_pos = dpo.dpo_drift(up.exp(), up, positive, ALPHA, BETA)
    base_neg = dpo.dpo_drift(down.exp(), down, negative, ALPHA, BETA)
    assert (base_pos > 0).all() and (base_neg > 0).all()

    assert not torch.allclose(dpo.dpo_drift(up.exp(), up, positive, 8.0, BETA), base_pos)
    assert torch.allclose(dpo.dpo_drift(down.exp(), down, negative, 8.0, BETA), base_neg)
    assert torch.allclose(dpo.dpo_drift(up.exp(), up, positive, ALPHA, 3.0), base_pos)
    assert not torch.allclose(dpo.dpo_drift(down.exp(), down, negative, ALPHA, 3.0), base_neg)


def test_zero_advantage_is_treated_as_positive(dpo):
    """`gae >= 0.0` in the reference, not `> 0.0`. At A = 0 both branches give
    zero anyway, so this only fixes the boundary convention."""
    logratio = torch.full((4,), 0.3, dtype=torch.float64)
    drift = dpo.dpo_drift(logratio.exp(), logratio, torch.zeros(4, dtype=torch.float64), ALPHA, BETA)
    assert torch.allclose(drift, torch.zeros_like(drift), atol=1e-15)


def test_agent_is_unchanged_from_ppo(dpo):
    """DPO changes one loss line and nothing else."""
    ppo = load_trainer('ppo_atari')
    envs = DiscreteEnvStub(6)
    torch.manual_seed(0)
    ours = dpo.Agent(envs)
    torch.manual_seed(0)
    theirs = ppo.Agent(envs)
    for key, value in ours.state_dict().items():
        assert torch.equal(value, theirs.state_dict()[key]), key
