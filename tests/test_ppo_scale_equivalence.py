"""PPO-scale: the reduction to PPO, the EWMA algebra, and the scaling recipe.

The paper (Beukman et al., RLJ/RLC 2026) contributes an analysis rather than a
loss, so most of what can be pinned here is algebra. Three groups:

* the *reduction*, which is load-bearing in the same way it is for
  `outer_ppo_atari`: with no proximal policy the surrogate must be
  `ppo_atari.py`'s, bit for bit. A paper about escaping plateaus is only
  measurable against a baseline it can reproduce exactly.
* the PPO-EWMA machinery -- the centre-of-mass conversion, the debiased
  recursion, and the identity that makes the decoupled loss a *decomposition* of
  PPO rather than a different objective.
* the scaling recipe and DDR, which are arithmetic, and whose failure modes are
  silent (a minibatch count that stops growing still trains, just worse).

The numerical diff against openai/ppo-ewma lives in
`tests/crosscheck/check_ppo_scale.py`; this file states expectations
independently so it runs without the checkout.
"""

import math

import numpy as np
import pytest
import torch
import torch.nn as nn

from conftest import DiscreteEnvStub, load_trainer

TRAINER = 'ppo_scale_atari'


@pytest.fixture(scope='module')
def ppo_scale():
    return load_trainer(TRAINER)


def tiny_agent(ppo_scale, seed=0, n_actions=6):
    torch.manual_seed(seed)
    return ppo_scale.Agent(DiscreteEnvStub(n_actions=n_actions))


def random_batch(size=256, n_actions=6, seed=0):
    generator = torch.Generator().manual_seed(seed)
    logits = torch.randn(size, n_actions, generator=generator, dtype=torch.float64)
    actions = torch.randint(0, n_actions, (size,), generator=generator)
    behaviour = torch.distributions.Categorical(
        logits=logits + 0.7 * torch.randn(size, n_actions, generator=generator, dtype=torch.float64))
    newlogp = torch.distributions.Categorical(logits=logits).log_prob(actions)
    advantages = torch.randn(size, generator=generator, dtype=torch.float64)
    return advantages, newlogp, behaviour.log_prob(actions).detach()


# ---------------------------------------------------------------------------
# the reduction to PPO
# ---------------------------------------------------------------------------

def ppo_reference_surrogate(advantages, newlogp, behaviour_logp, clip_coef):
    """`ppo_atari.py`'s policy loss, transcribed line for line."""
    logratio = newlogp - behaviour_logp
    ratio = logratio.exp()
    pg_loss1 = -advantages * ratio
    pg_loss2 = -advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
    return torch.max(pg_loss1, pg_loss2).mean()


@pytest.mark.parametrize('clip_coef', [0.1, 0.2, 0.3])
def test_no_proximal_policy_is_ppo_bit_exactly(ppo_scale, clip_coef):
    advantages, newlogp, behaviour_logp = random_batch()
    ours, _ = ppo_scale.policy_surrogate(
        advantages, newlogp, behaviour_logp, None, clip_coef, imp_samp_max=100.0)
    theirs = ppo_reference_surrogate(advantages, newlogp, behaviour_logp, clip_coef)
    assert torch.equal(ours, theirs)


def test_reduction_holds_over_many_random_batches(ppo_scale):
    """One batch could agree by luck; the clip branch must be hit both ways."""
    for seed in range(25):
        advantages, newlogp, behaviour_logp = random_batch(size=128, seed=seed)
        ours, _ = ppo_scale.policy_surrogate(
            advantages, newlogp, behaviour_logp, None, 0.1, imp_samp_max=100.0)
        assert torch.equal(ours, ppo_reference_surrogate(advantages, newlogp, behaviour_logp, 0.1))


def test_imp_samp_max_is_inert_without_a_proximal_policy(ppo_scale):
    """The IS clip belongs to the decoupled path; PPO's path must ignore it."""
    advantages, newlogp, behaviour_logp = random_batch()
    losses = [
        ppo_scale.policy_surrogate(advantages, newlogp, behaviour_logp, None, 0.1, imp_samp_max=m)[0]
        for m in (0.0, 1.0, 100.0, 1e9)
    ]
    assert all(torch.equal(loss, losses[0]) for loss in losses)


def test_reduction_gradient_is_bit_exact(ppo_scale):
    """Agreement on the loss is not agreement on what trains."""
    generator = torch.Generator().manual_seed(3)
    logits = torch.randn(64, 6, generator=generator, dtype=torch.float64, requires_grad=True)
    actions = torch.randint(0, 6, (64,), generator=generator)
    advantages = torch.randn(64, generator=generator, dtype=torch.float64)
    behaviour_logp = torch.randn(64, generator=generator, dtype=torch.float64) - 2.0

    def grad_of(loss_fn):
        newlogp = torch.distributions.Categorical(logits=logits).log_prob(actions)
        return torch.autograd.grad(loss_fn(newlogp), logits)[0]

    ours = grad_of(lambda lp: ppo_scale.policy_surrogate(
        advantages, lp, behaviour_logp, None, 0.1, 100.0)[0])
    theirs = grad_of(lambda lp: ppo_reference_surrogate(advantages, lp, behaviour_logp, 0.1))
    assert torch.equal(ours, theirs)


# ---------------------------------------------------------------------------
# centre of mass <-> decay
# ---------------------------------------------------------------------------

def test_com_matches_the_papers_worked_example(ppo_scale):
    """Section 2.2: a COM of 1024 is beta = 1024/1025 ~ 0.99902."""
    assert ppo_scale.com_to_decay(1024.0) == 1024.0 / 1025.0
    assert ppo_scale.com_to_decay(1024.0) == pytest.approx(0.99902, abs=1e-5)


@pytest.mark.parametrize('com', [0.0, 1.0, 8.0, 256.0, 1024.0, 4096.0, 8192.0])
def test_com_decay_round_trip(ppo_scale, com):
    assert ppo_scale.decay_to_com(ppo_scale.com_to_decay(com)) == pytest.approx(com, rel=1e-9)


def test_com_zero_is_no_lag(ppo_scale):
    assert ppo_scale.com_to_decay(0.0) == 0.0


def test_negative_com_is_rejected(ppo_scale):
    with pytest.raises(ValueError):
        ppo_scale.com_to_decay(-1.0)


def test_com_is_monotonic_in_decay(ppo_scale):
    """Larger COM means an older, more strongly regularising reference."""
    decays = [ppo_scale.com_to_decay(c) for c in (0.0, 8.0, 256.0, 4096.0)]
    assert decays == sorted(decays)
    assert all(0.0 <= d < 1.0 for d in decays)


# ---------------------------------------------------------------------------
# the debiased EWMA
# ---------------------------------------------------------------------------

def test_proximal_starts_equal_to_the_policy(ppo_scale):
    agent = tiny_agent(ppo_scale)
    proximal = ppo_scale.ProximalPolicy(agent, ppo_scale.com_to_decay(256.0))
    for ours, theirs in zip(proximal.model.parameters(), agent.parameters()):
        assert torch.equal(ours, theirs)


def test_proximal_takes_no_gradient(ppo_scale):
    """The reference must never be trained towards; it only follows."""
    agent = tiny_agent(ppo_scale)
    proximal = ppo_scale.ProximalPolicy(agent, ppo_scale.com_to_decay(256.0))
    assert all(not p.requires_grad for p in proximal.model.parameters())

    observations = torch.randint(0, 255, (4, 4, 84, 84), dtype=torch.uint8).float()
    logp = proximal.log_prob(observations, torch.zeros(4, dtype=torch.long))
    assert not logp.requires_grad


def test_first_update_is_the_mean_of_the_two_weights(ppo_scale):
    """Debiasing means an exact average from step one, not a beta-weighted one."""
    agent = tiny_agent(ppo_scale)
    start = [p.detach().clone() for p in agent.parameters()]
    proximal = ppo_scale.ProximalPolicy(agent, decay=1.0 - 1e-12)  # COM ~ 1e12
    with torch.no_grad():
        for parameter in agent.parameters():
            parameter.add_(1.0)
    proximal.update(agent)
    # Naive at this decay would still be sitting on `start`; debiased is halfway.
    for ewma, original in zip(proximal.model.parameters(), start):
        assert torch.allclose(ewma, original + 0.5, atol=1e-6)


def test_total_weight_converges_to_the_geometric_sum(ppo_scale):
    """w -> 1/(1 - decay), which is where the debiased update becomes the naive one."""
    agent = tiny_agent(ppo_scale)
    decay = ppo_scale.com_to_decay(8.0)
    proximal = ppo_scale.ProximalPolicy(agent, decay)
    for _ in range(500):
        proximal.update(agent)
    assert proximal.total_weight == pytest.approx(1.0 / (1.0 - decay), rel=1e-9)


def test_debiased_recursion_approaches_the_naive_one(ppo_scale):
    """The two differ only in the transient, which is what the correction buys."""
    agent = tiny_agent(ppo_scale)
    decay = ppo_scale.com_to_decay(8.0)
    proximal = ppo_scale.ProximalPolicy(agent, decay)
    naive = [p.detach().clone() for p in agent.parameters()]

    gaps = []
    generator = torch.Generator().manual_seed(7)
    for step in range(400):
        with torch.no_grad():
            for parameter in agent.parameters():
                parameter.add_(0.01 * torch.randn(parameter.shape, generator=generator))
        proximal.update(agent)
        naive = [decay * n + (1 - decay) * p.detach() for n, p in zip(naive, agent.parameters())]
        if step in (2, 399):
            gaps.append(max(float((a - b).abs().max())
                            for a, b in zip(proximal.model.parameters(), naive)))
    early, late = gaps
    assert early > 0, 'the correction should bite early'
    assert late < early / 10, f'the two should converge: {early:.3e} -> {late:.3e}'


def test_com_zero_makes_the_proximal_policy_the_current_one(ppo_scale):
    agent = tiny_agent(ppo_scale)
    proximal = ppo_scale.ProximalPolicy(agent, ppo_scale.com_to_decay(0.0))
    with torch.no_grad():
        for parameter in agent.parameters():
            parameter.add_(0.5)
    proximal.update(agent)
    for ours, theirs in zip(proximal.model.parameters(), agent.parameters()):
        assert torch.equal(ours, theirs)


def test_larger_com_lags_further_behind(ppo_scale):
    """The mechanism the paper turns into a knob: age of the reference."""
    distances = []
    for com in (8.0, 256.0, 4096.0):
        agent = tiny_agent(ppo_scale)
        proximal = ppo_scale.ProximalPolicy(agent, ppo_scale.com_to_decay(com))
        for _ in range(64):
            with torch.no_grad():
                for parameter in agent.parameters():
                    parameter.add_(0.01)
            proximal.update(agent)
        distances.append(max(float((a - b.detach()).abs().max())
                             for a, b in zip(proximal.model.parameters(), agent.parameters())))
    assert distances == sorted(distances), f'lag should grow with COM: {distances}'


def test_reset_snaps_the_reference_onto_the_policy(ppo_scale):
    agent = tiny_agent(ppo_scale)
    proximal = ppo_scale.ProximalPolicy(agent, ppo_scale.com_to_decay(4096.0))
    with torch.no_grad():
        for parameter in agent.parameters():
            parameter.add_(1.0)
    proximal.reset(agent)
    for ours, theirs in zip(proximal.model.parameters(), agent.parameters()):
        assert torch.equal(ours, theirs)


def test_decay_of_one_is_rejected(ppo_scale):
    """decay = 1 never updates, so the reference would freeze at initialisation."""
    agent = tiny_agent(ppo_scale)
    with pytest.raises(ValueError):
        ppo_scale.ProximalPolicy(agent, 1.0)


# ---------------------------------------------------------------------------
# the decoupled surrogate
# ---------------------------------------------------------------------------

def test_unclipped_decoupled_loss_is_the_plain_is_ratio(ppo_scale):
    """Section 2.2: at eps = inf the two ratios multiply back to pi_theta/pi_behaviour.

    This is the identity that makes PPO-EWMA a decomposition of PPO rather than a
    different objective, so it holds for *any* proximal policy -- the loss below
    must not depend on `proximal_logp` at all.
    """
    advantages, newlogp, behaviour_logp = random_batch()
    generator = torch.Generator().manual_seed(11)
    expected = (-advantages * (newlogp - behaviour_logp).exp()).mean()

    for _ in range(4):
        proximal_logp = behaviour_logp + torch.randn(
            behaviour_logp.shape, generator=generator, dtype=torch.float64)
        # clip_coef = 0 disables clipping, which is the eps = inf limit
        loss, _ = ppo_scale.policy_surrogate(
            advantages, newlogp, behaviour_logp, proximal_logp, 0.0, imp_samp_max=0.0)
        assert torch.allclose(loss, expected, atol=1e-12)


def test_logratio_is_measured_against_the_clips_own_reference(ppo_scale):
    """KL reporting must follow the regulariser, or the COM knob reads as a no-op."""
    advantages, newlogp, behaviour_logp = random_batch()
    proximal_logp = behaviour_logp + 0.3

    _, decoupled = ppo_scale.policy_surrogate(
        advantages, newlogp, behaviour_logp, proximal_logp, 0.1, 100.0)
    assert torch.allclose(decoupled, newlogp - proximal_logp)

    _, plain = ppo_scale.policy_surrogate(advantages, newlogp, behaviour_logp, None, 0.1, 100.0)
    assert torch.allclose(plain, newlogp - behaviour_logp)


def test_proximal_equal_to_behaviour_recovers_ppo(ppo_scale):
    """With no lag the decoupled loss is PPO's, up to log-vs-linear clipping."""
    advantages, newlogp, behaviour_logp = random_batch()
    ours, _ = ppo_scale.policy_surrogate(
        advantages, newlogp, behaviour_logp, behaviour_logp.clone(), 0.1, imp_samp_max=0.0)
    theirs = ppo_reference_surrogate(advantages, newlogp, behaviour_logp, 0.1)
    assert torch.allclose(ours, theirs, atol=1e-12)


def test_imp_samp_max_bounds_the_importance_ratio(ppo_scale):
    """Stale behaviour policies produce ratios that overflow if left alone."""
    size = 512
    generator = torch.Generator().manual_seed(5)
    newlogp = torch.zeros(size, dtype=torch.float64)
    # A behaviour policy up to e^30 less likely: the raw ratio would be ~1e13.
    behaviour_logp = -30.0 * torch.rand(size, generator=generator, dtype=torch.float64)
    advantages = -torch.ones(size, dtype=torch.float64)  # unclipped branch wins
    proximal_logp = newlogp.clone()

    bounded, _ = ppo_scale.policy_surrogate(
        advantages, newlogp, behaviour_logp, proximal_logp, 0.0, imp_samp_max=100.0)
    unbounded, _ = ppo_scale.policy_surrogate(
        advantages, newlogp, behaviour_logp, proximal_logp, 0.0, imp_samp_max=0.0)

    # -adv * ratio with adv = -1 is +ratio, so the mean is the mean ratio.
    assert bounded <= 100.0 + 1e-9
    assert unbounded > 1e5, 'the test batch should actually exercise the bound'


def test_imp_samp_max_does_not_alter_the_gradient_path(ppo_scale):
    """The bound lifts the detached denominator, so `newlogp` still carries grad."""
    generator = torch.Generator().manual_seed(13)
    logits = torch.randn(32, 6, generator=generator, dtype=torch.float64, requires_grad=True)
    actions = torch.randint(0, 6, (32,), generator=generator)
    advantages = torch.randn(32, generator=generator, dtype=torch.float64)
    behaviour_logp = -20.0 * torch.rand(32, generator=generator, dtype=torch.float64)
    newlogp = torch.distributions.Categorical(logits=logits).log_prob(actions)
    proximal_logp = newlogp.detach().clone()

    loss, _ = ppo_scale.policy_surrogate(
        advantages, newlogp, behaviour_logp, proximal_logp, 0.1, imp_samp_max=10.0)
    grad = torch.autograd.grad(loss, logits)[0]
    assert torch.isfinite(grad).all()
    assert grad.abs().sum() > 0


# ---------------------------------------------------------------------------
# the scaling recipe
# ---------------------------------------------------------------------------

def test_default_settings_match_ppos_partition(ppo_scale):
    """8 envs x 128 steps at minibatch 256 is `ppo_atari.py`'s 4 minibatches."""
    size, count = ppo_scale.resolve_minibatching(8 * 128, 256, None)
    assert (size, count) == (256, 4)


@pytest.mark.parametrize('num_envs', [8, 16, 64, 256, 2048])
def test_fixed_minibatch_size_scales_the_count(ppo_scale, num_envs):
    """The recipe: more environments buy more optimisation steps, same step size."""
    size, count = ppo_scale.resolve_minibatching(num_envs * 128, 256, None)
    assert size == 256
    assert count == num_envs * 128 // 256
    assert count == 4 * (num_envs // 8), 'count should scale linearly with num_envs'


@pytest.mark.parametrize('num_envs', [8, 64, 2048])
def test_fixed_count_scales_the_size_instead(ppo_scale, num_envs):
    """The Figure 9 arm the paper recommends against, kept reproducible."""
    size, count = ppo_scale.resolve_minibatching(num_envs * 128, 256, num_minibatches=4)
    assert count == 4
    assert size == num_envs * 128 // 4


def test_a_short_trailing_minibatch_still_counts_as_a_step(ppo_scale):
    """The loop strides by size and takes the remainder, so the count must ceil."""
    size, count = ppo_scale.resolve_minibatching(1000, 256, None)
    assert size == 256
    assert count == 4  # 256 + 256 + 256 + 232
    assert len(range(0, 1000, 256)) == count


def test_minibatch_larger_than_the_rollout_is_rejected(ppo_scale):
    """Silently shrinking it would quietly undo the recipe."""
    with pytest.raises(ValueError, match='exceeds the rollout'):
        ppo_scale.resolve_minibatching(1024, 4096, None)


@pytest.mark.parametrize('bad', [0, -1])
def test_degenerate_partitions_are_rejected(ppo_scale, bad):
    with pytest.raises(ValueError):
        ppo_scale.resolve_minibatching(1024, bad, None)
    with pytest.raises(ValueError):
        ppo_scale.resolve_minibatching(1024, 256, bad)


def test_too_many_minibatches_is_rejected(ppo_scale):
    with pytest.raises(ValueError):
        ppo_scale.resolve_minibatching(1024, 256, num_minibatches=2048)


# ---------------------------------------------------------------------------
# DDR
# ---------------------------------------------------------------------------

def test_ddr_is_data_over_divergence(ppo_scale):
    assert ppo_scale.data_to_divergence_ratio(524288, 0.05) == pytest.approx(524288 / 0.05)


def test_ddr_rises_with_parallel_environments(ppo_scale):
    """Why the paper's fix works: more envs raise DDR from both sides at once."""
    small = ppo_scale.data_to_divergence_ratio(8 * 128, 0.02)
    large = ppo_scale.data_to_divergence_ratio(2048 * 128, 0.02)
    assert large == pytest.approx(small * 256)


def test_ddr_falls_as_the_policy_moves_further(ppo_scale):
    kls = [0.001, 0.01, 0.1, 1.0]
    ratios = [ppo_scale.data_to_divergence_ratio(1024, kl) for kl in kls]
    assert ratios == sorted(ratios, reverse=True)


@pytest.mark.parametrize('kl', [0.0, -1e-9, float('nan'), float('inf')])
def test_ddr_is_nan_for_a_degenerate_divergence(ppo_scale, kl):
    """A policy that did not move is not an infinitely good ratio."""
    assert math.isnan(ppo_scale.data_to_divergence_ratio(1024, kl))


def test_ddr_matches_the_k3_estimator_the_loop_feeds_it(ppo_scale):
    """The logged KL is Schulman's k3, which is non-negative and unbiased."""
    generator = torch.Generator().manual_seed(17)
    logratio = 0.1 * torch.randn(100000, generator=generator, dtype=torch.float64)
    k3 = ((logratio.exp() - 1) - logratio).mean()
    assert k3 > 0
    assert k3 == pytest.approx(0.5 * (logratio ** 2).mean(), rel=0.05)
    assert math.isfinite(ppo_scale.data_to_divergence_ratio(1024, float(k3)))
