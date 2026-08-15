"""IMPALA / V-trace: equivalence against torchbeast's reference.

torchbeast's `core/vtrace.py` is a direct PyTorch port of DeepMind's original
`scalable_agent/vtrace.py` and imports nothing but torch, so it can be executed
here directly rather than transcribed — every test below diffs against the
authors' code actually running.

On top of that, the properties V-trace is supposed to have are checked
independently of any implementation: that it reduces to n-step returns when the
policies match, that clipping at rho_bar is what bounds the correction, that
episode boundaries cut the trace, and that the advantage bootstraps off `vs`
rather than off `V`.
"""

import importlib.util
import os

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from conftest import REPO_ROOT, DiscreteEnvStub, load_trainer

TRAINER = 'impala_atari'


@pytest.fixture(scope='module')
def impala():
    return load_trainer(TRAINER)


@pytest.fixture(scope='module')
def torchbeast_vtrace():
    """Load torchbeast's vtrace module straight from the clone.

    It imports only torch and collections, so no torchbeast install (and no
    C++ extension build) is needed.
    """
    path = os.path.join(REPO_ROOT, 'third_party', 'upstream', 'torchbeast',
                        'torchbeast', 'core', 'vtrace.py')
    if not os.path.exists(path):
        pytest.skip('torchbeast clone not present under third_party/upstream')
    spec = importlib.util.spec_from_file_location('_torchbeast_vtrace', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def random_batch(seed, num_steps=20, num_envs=8, num_actions=6, done_prob=0.1, dtype=torch.float64):
    generator = torch.Generator().manual_seed(seed)
    behaviour = torch.randn(num_steps, num_envs, num_actions, generator=generator, dtype=dtype)
    target = torch.randn(num_steps, num_envs, num_actions, generator=generator, dtype=dtype)
    actions = torch.randint(0, num_actions, (num_steps, num_envs), generator=generator)
    dones = (torch.rand(num_steps, num_envs, generator=generator) < done_prob).to(dtype)
    return dict(
        behaviour_logits=behaviour,
        target_logits=target,
        actions=actions,
        discounts=(1.0 - dones) * 0.99,
        rewards=torch.randn(num_steps, num_envs, generator=generator, dtype=dtype),
        values=torch.randn(num_steps, num_envs, generator=generator, dtype=dtype),
        bootstrap_value=torch.randn(num_envs, generator=generator, dtype=dtype),
    )


# ---------------------------------------------------------------------------
# against torchbeast, running
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('seed', range(8))
def test_vtrace_from_logits_matches_torchbeast(impala, torchbeast_vtrace, seed):
    batch = random_batch(seed)
    vs, pg_advantages, log_rhos = impala.vtrace_from_logits(
        behavior_policy_logits=batch['behaviour_logits'],
        target_policy_logits=batch['target_logits'],
        actions=batch['actions'],
        discounts=batch['discounts'],
        rewards=batch['rewards'],
        values=batch['values'],
        bootstrap_value=batch['bootstrap_value'],
    )
    reference = torchbeast_vtrace.from_logits(
        behavior_policy_logits=batch['behaviour_logits'],
        target_policy_logits=batch['target_logits'],
        actions=batch['actions'],
        discounts=batch['discounts'],
        rewards=batch['rewards'],
        values=batch['values'],
        bootstrap_value=batch['bootstrap_value'],
    )
    assert torch.allclose(vs, reference.vs, rtol=0, atol=1e-13)
    assert torch.allclose(pg_advantages, reference.pg_advantages, rtol=0, atol=1e-13)
    assert torch.allclose(log_rhos, reference.log_rhos, rtol=0, atol=1e-13)


@pytest.mark.parametrize('clip_rho,clip_pg', [(1.0, 1.0), (2.0, 1.0), (1.0, 5.0), (None, None), (10.0, 0.5)])
def test_clipping_thresholds_match_torchbeast(impala, torchbeast_vtrace, clip_rho, clip_pg):
    batch = random_batch(42)
    log_rhos = (impala.action_log_probs(batch['target_logits'], batch['actions'])
                - impala.action_log_probs(batch['behaviour_logits'], batch['actions']))
    common = dict(
        log_rhos=log_rhos,
        discounts=batch['discounts'],
        rewards=batch['rewards'],
        values=batch['values'],
        bootstrap_value=batch['bootstrap_value'],
        clip_rho_threshold=clip_rho,
        clip_pg_rho_threshold=clip_pg,
    )
    vs, pg_advantages = impala.vtrace_from_importance_weights(**common)
    reference = torchbeast_vtrace.from_importance_weights(**common)
    assert torch.allclose(vs, reference.vs, rtol=0, atol=1e-13)
    assert torch.allclose(pg_advantages, reference.pg_advantages, rtol=0, atol=1e-13)


def test_action_log_probs_matches_torchbeast(impala, torchbeast_vtrace):
    batch = random_batch(7)
    ours = impala.action_log_probs(batch['target_logits'], batch['actions'])
    theirs = torchbeast_vtrace.action_log_probs(batch['target_logits'], batch['actions'])
    assert torch.allclose(ours, theirs, rtol=0, atol=1e-14)
    # ...and it really is log softmax at the taken action.
    expected = F.log_softmax(batch['target_logits'], dim=-1).gather(
        -1, batch['actions'].unsqueeze(-1)).squeeze(-1)
    assert torch.allclose(ours, expected, rtol=0, atol=1e-14)


@pytest.mark.parametrize('seed', range(3))
def test_matches_torchbeast_with_extreme_ratios(impala, torchbeast_vtrace, seed):
    """Widely separated policies: rho spans many orders of magnitude."""
    batch = random_batch(200 + seed)
    batch['behaviour_logits'] = batch['behaviour_logits'] * 8.0
    batch['target_logits'] = batch['target_logits'] * 8.0
    vs, pg_advantages, log_rhos = impala.vtrace_from_logits(
        behavior_policy_logits=batch['behaviour_logits'],
        target_policy_logits=batch['target_logits'],
        actions=batch['actions'],
        discounts=batch['discounts'],
        rewards=batch['rewards'],
        values=batch['values'],
        bootstrap_value=batch['bootstrap_value'],
    )
    reference = torchbeast_vtrace.from_logits(
        behavior_policy_logits=batch['behaviour_logits'],
        target_policy_logits=batch['target_logits'],
        actions=batch['actions'],
        discounts=batch['discounts'],
        rewards=batch['rewards'],
        values=batch['values'],
        bootstrap_value=batch['bootstrap_value'],
    )
    assert log_rhos.exp().max() > 50  # the regime actually got exercised
    assert torch.allclose(vs, reference.vs, rtol=0, atol=1e-12)
    assert torch.allclose(pg_advantages, reference.pg_advantages, rtol=0, atol=1e-12)


# ---------------------------------------------------------------------------
# the properties V-trace is supposed to have
# ---------------------------------------------------------------------------

def test_on_policy_vtrace_is_nstep_returns(impala):
    """The reduction this repo's default settings rely on.

    With mu == pi every rho and c is exactly 1, so
    `vs = V + sum gamma^k delta_k` telescopes into the n-step return
    `r_t + gamma r_{t+1} + ... + gamma^n V(x_{t+n})`.
    """
    batch = random_batch(11)
    log_rhos = torch.zeros_like(batch['rewards'])
    vs, pg_advantages = impala.vtrace_from_importance_weights(
        log_rhos=log_rhos,
        discounts=batch['discounts'],
        rewards=batch['rewards'],
        values=batch['values'],
        bootstrap_value=batch['bootstrap_value'],
    )

    num_steps = batch['rewards'].shape[0]
    expected = torch.zeros_like(batch['rewards'])
    running = batch['bootstrap_value']
    for t in reversed(range(num_steps)):
        running = batch['rewards'][t] + batch['discounts'][t] * running
        expected[t] = running
    assert torch.allclose(vs, expected, rtol=0, atol=1e-11)

    # ...and the advantage becomes the ordinary TD error against those targets.
    vs_t_plus_1 = torch.cat([vs[1:], batch['bootstrap_value'].unsqueeze(0)], dim=0)
    td = batch['rewards'] + batch['discounts'] * vs_t_plus_1 - batch['values']
    assert torch.allclose(pg_advantages, td, rtol=0, atol=1e-12)


def test_identical_policies_give_unit_rhos(impala):
    batch = random_batch(3)
    _, _, log_rhos = impala.vtrace_from_logits(
        behavior_policy_logits=batch['target_logits'],
        target_policy_logits=batch['target_logits'],
        actions=batch['actions'],
        discounts=batch['discounts'],
        rewards=batch['rewards'],
        values=batch['values'],
        bootstrap_value=batch['bootstrap_value'],
    )
    assert torch.allclose(log_rhos, torch.zeros_like(log_rhos), atol=1e-13)


def test_done_cuts_the_trace(impala):
    """A terminal step must stop the recursion; nothing past it may leak back."""
    num_steps, num_envs = 6, 1
    rewards = torch.ones(num_steps, num_envs, dtype=torch.float64)
    values = torch.zeros(num_steps, num_envs, dtype=torch.float64)
    bootstrap = torch.tensor([1000.0], dtype=torch.float64)
    log_rhos = torch.zeros(num_steps, num_envs, dtype=torch.float64)

    discounts = torch.full((num_steps, num_envs), 0.99, dtype=torch.float64)
    discounts[2] = 0.0  # step 2 terminates

    vs, _ = impala.vtrace_from_importance_weights(
        log_rhos=log_rhos, discounts=discounts, rewards=rewards,
        values=values, bootstrap_value=bootstrap)

    # Steps 0..2 see only the rewards up to the terminal step.
    assert np.isclose(vs[2].item(), 1.0, atol=1e-12)
    assert np.isclose(vs[1].item(), 1.0 + 0.99 * 1.0, atol=1e-12)
    assert np.isclose(vs[0].item(), 1.0 + 0.99 * (1.0 + 0.99), atol=1e-12)
    # Step 3 onwards does see the bootstrap.
    assert vs[3].item() > 900


def test_rho_clipping_bounds_the_correction(impala):
    """Raising rho_bar past the largest ratio must stop changing anything."""
    batch = random_batch(5)
    log_rhos = (impala.action_log_probs(batch['target_logits'], batch['actions'])
                - impala.action_log_probs(batch['behaviour_logits'], batch['actions']))
    common = dict(log_rhos=log_rhos, discounts=batch['discounts'], rewards=batch['rewards'],
                  values=batch['values'], bootstrap_value=batch['bootstrap_value'])

    tight, _ = impala.vtrace_from_importance_weights(clip_rho_threshold=1.0, **common)
    loose, _ = impala.vtrace_from_importance_weights(clip_rho_threshold=1e6, **common)
    unclipped, _ = impala.vtrace_from_importance_weights(clip_rho_threshold=None, **common)

    assert torch.allclose(loose, unclipped, rtol=0, atol=1e-12)
    assert not torch.allclose(tight, unclipped, rtol=1e-3, atol=1e-3)


def test_c_bar_is_hard_coded_to_one(impala):
    """`cs = clamp(rhos, max=1)` regardless of rho_bar, as in every reference.

    Raising rho_bar changes the deltas but must not change the trace product,
    so a run with huge rho_bar must still differ from one where the traces were
    left unclipped. Detected here by comparing against a hand-rolled recursion
    that uses `c = rho` instead.
    """
    num_steps, num_envs = 5, 2
    torch.manual_seed(0)
    log_rhos = torch.abs(torch.randn(num_steps, num_envs, dtype=torch.float64)) + 1.0  # rho > e
    discounts = torch.full((num_steps, num_envs), 0.99, dtype=torch.float64)
    rewards = torch.randn(num_steps, num_envs, dtype=torch.float64)
    values = torch.randn(num_steps, num_envs, dtype=torch.float64)
    bootstrap = torch.randn(num_envs, dtype=torch.float64)

    vs, _ = impala.vtrace_from_importance_weights(
        log_rhos=log_rhos, discounts=discounts, rewards=rewards, values=values,
        bootstrap_value=bootstrap, clip_rho_threshold=1e9)

    rhos = log_rhos.exp()
    values_t_plus_1 = torch.cat([values[1:], bootstrap.unsqueeze(0)], dim=0)
    deltas = rhos * (rewards + discounts * values_t_plus_1 - values)
    acc = torch.zeros_like(bootstrap)
    wrong = torch.zeros_like(rewards)
    for t in reversed(range(num_steps)):
        acc = deltas[t] + discounts[t] * rhos[t] * acc  # c = rho, the bug
        wrong[t] = acc
    wrong = wrong + values

    assert not torch.allclose(vs, wrong, rtol=1e-4, atol=1e-4)


def test_pg_advantage_bootstraps_off_vs_not_values(impala):
    """`A_t = rho_t (r_t + gamma v_{t+1} - V_t)`, with v the V-trace target."""
    batch = random_batch(9)
    vs, pg_advantages = impala.vtrace_from_importance_weights(
        log_rhos=torch.zeros_like(batch['rewards']),
        discounts=batch['discounts'], rewards=batch['rewards'],
        values=batch['values'], bootstrap_value=batch['bootstrap_value'])

    vs_next = torch.cat([vs[1:], batch['bootstrap_value'].unsqueeze(0)], dim=0)
    values_next = torch.cat([batch['values'][1:], batch['bootstrap_value'].unsqueeze(0)], dim=0)
    right = batch['rewards'] + batch['discounts'] * vs_next - batch['values']
    wrong = batch['rewards'] + batch['discounts'] * values_next - batch['values']

    assert torch.allclose(pg_advantages, right, rtol=0, atol=1e-12)
    assert not torch.allclose(pg_advantages, wrong, rtol=1e-4, atol=1e-4)


def test_vtrace_returns_carry_no_gradient(impala):
    """monobeast treats `vs` as data; a leaked graph would train the critic twice."""
    values = torch.randn(4, 3, dtype=torch.float64, requires_grad=True)
    vs, pg_advantages = impala.vtrace_from_importance_weights(
        log_rhos=torch.zeros(4, 3, dtype=torch.float64),
        discounts=torch.full((4, 3), 0.99, dtype=torch.float64),
        rewards=torch.randn(4, 3, dtype=torch.float64),
        values=values,
        bootstrap_value=torch.randn(3, dtype=torch.float64))
    assert not vs.requires_grad
    assert not pg_advantages.requires_grad


# ---------------------------------------------------------------------------
# the learner losses
# ---------------------------------------------------------------------------

def test_losses_match_monobeast(impala):
    """monobeast sums; the port must too, or the learning rate is off by T*B."""
    torch.manual_seed(0)
    num_steps, num_envs, num_actions = 7, 5, 6
    logits = torch.randn(num_steps, num_envs, num_actions, dtype=torch.float64)
    actions = torch.randint(0, num_actions, (num_steps, num_envs))
    advantages = torch.randn(num_steps, num_envs, dtype=torch.float64)

    # monobeast's `compute_baseline_loss`
    assert np.isclose(
        impala.compute_baseline_loss(advantages).item(),
        (0.5 * torch.sum(advantages**2)).item(), rtol=0, atol=1e-12)

    # monobeast's `compute_entropy_loss`
    policy = F.softmax(logits, dim=-1)
    log_policy = F.log_softmax(logits, dim=-1)
    assert np.isclose(
        impala.compute_entropy_loss(logits).item(),
        torch.sum(policy * log_policy).item(), rtol=0, atol=1e-11)

    # monobeast's `compute_policy_gradient_loss`
    cross_entropy = F.nll_loss(
        F.log_softmax(torch.flatten(logits, 0, 1), dim=-1),
        target=torch.flatten(actions, 0, 1), reduction='none').view_as(advantages)
    assert np.isclose(
        impala.compute_policy_gradient_loss(logits, actions, advantages).item(),
        torch.sum(cross_entropy * advantages).item(), rtol=0, atol=1e-11)


def test_entropy_loss_is_negative_entropy(impala):
    """It is a *loss*: a uniform policy must score lower than a peaked one."""
    uniform = torch.zeros(1, 1, 6, dtype=torch.float64)
    peaked = torch.tensor([[[10.0, 0, 0, 0, 0, 0]]], dtype=torch.float64)
    assert impala.compute_entropy_loss(uniform).item() < impala.compute_entropy_loss(peaked).item()
    assert np.isclose(impala.compute_entropy_loss(uniform).item(), -np.log(6), atol=1e-12)


def test_mean_reduction_is_sum_over_count(impala):
    torch.manual_seed(1)
    num_steps, num_envs, num_actions = 4, 3, 5
    logits = torch.randn(num_steps, num_envs, num_actions, dtype=torch.float64)
    actions = torch.randint(0, num_actions, (num_steps, num_envs))
    advantages = torch.randn(num_steps, num_envs, dtype=torch.float64)
    count = num_steps * num_envs

    for summed, meaned in (
        (impala.compute_baseline_loss(advantages, 'sum'), impala.compute_baseline_loss(advantages, 'mean')),
        (impala.compute_entropy_loss(logits, 'sum'), impala.compute_entropy_loss(logits, 'mean')),
        (impala.compute_policy_gradient_loss(logits, actions, advantages, 'sum'),
         impala.compute_policy_gradient_loss(logits, actions, advantages, 'mean')),
    ):
        assert np.isclose(summed.item() / count, meaned.item(), rtol=0, atol=1e-12)


def test_policy_gradient_loss_detaches_the_advantage(impala):
    logits = torch.randn(3, 2, 4, dtype=torch.float64, requires_grad=True)
    advantages = torch.randn(3, 2, dtype=torch.float64, requires_grad=True)
    loss = impala.compute_policy_gradient_loss(logits, torch.randint(0, 4, (3, 2)), advantages)
    loss.backward()
    assert advantages.grad is None
    assert logits.grad is not None


# ---------------------------------------------------------------------------
# end to end: one learner step against torchbeast's, on shared weights
# ---------------------------------------------------------------------------

def test_full_learner_step_matches_monobeast(impala, torchbeast_vtrace):
    """Build the whole monobeast loss from its own pieces and diff the gradient.

    This is the check that would catch a wrong `discounts`, a dropped bootstrap
    row, or the target/behaviour logits swapped — none of which the
    component-level tests above can see.
    """
    torch.manual_seed(0)
    num_steps, num_envs, num_actions = 12, 4, 6
    envs = DiscreteEnvStub(num_actions)
    agent = impala.Agent(envs).double()

    generator = torch.Generator().manual_seed(5)
    obs = torch.randint(0, 255, (num_steps + 1, num_envs, 4, 84, 84),
                        generator=generator, dtype=torch.uint8).double()
    actions = torch.randint(0, num_actions, (num_steps, num_envs), generator=generator)
    behaviour_logits = torch.randn(num_steps, num_envs, num_actions, generator=generator, dtype=torch.float64)
    rewards = torch.randn(num_steps, num_envs, generator=generator, dtype=torch.float64)
    dones = (torch.rand(num_steps, num_envs, generator=generator) < 0.15).double()
    gamma, vf_coef, ent_coef = 0.99, 0.5, 0.0006

    def learner_outputs():
        flat_logits, flat_values = agent(obs.flatten(0, 1))
        return (flat_logits.view(num_steps + 1, num_envs, -1),
                flat_values.view(num_steps + 1, num_envs))

    # --- ours
    target_logits, values = learner_outputs()
    bootstrap_value = values[-1]
    vs, pg_advantages, _ = impala.vtrace_from_logits(
        behavior_policy_logits=behaviour_logits,
        target_policy_logits=target_logits[:-1],
        actions=actions,
        discounts=(1.0 - dones) * gamma,
        rewards=rewards,
        values=values[:-1],
        bootstrap_value=bootstrap_value,
    )
    ours = (impala.compute_policy_gradient_loss(target_logits[:-1], actions, pg_advantages)
            + vf_coef * impala.compute_baseline_loss(vs - values[:-1])
            + ent_coef * impala.compute_entropy_loss(target_logits[:-1]))
    our_grads = torch.autograd.grad(ours, list(agent.parameters()), retain_graph=False)

    # --- monobeast, using torchbeast's own vtrace
    target_logits, values = learner_outputs()
    reference = torchbeast_vtrace.from_logits(
        behavior_policy_logits=behaviour_logits,
        target_policy_logits=target_logits[:-1],
        actions=actions,
        discounts=(1.0 - dones) * gamma,
        rewards=rewards,
        values=values[:-1],
        bootstrap_value=values[-1],
    )
    cross_entropy = F.nll_loss(
        F.log_softmax(torch.flatten(target_logits[:-1], 0, 1), dim=-1),
        target=torch.flatten(actions, 0, 1), reduction='none').view_as(reference.pg_advantages)
    policy = F.softmax(target_logits[:-1], dim=-1)
    log_policy = F.log_softmax(target_logits[:-1], dim=-1)
    theirs = (torch.sum(cross_entropy * reference.pg_advantages.detach())
              + vf_coef * 0.5 * torch.sum((reference.vs - values[:-1]) ** 2)
              + ent_coef * torch.sum(policy * log_policy))
    their_grads = torch.autograd.grad(theirs, list(agent.parameters()), retain_graph=False)

    assert np.isclose(ours.item(), theirs.item(), rtol=0, atol=1e-9)
    for got, want in zip(our_grads, their_grads):
        assert torch.allclose(got, want, rtol=0, atol=1e-10)


# ---------------------------------------------------------------------------
# the agent
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('encoder', ['nature', 'impala'])
def test_agent_forward_shapes(impala, encoder):
    torch.manual_seed(0)
    envs = DiscreteEnvStub(6)
    agent = impala.Agent(envs, encoder)
    obs = torch.randint(0, 255, (5, 4, 84, 84), dtype=torch.uint8).float()
    logits, value = agent(obs)
    assert logits.shape == (5, 6)
    assert value.shape == (5,)
    action, sampled_logits, sampled_value = agent.get_action_logits_and_value(obs)
    assert action.shape == (5,) and sampled_logits.shape == (5, 6) and sampled_value.shape == (5,)


def test_impala_encoder_is_the_deep_residual_model(impala):
    """The paper's "deep" model: 3 conv sequences x (1 conv + 2 residual blocks)."""
    envs = DiscreteEnvStub(6)
    nature_convs = [m for m in impala.Agent(envs, 'nature').modules() if isinstance(m, torch.nn.Conv2d)]
    deep_convs = [m for m in impala.Agent(envs, 'impala').modules() if isinstance(m, torch.nn.Conv2d)]
    assert len(nature_convs) == 3
    assert len(deep_convs) == 15  # 3 sequences * (1 + 2 blocks * 2 convs)
    assert [c.out_channels for c in deep_convs[:1]] == [16]

    # 84 -> 42 -> 21 -> 11 at 32 channels, so the trunk sees 3872 features
    # rather than the Nature CNN's 3136.
    deep_linear = [m for m in impala.Agent(envs, 'impala').modules() if isinstance(m, torch.nn.Linear)]
    assert deep_linear[0].in_features == 32 * 11 * 11


def test_unknown_encoder_rejected(impala):
    with pytest.raises(ValueError, match='unsupported encoder'):
        impala.Agent(DiscreteEnvStub(4), 'resnet50')


def test_lr_schedule_matches_torchbeast_lambda(impala):
    total = 1000
    assert np.isclose(impala.torchbeast_lr_fraction(0, total), 1.0)
    assert np.isclose(impala.torchbeast_lr_fraction(250, total), 0.75)
    assert np.isclose(impala.torchbeast_lr_fraction(total, total), 0.0)
    # `min(...)` in torchbeast's lambda clamps past the end.
    assert np.isclose(impala.torchbeast_lr_fraction(10 * total, total), 0.0)
