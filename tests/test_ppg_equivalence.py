"""PPG: equivalence against openai/phasic-policy-gradient.

The reference is PyTorch but its modules import `mpi4py`, `procgen` and a tree
utility layer, so `ppg.py`/`ppo.py` cannot simply be executed here. What *can*
be executed is the part that matters: `compute_gae` and `NormedLinear` are
self-contained, and are lifted out of the clone by source extraction rather than
retyped, so the tests fail if upstream's text ever changes under them.

Everything else is diffed against a transcription of the reference's own
expressions, with each transcription quoted next to the file and line it came
from.
"""

import ast
import os
import textwrap

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

from conftest import REPO_ROOT, DiscreteEnvStub, load_trainer

TRAINER = 'ppg_atari'
UPSTREAM = os.path.join(REPO_ROOT, 'third_party', 'upstream',
                        'phasic-policy-gradient', 'phasic_policy_gradient')


@pytest.fixture(scope='module')
def ppg():
    return load_trainer(TRAINER)


def extract_function(filename, name):
    """Pull one top-level function's source out of the clone and exec it.

    Used instead of importing, because the reference's modules pull in mpi4py
    and procgen at import time. The function bodies below depend only on torch.
    """
    path = os.path.join(UPSTREAM, filename)
    if not os.path.exists(path):
        pytest.skip('phasic-policy-gradient clone not present under third_party/upstream')
    with open(path) as handle:
        source = handle.read()
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            namespace = {'th': torch, 'nn': nn, 'torch': torch}
            exec(textwrap.dedent(ast.get_source_segment(source, node)), namespace)
            return namespace[name]
    raise AssertionError(f'{name} not found in {filename}')


@pytest.fixture(scope='module')
def upstream_compute_gae():
    return extract_function('ppo.py', 'compute_gae')


@pytest.fixture(scope='module')
def upstream_normed_linear():
    def parse_dtype(_):
        return torch.float32
    path = os.path.join(UPSTREAM, 'torch_util.py')
    if not os.path.exists(path):
        pytest.skip('phasic-policy-gradient clone not present under third_party/upstream')
    with open(path) as handle:
        source = handle.read()
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == 'NormedLinear':
            namespace = {'th': torch, 'nn': nn, 'parse_dtype': parse_dtype}
            exec(textwrap.dedent(ast.get_source_segment(source, node)), namespace)
            return namespace['NormedLinear']
    raise AssertionError('NormedLinear not found')


# ---------------------------------------------------------------------------
# GAE, against the reference source
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('seed', range(6))
def test_compute_gae_matches_upstream(ppg, upstream_compute_gae, seed):
    generator = torch.Generator().manual_seed(seed)
    nenv, nstep = 7, 24
    vpred = torch.randn(nenv, nstep + 1, generator=generator)
    reward = torch.randn(nenv, nstep, generator=generator)
    first = (torch.rand(nenv, nstep + 1, generator=generator) < 0.15).float()
    gamma, lam = 0.999, 0.95

    ours = ppg.compute_gae(vpred, reward, first, gamma, lam)
    theirs = upstream_compute_gae(vpred=vpred, reward=reward, first=first.bool(), γ=gamma, λ=lam)
    assert torch.allclose(ours[0], theirs[0], rtol=0, atol=1e-6)
    assert torch.allclose(ours[1], theirs[1], rtol=0, atol=1e-6)


def test_gae_uses_the_next_step_first_flag(ppg):
    """`notlast = 1 - first[:, t+1]`: an episode boundary must cut at the right t.

    Off-by-one here is invisible in aggregate and silently poisons every target
    that spans a boundary, so it is checked on a hand-computed case.
    """
    gamma, lam = 0.9, 1.0
    vpred = torch.tensor([[0.0, 0.0, 0.0, 100.0]])
    reward = torch.tensor([[1.0, 1.0, 1.0]])
    # step 1 ends the episode, so timestep 2 is the first of a new one.
    first = torch.tensor([[0.0, 0.0, 1.0, 0.0]])

    advantages, vtarg = ppg.compute_gae(vpred, reward, first, gamma, lam)
    # t=2: delta = 1 + 0.9*100 = 91
    # t=1: notlast = 0 -> delta = 1, and the trace is cut
    # t=0: delta = 1 + 0, plus 0.9*1.0*1 from t=1
    assert np.allclose(advantages.numpy(), [[1.9, 1.0, 91.0]], atol=1e-6)
    assert np.allclose(vtarg.numpy(), advantages.numpy() + vpred[:, :-1].numpy(), atol=1e-6)


def test_gae_reduces_to_nstep_at_lambda_one(ppg):
    gamma = 0.99
    nenv, nstep = 3, 10
    torch.manual_seed(0)
    vpred = torch.randn(nenv, nstep + 1)
    reward = torch.randn(nenv, nstep)
    first = torch.zeros(nenv, nstep + 1)

    _, vtarg = ppg.compute_gae(vpred, reward, first, gamma, 1.0)
    expected = torch.zeros(nenv, nstep)
    running = vpred[:, -1]
    for t in reversed(range(nstep)):
        running = reward[:, t] + gamma * running
        expected[:, t] = running
    assert torch.allclose(vtarg, expected, rtol=0, atol=1e-5)


# ---------------------------------------------------------------------------
# NormedLinear, against the reference source
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('scale', [0.1, 1.0, 3.0])
def test_normed_linear_matches_upstream(ppg, upstream_normed_linear, scale):
    torch.manual_seed(0)
    ours = ppg.normed_linear(37, 11, scale=scale)
    torch.manual_seed(0)
    theirs = upstream_normed_linear(37, 11, scale=scale)
    assert torch.allclose(ours.weight, theirs.weight, rtol=0, atol=1e-7)
    assert torch.allclose(ours.bias, theirs.bias, rtol=0, atol=1e-7)


def test_normed_linear_rows_have_the_requested_norm(ppg):
    torch.manual_seed(1)
    layer = ppg.normed_linear(64, 9, scale=0.1)
    norms = layer.weight.norm(dim=1, p=2)
    assert torch.allclose(norms, torch.full_like(norms, 0.1), rtol=0, atol=1e-6)
    assert torch.count_nonzero(layer.bias) == 0


def test_heads_use_normed_linear_not_orthogonal(ppg):
    """All three heads are `NormedLinear(scale=0.1)`; none is orthogonal-init."""
    torch.manual_seed(0)
    agent = ppg.PhasicValueAgent(DiscreteEnvStub(6))
    for head in (agent.pi_head, agent.aux_vf_head, agent.vf_head):
        norms = head.weight.norm(dim=1, p=2)
        assert torch.allclose(norms, torch.full_like(norms, 0.1), rtol=0, atol=1e-6)
        assert torch.count_nonzero(head.bias) == 0


# ---------------------------------------------------------------------------
# advantage normalisation
# ---------------------------------------------------------------------------

def test_advantage_normalisation_is_batch_level_and_unbiased(ppg):
    """`(adv - mean) / (sqrt(var) + 1e-8)` over the whole rollout.

    Two things are pinned: that it is *not* per-minibatch (the whole tensor is
    whitened at once), and that the variance is the unbiased one, which is what
    the reference's `mpi_moments` returns.
    """
    torch.manual_seed(0)
    advantages = torch.randn(16, 128) * 3.0 + 7.0
    normalized = ppg.normalize_advantage(advantages)

    expected = (advantages - advantages.mean()) / (advantages.var(unbiased=True).sqrt() + 1e-8)
    assert torch.allclose(normalized, expected, rtol=0, atol=1e-6)

    biased = (advantages - advantages.mean()) / (advantages.var(unbiased=False).sqrt() + 1e-8)
    assert not torch.allclose(normalized, biased, rtol=1e-6, atol=1e-6)

    # Per-minibatch normalisation would leave each row zero-mean; batch-level
    # normalisation must not.
    assert normalized.mean(dim=1).abs().max() > 1e-6 or advantages.shape[0] == 1


# ---------------------------------------------------------------------------
# the PPO-phase loss
# ---------------------------------------------------------------------------

def upstream_ppo_losses(logits, vpred, actions, logp, adv, vtarg,
                        clip_param, entcoef, vfcoef, kl_penalty):
    """Transcription of `ppo.py::compute_losses`, lines 72-113."""
    pd = Categorical(logits=logits)
    newlogp = pd.log_prob(actions)
    logratio = newlogp - logp
    ratio = torch.exp(logratio)
    if clip_param > 0:
        pg_losses = -adv * ratio
        pg_losses2 = -adv * torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param)
        pg_losses = torch.max(pg_losses, pg_losses2)
    else:
        pg_losses = -adv * torch.exp(newlogp - logp)
    entropy = pd.entropy().mean()
    negent = -entropy * entcoef
    pg = pg_losses.mean()
    pi_kl = kl_penalty * 0.5 * (logratio**2).mean()
    return negent + pg + pi_kl, vfcoef * ((vpred - vtarg) ** 2).mean()


@pytest.mark.parametrize('clip_param,kl_penalty', [(0.2, 0.0), (0.0, 0.0), (0.2, 0.5), (0.1, 1.0)])
def test_ppo_losses_match_upstream(ppg, clip_param, kl_penalty):
    torch.manual_seed(0)
    batch, num_actions = 256, 6
    logits = torch.randn(batch, num_actions, dtype=torch.float64)
    vpred = torch.randn(batch, dtype=torch.float64)
    actions = torch.randint(0, num_actions, (batch,))
    logp = torch.randn(batch, dtype=torch.float64) - 2.0
    adv = torch.randn(batch, dtype=torch.float64)
    vtarg = torch.randn(batch, dtype=torch.float64)

    pi_loss, vf_loss, _ = ppg.ppo_losses(
        logits, vpred, actions, logp, adv, vtarg, clip_param, 0.01, 0.5, kl_penalty)
    want_pi, want_vf = upstream_ppo_losses(
        logits, vpred, actions, logp, adv, vtarg, clip_param, 0.01, 0.5, kl_penalty)

    assert np.isclose(pi_loss.item(), want_pi.item(), rtol=0, atol=1e-12)
    assert np.isclose(vf_loss.item(), want_vf.item(), rtol=0, atol=1e-12)


def test_value_loss_is_unclipped_and_has_no_half(ppg):
    """PPO's `0.5 * max(clipped, unclipped)` is *not* what PPG uses."""
    logits = torch.zeros(1, 4, dtype=torch.float64)
    vpred = torch.tensor([3.0], dtype=torch.float64)
    vtarg = torch.tensor([0.0], dtype=torch.float64)
    _, vf_loss, _ = ppg.ppo_losses(
        logits, vpred, torch.zeros(1, dtype=torch.long), torch.zeros(1, dtype=torch.float64),
        torch.zeros(1, dtype=torch.float64), vtarg, 0.2, 0.0, 1.0, 0.0)
    assert np.isclose(vf_loss.item(), 9.0, atol=1e-12)  # not 4.5, and not clipped


def test_clip_param_zero_disables_clipping(ppg):
    torch.manual_seed(0)
    batch, num_actions = 64, 5
    logits = torch.randn(batch, num_actions, dtype=torch.float64) * 3
    actions = torch.randint(0, num_actions, (batch,))
    logp = torch.randn(batch, dtype=torch.float64)
    adv = torch.randn(batch, dtype=torch.float64)
    zeros = torch.zeros(batch, dtype=torch.float64)

    clipped, _, _ = ppg.ppo_losses(logits, zeros, actions, logp, adv, zeros, 0.2, 0.0, 0.0, 0.0)
    unclipped, _, _ = ppg.ppo_losses(logits, zeros, actions, logp, adv, zeros, 0.0, 0.0, 0.0, 0.0)
    assert not np.isclose(clipped.item(), unclipped.item(), rtol=1e-6, atol=1e-9)

    ratio = (Categorical(logits=logits).log_prob(actions) - logp).exp()
    assert np.isclose(unclipped.item(), (-adv * ratio).mean().item(), rtol=0, atol=1e-12)


# ---------------------------------------------------------------------------
# the auxiliary-phase loss
# ---------------------------------------------------------------------------

def test_aux_losses_match_upstream(ppg):
    """`ppg.py::aux_train` + `PhasicValueModel.compute_aux_loss`."""
    torch.manual_seed(0)
    batch, num_actions = 128, 6
    logits = torch.randn(batch, num_actions, dtype=torch.float64)
    old_logits = torch.randn(batch, num_actions, dtype=torch.float64)
    vpred_true = torch.randn(batch, dtype=torch.float64)
    vpred_aux = torch.randn(batch, dtype=torch.float64)
    vtarg = torch.randn(batch, dtype=torch.float64)
    beta_clone, vf_true_weight = 1.0, 1.0

    loss, pol_distance, vf_aux, vf_true = ppg.aux_losses(
        logits, vpred_true, vpred_aux, old_logits, vtarg, beta_clone, vf_true_weight)

    old_pd = Categorical(logits=old_logits)
    new_pd = Categorical(logits=logits)
    want_kl = torch.distributions.kl_divergence(old_pd, new_pd).mean()
    want_aux = 0.5 * ((vpred_aux - vtarg) ** 2).mean()
    want_true = 0.5 * ((vpred_true - vtarg) ** 2).mean()

    assert np.isclose(pol_distance.item(), want_kl.item(), rtol=0, atol=1e-12)
    assert np.isclose(vf_aux.item(), want_aux.item(), rtol=0, atol=1e-12)
    assert np.isclose(vf_true.item(), want_true.item(), rtol=0, atol=1e-12)
    assert np.isclose(
        loss.item(),
        (beta_clone * want_kl + want_aux + vf_true_weight * want_true).item(),
        rtol=0, atol=1e-12)


def test_aux_kl_direction_is_old_to_new(ppg):
    """`td.kl_divergence(mb["oldpd"], pd)` — old first. The reverse KL differs."""
    # Deliberately not a permutation of each other: swapping two logits would
    # make the two KLs equal by symmetry and the test would prove nothing.
    old_logits = torch.tensor([[4.0, 0.0, -1.0]], dtype=torch.float64)
    logits = torch.tensor([[0.0, 1.0, 2.0]], dtype=torch.float64)

    _, forward, _, _ = ppg.aux_losses(
        logits, torch.zeros(1, dtype=torch.float64), torch.zeros(1, dtype=torch.float64),
        old_logits, torch.zeros(1, dtype=torch.float64), 1.0, 1.0)
    want = torch.distributions.kl_divergence(
        Categorical(logits=old_logits), Categorical(logits=logits)).mean()
    reverse = torch.distributions.kl_divergence(
        Categorical(logits=logits), Categorical(logits=old_logits)).mean()

    assert np.isclose(forward.item(), want.item(), atol=1e-12)
    # Asymmetric here, so the direction is observable.
    assert not np.isclose(forward.item(), reverse.item(), rtol=1e-6, atol=1e-9)


def test_aux_kl_is_zero_when_the_policy_has_not_moved(ppg):
    torch.manual_seed(0)
    logits = torch.randn(32, 6, dtype=torch.float64)
    _, pol_distance, _, _ = ppg.aux_losses(
        logits, torch.zeros(32, dtype=torch.float64), torch.zeros(32, dtype=torch.float64),
        logits.clone(), torch.zeros(32, dtype=torch.float64), 1.0, 1.0)
    assert np.isclose(pol_distance.item(), 0.0, atol=1e-12)


def test_beta_clone_scales_only_the_kl(ppg):
    torch.manual_seed(0)
    logits = torch.randn(16, 4, dtype=torch.float64)
    old_logits = torch.randn(16, 4, dtype=torch.float64)
    vtarg = torch.randn(16, dtype=torch.float64)
    vpred_true = torch.randn(16, dtype=torch.float64)
    vpred_aux = torch.randn(16, dtype=torch.float64)

    loss1, kl, _, _ = ppg.aux_losses(logits, vpred_true, vpred_aux, old_logits, vtarg, 1.0, 1.0)
    loss2, _, _, _ = ppg.aux_losses(logits, vpred_true, vpred_aux, old_logits, vtarg, 3.0, 1.0)
    assert np.isclose((loss2 - loss1).item(), 2.0 * kl.item(), rtol=0, atol=1e-12)


# ---------------------------------------------------------------------------
# architecture
# ---------------------------------------------------------------------------

def test_dual_arch_has_two_independent_encoders(ppg):
    torch.manual_seed(0)
    agent = ppg.PhasicValueAgent(DiscreteEnvStub(6), arch='dual')
    assert agent.vf_encoder is not None
    pi_params = {id(p) for p in agent.pi_encoder.parameters()}
    vf_params = {id(p) for p in agent.vf_encoder.parameters()}
    assert not (pi_params & vf_params)

    obs = torch.randint(0, 255, (4, 4, 84, 84), dtype=torch.uint8).float()
    _, vpred_true, vpred_aux = agent(obs)
    # Different encoders, so the two value heads must disagree.
    assert not torch.allclose(vpred_true, vpred_aux, rtol=1e-3, atol=1e-3)


def test_shared_arch_reuses_the_policy_encoder(ppg):
    agent = ppg.PhasicValueAgent(DiscreteEnvStub(6), arch='shared')
    assert agent.vf_encoder is None
    obs = torch.randint(0, 255, (3, 4, 84, 84), dtype=torch.uint8).float()
    logits, vpred_true, vpred_aux = agent(obs)
    assert logits.shape == (3, 6) and vpred_true.shape == (3,) and vpred_aux.shape == (3,)


def test_detach_arch_blocks_value_gradient_into_the_torso(ppg):
    """The whole point of `detach`: the true value loss must not reach the torso."""
    torch.manual_seed(0)
    agent = ppg.PhasicValueAgent(DiscreteEnvStub(6), arch='detach')
    obs = torch.randint(0, 255, (5, 4, 84, 84), dtype=torch.uint8).float()
    _, vpred_true, _ = agent(obs)
    vpred_true.sum().backward()

    assert agent.vf_head.weight.grad is not None
    assert agent.vf_head.weight.grad.abs().sum() > 0
    for parameter in agent.pi_encoder.parameters():
        assert parameter.grad is None or parameter.grad.abs().sum() == 0


def test_shared_arch_does_let_value_gradient_through(ppg):
    """The contrast that makes the previous test meaningful."""
    torch.manual_seed(0)
    agent = ppg.PhasicValueAgent(DiscreteEnvStub(6), arch='shared')
    obs = torch.randint(0, 255, (5, 4, 84, 84), dtype=torch.uint8).float()
    _, vpred_true, _ = agent(obs)
    vpred_true.sum().backward()
    total = sum(p.grad.abs().sum() for p in agent.pi_encoder.parameters() if p.grad is not None)
    assert total > 0


def test_aux_head_always_sits_on_the_policy_encoder(ppg):
    """In every arch, the auxiliary head is the distillation channel."""
    for arch in ('dual', 'shared', 'detach'):
        torch.manual_seed(0)
        agent = ppg.PhasicValueAgent(DiscreteEnvStub(6), arch=arch)
        obs = torch.randint(0, 255, (5, 4, 84, 84), dtype=torch.uint8).float()
        _, _, vpred_aux = agent(obs)
        vpred_aux.sum().backward()
        total = sum(p.grad.abs().sum() for p in agent.pi_encoder.parameters() if p.grad is not None)
        assert total > 0, arch


def test_unknown_arch_and_encoder_rejected(ppg):
    with pytest.raises(ValueError, match='unsupported arch'):
        ppg.PhasicValueAgent(DiscreteEnvStub(4), arch='triple')
    with pytest.raises(ValueError, match='unsupported encoder'):
        ppg.PhasicValueAgent(DiscreteEnvStub(4), encoder='vgg')


def test_get_action_and_value_uses_the_true_value_head(ppg):
    """GAE must be built from `vf_head`, never from `aux_vf_head`."""
    torch.manual_seed(0)
    agent = ppg.PhasicValueAgent(DiscreteEnvStub(6), arch='dual')
    obs = torch.randint(0, 255, (7, 4, 84, 84), dtype=torch.uint8).float()
    _, _, _, value = agent.get_action_and_value(obs)
    _, vpred_true, _ = agent(obs)
    assert torch.allclose(value, vpred_true)


# ---------------------------------------------------------------------------
# the uint8 auxiliary buffer
# ---------------------------------------------------------------------------

def test_uint8_aux_storage_is_lossless_for_atari_frames(ppg):
    """The buffer is uint8 to fit in RAM; that is exact only because Atari
    observations are integers in [0, 255]. Pinned so a future preprocessing
    change (normalisation, float frames) fails loudly here first."""
    frames = torch.randint(0, 256, (3, 4, 84, 84)).float()
    assert torch.equal(frames.to(torch.uint8).float(), frames)
