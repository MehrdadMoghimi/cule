"""A2C + SIL: equivalence against junhyukoh/self-imitation-learning.

The reference is TensorFlow, so the objective is transcribed from
`baselines/common/self_imitation.py` with line numbers quoted. The prioritized
buffer is not: baselines' segment trees are pure Python and NumPy, so
`SumSegmentTree`/`MinSegmentTree` and `PrioritizedReplayBuffer.sample` are
loaded from the clone and diffed against this port's tree directly.

The five details singled out in the trainer header each get a test that would
fail if it were implemented the obvious way instead of the reference's way.
"""

import ast
import importlib.util
import os
import sys
import textwrap

import numpy as np
import pytest
import torch

from conftest import REPO_ROOT, DiscreteEnvStub, load_trainer

TRAINER = 'a2c_sil_atari'
UPSTREAM = os.path.join(REPO_ROOT, 'third_party', 'upstream', 'sil')


@pytest.fixture(scope='module')
def sil():
    return load_trainer(TRAINER)


@pytest.fixture(scope='module')
def upstream_segment_tree():
    """baselines' segment trees, loaded from the clone (no TF import needed)."""
    path = os.path.join(UPSTREAM, 'baselines', 'common', 'segment_tree.py')
    if not os.path.exists(path):
        pytest.skip('self-imitation-learning clone not present under third_party/upstream')
    spec = importlib.util.spec_from_file_location('_sil_segment_tree', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# 1. only episodes with a positive reward enter the buffer
# ---------------------------------------------------------------------------

def test_zero_reward_episodes_are_discarded(sil):
    """`update_buffer` gates on `r > 0`. SIL amplifies success; it cannot invent it."""
    imitation = sil.SelfImitation(num_envs=1, obs_shape=(4, 84, 84), capacity=1024)
    trajectory = [(np.zeros((4, 84, 84), np.uint8), 0, 0.0) for _ in range(20)]
    assert imitation.update_buffer(trajectory) is False
    assert len(imitation.buffer) == 0

    trajectory[7] = (np.zeros((4, 84, 84), np.uint8), 1, 1.0)
    assert imitation.update_buffer(trajectory) is True
    assert len(imitation.buffer) == 20


def test_negative_reward_alone_does_not_qualify(sil):
    """`r > 0`, strictly. A run of clipped -1s is not a success story."""
    imitation = sil.SelfImitation(num_envs=1, obs_shape=(4, 84, 84), capacity=1024)
    trajectory = [(np.zeros((4, 84, 84), np.uint8), 0, -1.0) for _ in range(5)]
    assert imitation.update_buffer(trajectory) is False
    assert len(imitation.buffer) == 0


def test_stored_returns_are_monte_carlo_not_bootstrapped(sil):
    """`discount_with_dones(rewards, dones, gamma)` with the last done True."""
    gamma = 0.9
    imitation = sil.SelfImitation(num_envs=1, obs_shape=(2,), capacity=64, gamma=gamma,
                                  obs_dtype=np.float32)
    rewards = [0.0, 0.0, 1.0]
    trajectory = [(np.zeros(2, np.float32), 0, r) for r in rewards]
    imitation.update_buffer(trajectory)

    # R_2 = 1 ; R_1 = 0.9 ; R_0 = 0.81 -- and no bootstrap past the end.
    assert np.allclose(imitation.buffer.returns[:3], [0.81, 0.9, 1.0], atol=1e-6)


def test_step_splits_episodes_per_environment(sil):
    """Each env keeps its own running trajectory; a done flushes only that one."""
    imitation = sil.SelfImitation(num_envs=3, obs_shape=(2,), capacity=256, obs_dtype=np.float32)
    observations = np.zeros((3, 2), np.float32)
    for _ in range(4):
        imitation.step(observations, np.zeros(3, np.int64), np.array([1.0, 0.0, 0.0]),
                       np.array([False, False, False]))
    imitation.step(observations, np.zeros(3, np.int64), np.array([1.0, 0.0, 0.0]),
                   np.array([True, False, False]))

    assert len(imitation.buffer) == 5      # env 0's episode, which scored
    assert imitation.running_episodes[0] == []
    assert len(imitation.running_episodes[1]) == 5
    assert len(imitation.running_episodes[2]) == 5


# ---------------------------------------------------------------------------
# 2. the normaliser is max(#valid, min_batch_size)
# ---------------------------------------------------------------------------

def upstream_sil_loss(logprobs, entropies, values, returns, weights,
                      clip, max_nlogp, min_batch_size, w_value, w_entropy):
    """Transcription of `self_imitation.py::build_loss_op`, lines 310-338."""
    nlogp = -logprobs
    mask = (returns - values > 0.0).to(values.dtype)
    num_valid_samples = mask.sum()
    num_samples = torch.clamp(num_valid_samples, min=float(min_batch_size))

    clipped_nlogp = (torch.minimum(nlogp, torch.tensor(max_nlogp, dtype=nlogp.dtype))
                     - nlogp).detach() + nlogp
    adv = torch.clamp(returns - values, 0.0, clip).detach()
    pg_loss = (weights * adv * clipped_nlogp).sum() / num_samples
    entropy = (weights * entropies * mask).sum() / num_samples
    loss = pg_loss - entropy * w_entropy

    delta = torch.clamp(values - returns, -clip, 0) * mask
    vf_loss = (weights * values * delta.detach()).sum() / num_samples
    return loss + 0.5 * w_value * vf_loss, adv, num_valid_samples


@pytest.mark.parametrize('seed', range(6))
def test_sil_loss_matches_upstream(sil, seed):
    generator = torch.Generator().manual_seed(seed)
    batch = 512
    logprobs = (torch.randn(batch, generator=generator, dtype=torch.float64) - 2.0).requires_grad_(True)
    entropies = torch.rand(batch, generator=generator, dtype=torch.float64)
    values = torch.randn(batch, generator=generator, dtype=torch.float64).requires_grad_(True)
    returns = torch.randn(batch, generator=generator, dtype=torch.float64)
    weights = torch.rand(batch, generator=generator, dtype=torch.float64) + 0.5

    got, advantage, num_valid = sil.sil_losses(
        logprobs, entropies, values, returns, weights,
        clip=1.0, max_nlogp=5.0, min_batch_size=64, w_value=0.01, w_entropy=0.01)
    want, want_adv, want_valid = upstream_sil_loss(
        logprobs, entropies, values, returns, weights,
        clip=1.0, max_nlogp=5.0, min_batch_size=64, w_value=0.01, w_entropy=0.01)

    assert np.isclose(got.item(), want.item(), rtol=0, atol=1e-12)
    assert torch.allclose(advantage, want_adv, rtol=0, atol=1e-14)
    assert num_valid.item() == want_valid.item()

    got_grads = torch.autograd.grad(got, [logprobs, values], retain_graph=True)
    want_grads = torch.autograd.grad(want, [logprobs, values], retain_graph=True)
    for a, b in zip(got_grads, want_grads):
        assert torch.allclose(a, b, rtol=0, atol=1e-13)


def test_normaliser_floors_at_min_batch_size(sil):
    """One lucky transition in a big batch must not be as loud as four hundred."""
    batch = 512
    values = torch.zeros(batch, dtype=torch.float64)
    returns = torch.zeros(batch, dtype=torch.float64)
    returns[0] = 1.0                     # exactly one valid sample
    logprobs = torch.full((batch,), -1.0, dtype=torch.float64)
    entropies = torch.zeros(batch, dtype=torch.float64)
    weights = torch.ones(batch, dtype=torch.float64)

    loss, _, num_valid = sil.sil_losses(
        logprobs, entropies, values, returns, weights,
        clip=1.0, max_nlogp=5.0, min_batch_size=64, w_value=0.0, w_entropy=0.0)
    assert num_valid.item() == 1
    # pg = W * adv * nlogp / max(1, 64) = 1 * 1 * 1 / 64
    assert np.isclose(loss.item(), 1.0 / 64, atol=1e-12)

    # Dividing by the count of valid samples would give 1.0 -- 64x louder.
    assert not np.isclose(loss.item(), 1.0, atol=1e-6)


def test_normaliser_uses_the_count_once_it_exceeds_the_floor(sil):
    batch = 512
    values = torch.zeros(batch, dtype=torch.float64)
    returns = torch.zeros(batch, dtype=torch.float64)
    returns[:128] = 1.0
    logprobs = torch.full((batch,), -1.0, dtype=torch.float64)
    entropies = torch.zeros(batch, dtype=torch.float64)
    weights = torch.ones(batch, dtype=torch.float64)

    loss, _, num_valid = sil.sil_losses(
        logprobs, entropies, values, returns, weights,
        clip=1.0, max_nlogp=5.0, min_batch_size=64, w_value=0.0, w_entropy=0.0)
    assert num_valid.item() == 128
    assert np.isclose(loss.item(), 128.0 / 128.0, atol=1e-12)


def test_a_batch_with_nothing_to_learn_from_is_silent(sil):
    """If the critic already dominates every return, SIL contributes no gradient."""
    batch = 256
    values = torch.ones(batch, dtype=torch.float64, requires_grad=True)
    returns = torch.zeros(batch, dtype=torch.float64)
    logprobs = torch.full((batch,), -1.0, dtype=torch.float64, requires_grad=True)
    entropies = torch.rand(batch, dtype=torch.float64)
    weights = torch.ones(batch, dtype=torch.float64)

    loss, advantage, num_valid = sil.sil_losses(
        logprobs, entropies, values, returns, weights,
        clip=1.0, max_nlogp=5.0, min_batch_size=64, w_value=0.01, w_entropy=0.01)
    assert num_valid.item() == 0
    assert torch.count_nonzero(advantage) == 0
    loss.backward()
    assert torch.count_nonzero(logprobs.grad) == 0
    assert torch.count_nonzero(values.grad) == 0


# ---------------------------------------------------------------------------
# 3. the value surrogate
# ---------------------------------------------------------------------------

def test_value_surrogate_gradient_is_the_clipped_regression_gradient(sil):
    """`d/dV [W * V * stop_grad(delta)] = W * delta`, with delta clipped to [-1, 0].

    That is the gradient of `0.5 (V - R)^2` where `R > V`, capped at magnitude
    `clip`. Implementing it as a literal squared error diverges as soon as the
    gap exceeds `clip`.
    """
    values = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64, requires_grad=True)
    returns = torch.tensor([0.5, 5.0, -1.0], dtype=torch.float64)  # gap .5, 5, none
    logprobs = torch.zeros(3, dtype=torch.float64)
    entropies = torch.zeros(3, dtype=torch.float64)
    weights = torch.ones(3, dtype=torch.float64)

    loss, _, _ = sil.sil_losses(
        logprobs, entropies, values, returns, weights,
        clip=1.0, max_nlogp=5.0, min_batch_size=1, w_value=1.0, w_entropy=0.0)
    loss.backward()

    # 0.5 * w_value * W * delta / num_samples, num_samples = max(2, 1) = 2
    # delta = clip(V - R, -1, 0) * mask = [-0.5, -1.0, 0.0]
    expected = 0.5 * 1.0 * np.array([-0.5, -1.0, 0.0]) / 2.0
    assert np.allclose(values.grad.numpy(), expected, atol=1e-12)

    # An unclipped squared error would give -1 * 5 = -5 in slot 1, not -1.
    assert not np.isclose(values.grad[1].item(), 0.5 * (-5.0) / 2.0, atol=1e-6)


def test_value_surrogate_only_pushes_the_value_up(sil):
    """delta is clipped to `[-clip, 0]`, so SIL never lowers a value estimate."""
    values = torch.tensor([10.0, -10.0], dtype=torch.float64, requires_grad=True)
    returns = torch.tensor([0.0, 0.0], dtype=torch.float64)
    loss, _, _ = sil.sil_losses(
        torch.zeros(2, dtype=torch.float64), torch.zeros(2, dtype=torch.float64),
        values, returns, torch.ones(2, dtype=torch.float64),
        clip=1.0, max_nlogp=5.0, min_batch_size=1, w_value=1.0, w_entropy=0.0)
    loss.backward()
    # Overestimating value (10 vs 0) is masked out entirely; underestimating
    # (-10 vs 0) yields a negative gradient, i.e. gradient descent raises V.
    assert values.grad[0].item() == 0.0
    assert values.grad[1].item() < 0.0


# ---------------------------------------------------------------------------
# 4. the gradient-transparent nlogp clip
# ---------------------------------------------------------------------------

def test_clipped_neg_logp_clips_value_but_not_gradient(sil):
    neg_logp = torch.tensor([1.0, 7.0, 20.0], dtype=torch.float64, requires_grad=True)
    clipped = sil.clipped_neg_logp(neg_logp, 5.0)
    assert np.allclose(clipped.detach().numpy(), [1.0, 5.0, 5.0], atol=1e-12)

    clipped.sum().backward()
    # The gradient is 1 everywhere, including where the clip bit.
    assert np.allclose(neg_logp.grad.numpy(), [1.0, 1.0, 1.0], atol=1e-12)


def test_naive_clamp_would_have_killed_the_gradient(sil):
    """The trap this construction exists to avoid."""
    neg_logp = torch.tensor([7.0], dtype=torch.float64, requires_grad=True)
    torch.clamp(neg_logp, max=5.0).sum().backward()
    assert neg_logp.grad.item() == 0.0  # gone: the rare action teaches nothing

    neg_logp2 = torch.tensor([7.0], dtype=torch.float64, requires_grad=True)
    sil.clipped_neg_logp(neg_logp2, 5.0).sum().backward()
    assert neg_logp2.grad.item() == 1.0


# ---------------------------------------------------------------------------
# 5. priorities
# ---------------------------------------------------------------------------

def test_priority_is_the_clipped_advantage(sil):
    values = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64)
    returns = torch.tensor([0.3, 9.0, -2.0], dtype=torch.float64)
    _, advantage, _ = sil.sil_losses(
        torch.zeros(3, dtype=torch.float64), torch.zeros(3, dtype=torch.float64),
        values, returns, torch.ones(3, dtype=torch.float64),
        clip=1.0, max_nlogp=5.0, min_batch_size=1, w_value=0.0, w_entropy=0.0)
    assert np.allclose(advantage.numpy(), [0.3, 1.0, 0.0], atol=1e-12)


def test_priorities_are_floored_at_1e_minus_6(sil):
    """A zero priority would make the entry unreachable *and* break the tree."""
    buffer = sil.PrioritizedReplayBuffer(64, 0.6, (2,), np.float32)
    for i in range(4):
        buffer.add(np.zeros(2, np.float32), i, float(i))
    buffer.update_priorities([0, 1], [0.0, 0.5])
    assert buffer.tree[0] == pytest.approx(1e-6**0.6)
    assert buffer.tree[1] == pytest.approx(0.5**0.6)


def test_new_entries_get_the_running_max_priority(sil):
    buffer = sil.PrioritizedReplayBuffer(64, 0.6, (2,), np.float32)
    buffer.add(np.zeros(2, np.float32), 0, 0.0)
    assert buffer.tree[0] == pytest.approx(1.0**0.6)
    buffer.update_priorities([0], [4.0])
    buffer.add(np.zeros(2, np.float32), 1, 0.0)
    assert buffer.max_priority == 4.0
    assert buffer.tree[1] == pytest.approx(4.0**0.6)


# ---------------------------------------------------------------------------
# the segment tree, against baselines' own
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('capacity', [8, 16, 100, 1024])
def test_segment_tree_matches_baselines(sil, upstream_segment_tree, capacity):
    generator = np.random.default_rng(capacity)
    ours = sil.SumMinSegmentTree(capacity)
    their_sum = upstream_segment_tree.SumSegmentTree(ours.size)
    their_min = upstream_segment_tree.MinSegmentTree(ours.size)

    for index in range(capacity):
        value = float(generator.random() + 1e-3)
        ours[index] = value
        their_sum[index] = value
        their_min[index] = value

    assert np.isclose(ours.sum(), their_sum.sum(), rtol=0, atol=1e-12)
    assert np.isclose(ours.min(), their_min.min(), rtol=0, atol=1e-12)

    total = their_sum.sum()
    for _ in range(200):
        mass = generator.random() * total
        assert ours.find_prefixsum_idx(mass) == their_sum.find_prefixsum_idx(mass)


def test_segment_tree_capacity_is_rounded_up_to_a_power_of_two(sil):
    assert sil.SumMinSegmentTree(100).size == 128
    assert sil.SumMinSegmentTree(1024).size == 1024
    assert sil.SumMinSegmentTree(100000).size == 131072


def test_sampling_is_proportional_to_priority(sil):
    """The property the tree exists for, measured rather than asserted."""
    buffer = sil.PrioritizedReplayBuffer(8, 1.0, (1,), np.float32)
    for i in range(4):
        buffer.add(np.zeros(1, np.float32), i, 0.0)
    buffer.update_priorities([0, 1, 2, 3], [1.0, 1.0, 1.0, 9.0])

    generator = np.random.default_rng(0)
    _, actions, _, _, _ = buffer.sample(20000, beta=0.1, generator=generator)
    share = np.mean(actions == 3)
    assert 0.70 < share < 0.80  # 9 / 12


def test_importance_weights_are_normalised_by_their_maximum(sil):
    """`weight / max_weight`, so the largest weight in any batch is 1."""
    buffer = sil.PrioritizedReplayBuffer(16, 0.6, (1,), np.float32)
    for i in range(8):
        buffer.add(np.zeros(1, np.float32), i, 0.0)
    buffer.update_priorities(range(8), [0.1, 0.2, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0])

    generator = np.random.default_rng(1)
    _, _, _, weights, _ = buffer.sample(4000, beta=0.4, generator=generator)
    assert weights.max() <= 1.0 + 1e-6
    assert weights.min() > 0.0
    # beta=0 would flatten them all to 1; beta>0 must not.
    assert weights.min() < 0.9


def test_buffer_wraps_and_reports_bounded_size(sil):
    buffer = sil.PrioritizedReplayBuffer(4, 0.6, (1,), np.float32)
    for i in range(10):
        buffer.add(np.full(1, i, np.float32), i, float(i))
    assert len(buffer) == 4
    # ring order: indices 0..3 hold entries 8, 9, 6, 7
    assert list(buffer.actions) == [8, 9, 6, 7]


# ---------------------------------------------------------------------------
# the inherited A2C half is untouched
# ---------------------------------------------------------------------------

def test_a2c_half_is_identical_to_the_standalone_trainer(sil):
    """SIL must not perturb the on-policy update it sits on top of."""
    a2c = load_trainer('a2c_atari')
    envs = DiscreteEnvStub(6)

    torch.manual_seed(0)
    sil_agent = sil.Agent(envs)
    torch.manual_seed(0)
    a2c_agent = a2c.Agent(envs)
    for key, value in sil_agent.state_dict().items():
        assert torch.equal(value, a2c_agent.state_dict()[key]), key

    torch.manual_seed(1)
    num_steps, num_envs = 5, 16
    rewards = torch.randn(num_steps, num_envs, dtype=torch.float64)
    next_dones = (torch.rand(num_steps, num_envs) < 0.2).double()
    last_values = torch.randn(num_envs, dtype=torch.float64)
    assert torch.equal(
        sil.nstep_returns(rewards, next_dones, last_values, 0.99),
        a2c.nstep_returns(rewards, next_dones, last_values, 0.99))

    batch = 80
    args = [torch.randn(batch, dtype=torch.float64) for _ in range(5)]
    assert np.isclose(
        sil.a2c_losses(*args, 0.01, 0.5)[0].item(),
        a2c.a2c_losses(*args, 0.01, 0.5)[0].item(), rtol=0, atol=1e-14)


def test_rmsprop_tf_like_matches_the_standalone_trainer(sil):
    a2c = load_trainer('a2c_atari')
    torch.manual_seed(0)
    init = torch.randn(12, dtype=torch.float64)
    grads = [torch.randn(12, dtype=torch.float64) for _ in range(4)]

    results = []
    for factory in (lambda p: sil.RMSpropTFLike(p, lr=7e-4, alpha=0.99, eps=1e-5),
                    lambda p: a2c.RMSpropTFLike(p, lr=7e-4, alpha=0.99, eps=1e-5)):
        tensor = init.clone().requires_grad_(True)
        optimizer = factory([tensor])
        for grad in grads:
            tensor.grad = grad.clone()
            optimizer.step()
        results.append(tensor.detach().numpy().copy())
    assert np.allclose(results[0], results[1], rtol=0, atol=1e-15)
