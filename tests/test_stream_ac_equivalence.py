"""Stream AC(lambda): the actor-critic specifics, checked against the paper.

The streaming machinery it shares with stream_q_atari.py (ObGD, SparseInit,
LayerNorm, trace independence) is already covered by
`tests/test_stream_equivalence.py`; these tests pin what stream AC changes.
streaming-drl is CC BY-NC 4.0, so as there, nothing is vendored and the checks
are against the published update rule.
"""

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from conftest import load_trainer

TRAINERS = ['stream_ac_atari', 'stream_ac_atari_torchcompile']
N_ACTIONS = 5


@pytest.fixture(params=TRAINERS)
def module(request):
    return load_trainer(request.param)


def test_actor_and_critic_have_separate_trunks(module):
    """Sharing a trunk would make the two eligibility traces interfere."""
    policy = module.StreamNetwork(N_ACTIONS)
    value = module.StreamNetwork(1)
    assert policy.network[-1].out_features == N_ACTIONS
    assert value.network[-1].out_features == 1
    policy_ids = {id(p) for p in policy.parameters()}
    value_ids = {id(p) for p in value.parameters()}
    assert policy_ids.isdisjoint(value_ids)


def test_policy_objective_matches_paper_formula(module):
    """d(-log pi(a|s) - c * sign(delta) * H(pi))/d(theta), per stream."""
    torch.manual_seed(1)
    policy = module.StreamNetwork(N_ACTIONS).double()
    per_stream_policy_grad = module.make_per_stream_policy_grad_fn(policy)

    observations = torch.randn(3, 4, 84, 84, dtype=torch.float64)
    actions = torch.tensor([0, 4, 2])
    sign_delta = torch.tensor([1.0, -1.0, 1.0], dtype=torch.float64)
    entropy_coeff = 0.01

    params = {name: p.detach() for name, p in policy.named_parameters()}
    batched = per_stream_policy_grad(
        params, observations, F.one_hot(actions, N_ACTIONS).double(), sign_delta, entropy_coeff)

    for stream in range(3):
        policy.zero_grad()
        logits = policy(observations[stream])
        distribution = torch.distributions.Categorical(logits=logits)
        objective = (
            -distribution.log_prob(actions[stream])
            - entropy_coeff * distribution.entropy() * sign_delta[stream]
        )
        objective.backward()
        for name, parameter in policy.named_parameters():
            torch.testing.assert_close(batched[name][stream], parameter.grad, rtol=1e-9, atol=1e-9)


def test_value_objective_is_negative_v(module):
    torch.manual_seed(2)
    value = module.StreamNetwork(1).double()
    per_stream_value_grad = module.make_per_stream_value_grad_fn(value)

    observations = torch.randn(2, 4, 84, 84, dtype=torch.float64)
    params = {name: p.detach() for name, p in value.named_parameters()}
    batched = per_stream_value_grad(params, observations)

    for stream in range(2):
        value.zero_grad()
        (-value(observations[stream]).squeeze(-1)).backward()
        for name, parameter in value.named_parameters():
            torch.testing.assert_close(batched[name][stream], parameter.grad, rtol=1e-9, atol=1e-9)


def test_entropy_bonus_direction_follows_sign_of_delta(module):
    """ObGD multiplies by delta, so the bonus contributes |delta| * c * grad(H).

    That is what makes the bonus vanish as the critic stops being surprised; if
    the sign were dropped the bonus would flip direction on negative TD errors.
    """
    torch.manual_seed(6)
    policy = module.StreamNetwork(N_ACTIONS).double()
    per_stream_policy_grad = module.make_per_stream_policy_grad_fn(policy)
    observations = torch.randn(1, 4, 84, 84, dtype=torch.float64)
    onehot = F.one_hot(torch.tensor([1]), N_ACTIONS).double()
    params = {name: p.detach() for name, p in policy.named_parameters()}

    positive = per_stream_policy_grad(
        params, observations, onehot, torch.tensor([1.0], dtype=torch.float64), 0.5)
    negative = per_stream_policy_grad(
        params, observations, onehot, torch.tensor([-1.0], dtype=torch.float64), 0.5)
    neutral = per_stream_policy_grad(
        params, observations, onehot, torch.tensor([0.0], dtype=torch.float64), 0.5)

    for name in params:
        entropy_term = neutral[name] - positive[name]
        # The +1 and -1 cases straddle the no-entropy case symmetrically.
        torch.testing.assert_close(negative[name] - neutral[name], entropy_term,
                                   rtol=1e-9, atol=1e-9)
        # And the entropy term is actually doing something.
        if entropy_term.abs().sum() > 0:
            break
    else:
        pytest.fail('entropy bonus had no effect on any parameter')


def test_actor_and_critic_use_different_kappas(module):
    args = module.Args()
    assert args.kappa_policy == 3.0
    assert args.kappa_value == 2.0
    assert args.entropy_coeff == 0.01
    assert args.learning_rate == 1.0
    assert args.gamma == 0.99
    assert args.lamda == 0.8
    assert args.sparsity == 0.9
    assert args.num_envs == 1, 'the published algorithm is single-stream'
    assert not hasattr(args, 'kappa'), 'stream AC replaces the single kappa with two'
    # On-policy: there is no epsilon schedule to inherit from stream Q.
    assert not hasattr(args, 'start_e')
    assert not hasattr(args, 'exploration_fraction')


def test_kappa_changes_the_step_size(module):
    """A larger kappa tightens the bound, so the actor takes smaller steps."""
    steps = []
    for kappa in (2.0, 3.0):
        parameter = torch.nn.Parameter(torch.zeros(4, dtype=torch.float64))
        optimizer = module.ObGD([parameter], num_streams=1, lr=1.0, gamma=0.0,
                                lamda=0.0, kappa=kappa)
        steps.append(optimizer.step([torch.ones(1, 4, dtype=torch.float64)],
                                    torch.tensor([3.0], dtype=torch.float64),
                                    torch.tensor([False])).item())
    assert steps[1] < steps[0]
    np.testing.assert_allclose(steps[0] / steps[1], 3.0 / 2.0, rtol=1e-12)


def test_gumbel_max_sampling_matches_categorical():
    """The compiled twin swaps Categorical.sample() for Gumbel-max.

    They must be the same distribution, or the two variants would not be the
    same algorithm.
    """
    torch.manual_seed(0)
    logits = torch.tensor([[0.5, -1.0, 2.0, 0.0, 0.25]]).repeat(60000, 1)

    uniform = torch.rand_like(logits).clamp_min(1e-10)
    gumbel = -torch.log((-torch.log(uniform)).clamp_min(1e-10))
    gumbel_samples = torch.argmax(logits + gumbel, dim=-1)

    expected = torch.softmax(logits[0], dim=-1)
    observed = torch.bincount(gumbel_samples, minlength=5).double() / gumbel_samples.numel()
    np.testing.assert_allclose(observed.numpy(), expected.numpy(), atol=0.01)


def test_both_variants_define_the_same_networks():
    eager, compiled = (load_trainer(name) for name in TRAINERS)
    torch.manual_seed(9)
    a = eager.StreamNetwork(N_ACTIONS)
    torch.manual_seed(9)
    b = compiled.StreamNetwork(N_ACTIONS)
    x = torch.randn(3, 4, 84, 84)
    torch.testing.assert_close(a(x), b(x), rtol=0, atol=0)
