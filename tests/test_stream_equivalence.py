"""Stream Q(lambda): the port checked against the paper's equations.

streaming-drl is CC BY-NC 4.0, so there is no vendored upstream source to diff
against.  Instead these tests pin the things a port can get wrong:

  * the ObGD update, against an independent NumPy transcription of Algorithm 3;
  * ObGD's overshooting bound, as an empirical property rather than a formula;
  * SparseInit's zero count and sampling bound;
  * that the trace is Watkins-cut on termination *and* on non-greedy actions;
  * that N-stream vectorisation reduces to the published single-stream algorithm
    at N = 1 and never mixes streams at N > 1.
"""

import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from conftest import DiscreteEnvStub, load_trainer
from reference.obgd_reference import ObGDReference, sparse_init_bound, sparse_init_zero_count

TRAINERS = ['stream_q_atari', 'stream_q_atari_torchcompile']


@pytest.fixture(params=TRAINERS)
def module(request):
    return load_trainer(request.param)


def test_obgd_matches_paper_equations(module):
    """One stream, many steps, against the NumPy transcription of Algorithm 3."""
    torch.manual_seed(0)
    params = [torch.randn(4, 3, dtype=torch.float64), torch.randn(5, dtype=torch.float64)]
    ours = module.ObGD([torch.nn.Parameter(p.clone()) for p in params],
                       num_streams=1, lr=1.0, gamma=0.99, lamda=0.8, kappa=2.0)
    theirs = ObGDReference([p.numpy() for p in params], lr=1.0, gamma=0.99, lamda=0.8, kappa=2.0)

    rng = np.random.default_rng(0)
    for step in range(12):
        grads = [rng.normal(size=(4, 3)), rng.normal(size=(5,))]
        delta = float(rng.normal() * 3)
        reset = step in (4, 9)

        our_step = ours.step(
            [torch.tensor(g, dtype=torch.float64).unsqueeze(0) for g in grads],
            torch.tensor([delta], dtype=torch.float64),
            torch.tensor([reset]),
        )
        their_step = theirs.step(grads, delta, reset)

        np.testing.assert_allclose(our_step.item(), their_step, rtol=1e-12, atol=1e-12)
        for ours_p, theirs_p in zip(ours.params, theirs.params):
            np.testing.assert_allclose(ours_p.detach().numpy(), theirs_p, rtol=1e-11, atol=1e-11)
        for our_trace, their_trace in zip(ours.traces, theirs.traces):
            np.testing.assert_allclose(our_trace[0].numpy(), their_trace, rtol=1e-11, atol=1e-11)


def test_obgd_step_size_is_the_overshooting_bound(module):
    """alpha_t = alpha / (max(|delta|,1) * ||z||_1 * alpha * kappa) when that exceeds 1."""
    parameter = torch.nn.Parameter(torch.zeros(3, dtype=torch.float64))
    optimizer = module.ObGD([parameter], num_streams=1, lr=1.0, gamma=0.0, lamda=0.0, kappa=2.0)

    # gamma*lamda = 0, so the trace is exactly the gradient.
    gradient = torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float64)
    delta = torch.tensor([5.0], dtype=torch.float64)
    step_size = optimizer.step([gradient], delta, torch.tensor([False]))
    expected = 1.0 / (5.0 * 3.0 * 1.0 * 2.0)
    np.testing.assert_allclose(step_size.item(), expected, rtol=1e-12)

    # A tiny trace leaves the bound inactive and the step size at lr.
    optimizer = module.ObGD([torch.nn.Parameter(torch.zeros(3, dtype=torch.float64))],
                            num_streams=1, lr=1.0, gamma=0.0, lamda=0.0, kappa=2.0)
    step_size = optimizer.step([gradient * 1e-3], delta, torch.tensor([False]))
    np.testing.assert_allclose(step_size.item(), 1.0, rtol=1e-12)


def test_obgd_does_not_overshoot_the_td_error(module):
    """The bound's purpose: one update must not flip the sign of the TD error.

    This is the property the paper proves for ObGD and is why it can run at
    lr = 1 with no replay buffer; a step-size bug would show up here even if the
    formula looked right.
    """
    torch.manual_seed(3)
    weight = torch.nn.Parameter(torch.randn(6, dtype=torch.float64))
    optimizer = module.ObGD([weight], num_streams=1, lr=1.0, gamma=0.99, lamda=0.8, kappa=2.0)

    features = torch.randn(6, dtype=torch.float64)
    for _ in range(50):
        target = torch.tensor(4.0, dtype=torch.float64)
        value = weight @ features
        delta = target - value
        # d(-value)/d(weight), as the streaming update backprops -Q(s, a).
        optimizer.step([(-features).unsqueeze(0)], delta.unsqueeze(0), torch.tensor([False]))
        new_delta = target - weight @ features
        assert torch.sign(new_delta) == torch.sign(delta), 'ObGD overshot the TD target'


def test_sparse_init_zero_count_and_bound(module):
    for shape, sparsity in [((64, 128), 0.9), ((32, 16, 3, 3), 0.9), ((10, 20), 0.5)]:
        tensor = torch.empty(*shape, dtype=torch.float64)
        module.sparse_init(tensor, sparsity)
        flat = tensor.reshape(shape[0], -1)
        fan_in = flat.shape[1]
        expected_zeros = sparse_init_zero_count(fan_in, sparsity)
        # Every output unit is masked independently, to the same count.
        zeros_per_unit = (flat == 0).sum(dim=1)
        assert torch.all(zeros_per_unit == expected_zeros), zeros_per_unit
        assert flat.abs().max().item() <= sparse_init_bound(fan_in)


def test_sparse_init_masks_differ_between_units(module):
    """Units must not share one mask, or the effective input set would collapse."""
    tensor = torch.empty(32, 200, dtype=torch.float64)
    module.sparse_init(tensor, 0.9)
    masks = {tuple(torch.nonzero(row).flatten().tolist()) for row in tensor}
    assert len(masks) > 1


def test_network_shape_and_size(module):
    net = module.QNetwork(DiscreteEnvStub(6))
    # Strides 5/3/2 take 84 -> 16 -> 5 -> 2, i.e. 64*2*2 = 256 features.
    assert net.network[10].in_features == 256
    batched = net(torch.zeros(7, 4, 84, 84))
    unbatched = net(torch.zeros(4, 84, 84))
    assert batched.shape == (7, 6) and unbatched.shape == (6,)
    # Small enough that one eligibility trace per stream is cheap.
    assert sum(p.numel() for p in net.parameters()) < 200_000


def test_layer_normalization_is_parameter_free(module):
    norm = module.LayerNormalization(3)
    assert list(norm.parameters()) == []
    x = torch.randn(5, 8, 4, 4, dtype=torch.float64)
    out = norm(x)
    # Statistics are taken per sample over (C, H, W), matching the paper's
    # whole-activation normalisation.
    np.testing.assert_allclose(out.flatten(1).mean(1).numpy(), 0.0, atol=1e-9)
    np.testing.assert_allclose(out.flatten(1).std(1, unbiased=False).numpy(), 1.0, atol=1e-4)
    # Batched and unbatched must agree.
    torch.testing.assert_close(norm(x)[0], norm(x[0]), rtol=0, atol=0)


def test_per_stream_gradients_match_individual_autograd(module):
    """vmap(grad(...)) must equal a per-stream backward, or the traces are wrong."""
    torch.manual_seed(5)
    net = module.QNetwork(DiscreteEnvStub(4)).double()
    per_stream_grad = module.make_per_stream_grad_fn(net)

    observations = torch.randn(3, 4, 84, 84, dtype=torch.float64)
    actions = torch.tensor([0, 3, 1])
    params = {name: p.detach() for name, p in net.named_parameters()}
    batched = per_stream_grad(params, observations, F.one_hot(actions, 4).double())

    for stream in range(3):
        net.zero_grad()
        (-net(observations[stream])[actions[stream]]).backward()
        for name, parameter in net.named_parameters():
            torch.testing.assert_close(
                batched[name][stream], parameter.grad, rtol=1e-9, atol=1e-9)


def test_streams_are_independent(module):
    """Stream 1's transitions must not touch stream 0's eligibility trace."""
    params = [torch.nn.Parameter(torch.zeros(4, dtype=torch.float64))]
    optimizer = module.ObGD(params, num_streams=2, lr=1.0, gamma=0.99, lamda=0.8, kappa=2.0)
    grads = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=torch.float64)
    optimizer.step([grads], torch.tensor([1.0, 1.0], dtype=torch.float64), torch.tensor([False, False]))
    # Reset only stream 0; stream 1's trace must survive untouched.
    optimizer.step([torch.zeros_like(grads)],
                   torch.tensor([0.0, 0.0], dtype=torch.float64),
                   torch.tensor([True, False]))
    trace = optimizer.traces[0]
    assert torch.all(trace[0] == 0)
    np.testing.assert_allclose(trace[1].numpy(), [0.0, 0.99 * 0.8, 0.0, 0.0], rtol=1e-12)


def test_single_stream_reductions_agree(module):
    """`mean` and `sum` differ only when there is more than one stream."""
    for reduction in ('mean', 'sum'):
        parameter = torch.nn.Parameter(torch.zeros(3, dtype=torch.float64))
        optimizer = module.ObGD([parameter], num_streams=1, lr=1.0, gamma=0.99,
                                lamda=0.8, kappa=2.0, reduction=reduction)
        optimizer.step([torch.ones(1, 3, dtype=torch.float64)],
                       torch.tensor([0.5], dtype=torch.float64), torch.tensor([False]))
        np.testing.assert_allclose(parameter.detach().numpy(),
                                   [-0.5 / 6, -0.5 / 6, -0.5 / 6], rtol=1e-12)


def test_hyperparameters_match_paper(module):
    args = module.Args()
    assert args.learning_rate == 1.0
    assert args.gamma == 0.99
    assert args.lamda == 0.8
    assert args.kappa == 2.0
    assert args.sparsity == 0.9
    assert args.hidden_size == 256
    assert args.exploration_fraction == 0.05
    assert args.start_e == 1.0 and args.end_e == 0.01
    assert args.num_envs == 1, 'the published algorithm is single-stream'
    # The paper scales rewards by the running return std instead of clipping.
    assert args.clip_rewards is False
    assert args.scale_rewards is True
    assert args.normalize_observations is True


def test_both_variants_agree():
    """The compiled twin must define the same network and optimizer maths."""
    eager, compiled = (load_trainer(name) for name in TRAINERS)
    torch.manual_seed(8)
    a = eager.QNetwork(DiscreteEnvStub(5))
    torch.manual_seed(8)
    b = compiled.QNetwork(DiscreteEnvStub(5))
    x = torch.randn(3, 4, 84, 84)
    torch.testing.assert_close(a(x), b(x), rtol=0, atol=0)

    grads = [torch.randn(2, 6, dtype=torch.float64)]
    delta = torch.tensor([1.5, -0.4], dtype=torch.float64)
    reset = torch.tensor([False, True])
    results = []
    for module in (eager, compiled):
        parameter = torch.nn.Parameter(torch.zeros(6, dtype=torch.float64))
        optimizer = module.ObGD([parameter], num_streams=2)
        step = optimizer.step([g.clone() for g in grads], delta.clone(), reset.clone())
        results.append((parameter.detach().clone(), step, optimizer.traces[0].clone()))
    for left, right in zip(results[0], results[1]):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
