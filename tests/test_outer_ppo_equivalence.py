"""Outer-PPO: reduction to PPO, and the outer update's arithmetic.

There is no released reference implementation for this paper, so nothing here is
a transcription. The load-bearing test is instead the *reduction*: at
`outer_lr = 1, outer_momentum = 0` a full training step must produce parameters
bit-identical to plain PPO's, tensor for tensor. An improvement that cannot
reproduce its own baseline exactly is not measurable against it.

Everything else pins the update algebra: the applied step, the momentum
recursion, and the fact that the inner optimiser's state survives the rewrite.
"""

import numpy as np
import pytest
import torch
import torch.nn as nn

from conftest import DiscreteEnvStub, load_trainer

TRAINER = 'outer_ppo_atari'


@pytest.fixture(scope='module')
def outer_ppo():
    return load_trainer(TRAINER)


def tiny_model(seed=0):
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(6, 5), nn.Tanh(), nn.Linear(5, 3)).double()


def run_inner_loop(model, steps=4, seed=1, lr=0.05):
    """A stand-in for PPO's epochs: whatever it does, only the endpoints matter."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-5)
    generator = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        x = torch.randn(8, 6, generator=generator, dtype=torch.float64)
        target = torch.randn(8, 3, generator=generator, dtype=torch.float64)
        loss = ((model(x) - target) ** 2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return optimizer


# ---------------------------------------------------------------------------
# the reduction to PPO
# ---------------------------------------------------------------------------

def test_identity_settings_are_a_bit_exact_no_op(outer_ppo):
    baseline = tiny_model()
    run_inner_loop(baseline)

    model = tiny_model()
    outer = outer_ppo.OuterUpdate(model.parameters(), outer_lr=1.0, outer_momentum=0.0)
    outer.snapshot()
    run_inner_loop(model)
    outer.apply()

    for a, b in zip(model.parameters(), baseline.parameters()):
        assert torch.equal(a, b)


def test_is_identity_flag_matches_the_settings(outer_ppo):
    model = tiny_model()
    assert outer_ppo.OuterUpdate(model.parameters(), 1.0, 0.0).is_identity
    assert not outer_ppo.OuterUpdate(model.parameters(), 1.5, 0.0).is_identity
    assert not outer_ppo.OuterUpdate(model.parameters(), 1.0, 0.9).is_identity


def test_identity_settings_stay_exact_over_many_iterations(outer_ppo):
    """Drift would show up cumulatively, so run the whole loop repeatedly."""
    baseline = tiny_model()
    model = tiny_model()
    outer = outer_ppo.OuterUpdate(model.parameters(), outer_lr=1.0, outer_momentum=0.0)

    baseline_optimizer = torch.optim.Adam(baseline.parameters(), lr=0.05, eps=1e-5)
    model_optimizer = torch.optim.Adam(model.parameters(), lr=0.05, eps=1e-5)
    for iteration in range(6):
        generator_seed = 100 + iteration
        for target_model, optimizer, wrapper in (
            (baseline, baseline_optimizer, None),
            (model, model_optimizer, outer),
        ):
            if wrapper is not None:
                wrapper.snapshot()
            generator = torch.Generator().manual_seed(generator_seed)
            for _ in range(3):
                x = torch.randn(8, 6, generator=generator, dtype=torch.float64)
                target = torch.randn(8, 3, generator=generator, dtype=torch.float64)
                loss = ((target_model(x) - target) ** 2).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            if wrapper is not None:
                wrapper.apply()

    for a, b in zip(model.parameters(), baseline.parameters()):
        assert torch.equal(a, b)


# ---------------------------------------------------------------------------
# the outer step
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('outer_lr', [0.5, 1.5, 3.0])
def test_outer_lr_scales_the_update_vector(outer_ppo, outer_lr):
    reference = tiny_model()
    start = [p.detach().clone() for p in reference.parameters()]
    run_inner_loop(reference)
    deltas = [p.detach() - s for p, s in zip(reference.parameters(), start)]

    model = tiny_model()
    outer = outer_ppo.OuterUpdate(model.parameters(), outer_lr=outer_lr, outer_momentum=0.0)
    outer.snapshot()
    run_inner_loop(model)
    outer.apply()

    for parameter, s, delta in zip(model.parameters(), start, deltas):
        assert torch.allclose(parameter, s + outer_lr * delta, rtol=0, atol=1e-13)


def test_momentum_follows_the_nesterov_recursion(outer_ppo):
    """`step = delta + m * buf`, then `buf <- m * buf + delta`."""
    momentum, outer_lr = 0.9, 1.0
    model = tiny_model()
    outer = outer_ppo.OuterUpdate(model.parameters(), outer_lr=outer_lr, outer_momentum=momentum)

    buffers = [torch.zeros_like(p) for p in model.parameters()]
    expected = [p.detach().clone() for p in model.parameters()]

    for iteration in range(4):
        starts = [p.detach().clone() for p in model.parameters()]
        outer.snapshot()
        run_inner_loop(model, steps=2, seed=iteration)
        deltas = [p.detach() - s for p, s in zip(model.parameters(), starts)]
        outer.apply()

        for index in range(len(buffers)):
            step = deltas[index] + momentum * buffers[index]
            buffers[index] = momentum * buffers[index] + deltas[index]
            expected[index] = starts[index] + outer_lr * step

    for parameter, want in zip(model.parameters(), expected):
        assert torch.allclose(parameter, want, rtol=0, atol=1e-12)


def test_momentum_accumulates_across_iterations(outer_ppo):
    """Two identical inner updates in a row must produce a larger second step."""
    momentum = 0.9
    model = tiny_model()
    outer = outer_ppo.OuterUpdate(model.parameters(), outer_lr=1.0, outer_momentum=momentum)
    parameter = next(iter(model.parameters()))

    magnitudes = []
    for _ in range(3):
        before = parameter.detach().clone()
        outer.snapshot()
        with torch.no_grad():
            for p in model.parameters():
                p.add_(torch.full_like(p, 0.01))  # a fixed inner update
        outer.apply()
        magnitudes.append((parameter.detach() - before).abs().mean().item())

    assert magnitudes[1] > magnitudes[0]
    assert magnitudes[2] > magnitudes[1]
    # First step is delta alone; second is delta * (1 + m).
    assert np.isclose(magnitudes[1] / magnitudes[0], 1 + momentum, rtol=1e-6)


def test_no_momentum_buffer_is_allocated_when_momentum_is_zero(outer_ppo):
    model = tiny_model()
    assert outer_ppo.OuterUpdate(model.parameters(), 1.5, 0.0).momentum_buffer is None
    assert outer_ppo.OuterUpdate(model.parameters(), 1.5, 0.9).momentum_buffer is not None


def test_apply_without_snapshot_is_an_error(outer_ppo):
    model = tiny_model()
    outer = outer_ppo.OuterUpdate(model.parameters(), 1.5, 0.0)
    with pytest.raises(RuntimeError, match='snapshot'):
        outer.apply()


def test_snapshot_is_released_after_apply(outer_ppo):
    """The snapshot doubles parameter memory; it must not be held between steps."""
    model = tiny_model()
    outer = outer_ppo.OuterUpdate(model.parameters(), 1.5, 0.0)
    outer.snapshot()
    assert outer.snapshot_values is not None
    outer.apply()
    assert outer.snapshot_values is None

    identity = outer_ppo.OuterUpdate(model.parameters(), 1.0, 0.0)
    identity.snapshot()
    identity.apply()
    assert identity.snapshot_values is None


def test_outer_update_does_not_disturb_the_inner_optimizer_state(outer_ppo):
    """Adam's moments must survive the rewrite; resetting them each iteration
    would silently turn the inner loop into a fresh optimiser every time."""
    model = tiny_model()
    outer = outer_ppo.OuterUpdate(model.parameters(), 2.0, 0.0)
    outer.snapshot()
    optimizer = run_inner_loop(model)
    moments_before = [optimizer.state[p]['exp_avg'].clone() for p in model.parameters()]
    outer.apply()
    for parameter, before in zip(model.parameters(), moments_before):
        assert torch.equal(optimizer.state[parameter]['exp_avg'], before)


def test_outer_update_carries_no_gradient(outer_ppo):
    model = tiny_model()
    outer = outer_ppo.OuterUpdate(model.parameters(), 1.7, 0.5)
    outer.snapshot()
    run_inner_loop(model)
    outer.apply()
    for parameter in model.parameters():
        assert parameter.is_leaf and parameter.requires_grad


def test_agent_is_unchanged_from_ppo(outer_ppo):
    ppo = load_trainer('ppo_atari')
    envs = DiscreteEnvStub(6)
    torch.manual_seed(0)
    ours = outer_ppo.Agent(envs)
    torch.manual_seed(0)
    theirs = ppo.Agent(envs)
    for key, value in ours.state_dict().items():
        assert torch.equal(value, theirs.state_dict()[key]), key
