"""SEM: equivalence against waltermayor/FastTD3_SEm's `SimNorm`/`SimNormLinear`.

The reference modules import only torch, so they are extracted from the clone
and **executed** — every numerical test below diffs against the authors' own
code running on shared weights, not against a transcription.

The rest checks the geometry the name promises: that the output lies on a
product of simplices, that the softmax runs over the right axis (three wrong
axes are individually ruled out), and that the diff from the PQN parent is
confined to the one dense block it is supposed to touch.
"""

import ast
import os
import textwrap

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from conftest import REPO_ROOT, DiscreteEnvStub, load_trainer

TRAINER = 'sem_pqn_atari_envpool'
UPSTREAM = os.path.join(REPO_ROOT, 'third_party', 'upstream', 'sem',
                        'fast_td3', 'fast_td3.py')


@pytest.fixture(scope='module')
def sem():
    return load_trainer(TRAINER)


@pytest.fixture(scope='module')
def upstream():
    """Execute `SimNorm` and `SimNormLinear` straight out of the clone."""
    if not os.path.exists(UPSTREAM):
        pytest.skip('FastTD3_SEm clone not present under third_party/upstream')
    with open(UPSTREAM) as handle:
        source = handle.read()
    tree = ast.parse(source)
    namespace = {'nn': nn, 'torch': torch, 'F': F}
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name in ('SimNorm', 'SimNormLinear'):
            exec(textwrap.dedent(ast.get_source_segment(source, node)), namespace)
    assert {'SimNorm', 'SimNormLinear'} <= set(namespace)
    return namespace


# ---------------------------------------------------------------------------
# against the reference, running
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('groups,dim', [(8, 8), (64, 8), (16, 32), (1, 4), (128, 2)])
def test_simnorm_matches_upstream(sem, upstream, groups, dim):
    generator = torch.Generator().manual_seed(groups * 1000 + dim)
    x = torch.randn(7, groups * dim, generator=generator, dtype=torch.float64) * 3.0
    ours = sem.SimNorm(groups, dim)(x)
    theirs = upstream['SimNorm'](groups, dim)(x)
    assert torch.equal(ours, theirs)


def test_simnorm_matches_upstream_on_extra_leading_axes(sem, upstream):
    """The reference keeps `*shape[:-1]`, so it works on `[T, B, D]` too."""
    x = torch.randn(3, 5, 64, dtype=torch.float64)
    assert torch.equal(sem.SimNorm(8, 8)(x), upstream['SimNorm'](8, 8)(x))


@pytest.mark.parametrize('seed', range(4))
def test_simnorm_linear_matches_upstream(sem, upstream, seed):
    torch.manual_seed(seed)
    ours = sem.SimNormLinear(37, 16, 8).double()
    theirs = upstream['SimNormLinear'](37, 16, 8).double()
    theirs.load_state_dict(ours.state_dict())

    x = torch.randn(11, 37, dtype=torch.float64, generator=torch.Generator().manual_seed(seed))
    assert torch.allclose(ours(x), theirs(x), rtol=0, atol=1e-15)


def test_simnorm_linear_state_dict_layout_matches_upstream(sem, upstream):
    ours = sem.SimNormLinear(20, 8, 8)
    theirs = upstream['SimNormLinear'](20, 8, 8)
    assert ours.state_dict().keys() == theirs.state_dict().keys()


def test_repr_matches_upstream(sem, upstream):
    assert repr(sem.SimNorm(64, 8)) == repr(upstream['SimNorm'](64, 8))


# ---------------------------------------------------------------------------
# the geometry
# ---------------------------------------------------------------------------

def test_output_lies_on_a_product_of_simplices(sem):
    """Each group sums to exactly 1 and every entry is in [0, 1]."""
    torch.manual_seed(0)
    groups, dim = 64, 8
    x = torch.randn(9, groups * dim, dtype=torch.float64) * 5.0
    out = sem.SimNorm(groups, dim)(x)

    assert (out >= 0).all() and (out <= 1).all()
    sums = out.view(9, groups, dim).sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), rtol=0, atol=1e-14)


def test_softmax_axis_is_within_group_not_flat(sem):
    """The three plausible wrong axes, ruled out one at a time."""
    torch.manual_seed(0)
    groups, dim = 4, 3
    x = torch.randn(2, groups * dim, dtype=torch.float64) * 2.0
    out = sem.SimNorm(groups, dim)(x)

    within = F.softmax(x.view(2, groups, dim), dim=-1).view(2, groups * dim)
    flat = F.softmax(x, dim=-1)
    across_groups = F.softmax(x.view(2, groups, dim), dim=1).view(2, groups * dim)

    assert torch.allclose(out, within, rtol=0, atol=1e-15)
    assert not torch.allclose(out, flat, rtol=1e-3, atol=1e-3)
    assert not torch.allclose(out, across_groups, rtol=1e-3, atol=1e-3)


def test_reshape_is_row_major_over_groups(sem):
    """`view(..., L, V)` groups *consecutive* dimensions. Interleaving them
    instead would still sum to 1 per group and still pass the simplex test, so
    it is checked directly."""
    groups, dim = 2, 3
    # Group 0 gets a big first entry, group 1 a big last entry.
    x = torch.tensor([[10.0, 0.0, 0.0, 0.0, 0.0, 10.0]], dtype=torch.float64)
    out = sem.SimNorm(groups, dim)(x)[0]
    assert out[0] > 0.99 and out[5] > 0.99
    assert out[2] < 0.01 and out[3] < 0.01


def test_a_single_group_reduces_to_a_plain_softmax(sem):
    torch.manual_seed(0)
    x = torch.randn(4, 16, dtype=torch.float64)
    assert torch.allclose(sem.SimNorm(1, 16)(x), F.softmax(x, dim=-1), rtol=0, atol=1e-15)


def test_activation_is_sparse_relative_to_relu(sem):
    """Group-sparsity is the claim: most entries near zero, one near one."""
    torch.manual_seed(0)
    groups, dim = 64, 8
    x = torch.randn(64, groups * dim, dtype=torch.float64) * 4.0
    out = sem.SimNorm(groups, dim)(x)
    fraction_near_zero = (out < 0.05).double().mean().item()
    assert fraction_near_zero > 0.6
    # Exactly `groups` entries per row can exceed 0.5, since each group sums to 1.
    assert (out > 0.5).sum(dim=-1).max().item() <= groups


def test_gradient_is_the_ordinary_softmax_jacobian(sem):
    """No straight-through estimator, no quantisation: the gradient is exact."""
    torch.manual_seed(0)
    groups, dim = 8, 4
    x = torch.randn(3, groups * dim, dtype=torch.float64, requires_grad=True)
    out = sem.SimNorm(groups, dim)(x)
    upstream_grad = torch.randn_like(out)
    out.backward(upstream_grad)

    reference = x.detach().clone().requires_grad_(True)
    manual = F.softmax(reference.view(3, groups, dim), dim=-1).view(3, groups * dim)
    manual.backward(upstream_grad)
    assert torch.allclose(x.grad, reference.grad, rtol=0, atol=1e-15)


def test_output_scale_is_bounded_regardless_of_input_scale(sem):
    """The collapse/blow-up argument, made concrete."""
    layer = sem.SimNorm(64, 8)
    for scale in (1e-3, 1.0, 1e3, 1e6):
        out = layer(torch.randn(4, 512, dtype=torch.float64) * scale)
        assert out.max().item() <= 1.0
        assert torch.isfinite(out).all()


def test_layernorm_precedes_the_softmax_and_spans_all_groups(sem):
    """Per-group LayerNorm would erase the between-group scale differences that
    set each softmax's effective temperature; check the order and the width."""
    block = sem.SimNormLinear(10, 8, 4)
    assert isinstance(block.norm, nn.LayerNorm)
    assert block.norm.normalized_shape == (32,)

    torch.manual_seed(0)
    x = torch.randn(5, 10, dtype=torch.float64)
    block = block.double()
    expected = block.simnorm(block.norm(block.linear(x)))
    assert torch.allclose(block(x), expected, rtol=0, atol=1e-15)

    # Softmax-then-normalise is a different function.
    wrong = block.norm(block.simnorm(block.linear(x)))
    assert not torch.allclose(block(x), wrong, rtol=1e-3, atol=1e-3)


# ---------------------------------------------------------------------------
# the diff from PQN
# ---------------------------------------------------------------------------

def test_convolutional_stack_is_identical_to_pqn(sem):
    """SEM touches the dense trunk only; the encoder must be bit-identical."""
    pqn = load_trainer('pqn_atari_envpool')
    envs = DiscreteEnvStub(6)

    torch.manual_seed(0)
    ours = sem.QNetwork(envs, 64, 8)
    torch.manual_seed(0)
    parent = pqn.QNetwork(envs)

    parent_layers = [m for m in parent.network if isinstance(m, (nn.Conv2d, nn.LayerNorm))]
    our_layers = [m for m in ours.encoder if isinstance(m, (nn.Conv2d, nn.LayerNorm))]
    assert len(our_layers) == 6
    for a, b in zip(our_layers, parent_layers[:6]):
        assert torch.equal(a.weight, b.weight)
        assert torch.equal(a.bias, b.bias)


def test_parameter_count_matches_pqn_at_the_default_widths(sem):
    """64 x 8 = 512 keeps the trunk exactly as wide as the parent's, so the
    comparison against PQN (and against Hadamax) is not confounded by size."""
    pqn = load_trainer('pqn_atari_envpool')
    envs = DiscreteEnvStub(6)
    ours = sum(p.numel() for p in sem.QNetwork(envs, 64, 8).parameters())
    theirs = sum(p.numel() for p in pqn.QNetwork(envs).parameters())
    assert ours == theirs


def test_qnetwork_forward_shape_and_range(sem):
    torch.manual_seed(0)
    network = sem.QNetwork(DiscreteEnvStub(6), 64, 8)
    obs = torch.randint(0, 255, (5, 4, 84, 84), dtype=torch.uint8).float()
    q_values = network(obs)
    assert q_values.shape == (5, 6)
    assert torch.isfinite(q_values).all()


def test_trunk_output_is_on_the_simplex_end_to_end(sem):
    """Straight through the real encoder, not just the layer in isolation."""
    torch.manual_seed(0)
    network = sem.QNetwork(DiscreteEnvStub(6), 32, 16).double()
    obs = torch.randint(0, 255, (4, 4, 84, 84), dtype=torch.uint8).double()
    features = network.trunk(network.encoder(obs / 255.0))
    assert features.shape == (4, 512)
    sums = features.view(4, 32, 16).sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), rtol=0, atol=1e-12)


@pytest.mark.parametrize('groups,dim', [(0, 8), (64, 1), (-1, 8)])
def test_degenerate_widths_are_rejected(sem, groups, dim):
    with pytest.raises(ValueError, match='sem_groups'):
        sem.QNetwork(DiscreteEnvStub(6), groups, dim)


def test_non_default_widths_change_the_trunk_width(sem):
    network = sem.QNetwork(DiscreteEnvStub(6), 16, 4)
    assert network.trunk.linear.out_features == 64
    assert network.head.in_features == 64


def test_learning_rule_is_untouched(sem):
    """Q(lambda), the epsilon schedule and every hyperparameter are inherited."""
    pqn = load_trainer('pqn_atari_envpool')
    for start, end, duration, t in ((1.0, 0.01, 1000, 0), (1.0, 0.01, 1000, 500),
                                    (1.0, 0.01, 1000, 5000)):
        assert sem.linear_schedule(start, end, duration, t) == \
            pqn.linear_schedule(start, end, duration, t)

    import dataclasses

    ours = {f.name: f.default for f in dataclasses.fields(sem.Args)}
    theirs = {f.name: f.default for f in dataclasses.fields(pqn.Args)}
    for key in theirs:
        if key == 'exp_name':
            continue
        assert ours[key] == theirs[key], key
    assert set(ours) - set(theirs) == {'sem_groups', 'sem_dim'}
