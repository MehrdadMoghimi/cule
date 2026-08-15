"""Regression guard for the Gumbel-max behaviour policy.

Several trainers sample from softmax(Q / tau) with the Gumbel-max trick rather
than building a `Categorical`, so the sampling stays inside a compiled or
CUDA-graph-captured region.  The expression is easy to get subtly wrong:

    -torch.log(-torch.log(u).clamp_min(1e-10))     # WRONG
    -torch.log((-torch.log(u)).clamp_min(1e-10))   # right

Unary minus binds looser than the method call, so in the first form the clamp is
applied to `log(u)`, which is negative, pinning it to +1e-10; the outer
`log(-1e-10)` is then NaN for every element.  `argmax` over an all-NaN row
returns index 0, so the bug is silent: the agent simply takes action 0 forever
and nothing raises.

These tests pin both the numerics and the source text.
"""

import pathlib

import numpy as np
import pytest
import torch

from conftest import CLEANRL_DIR

WRONG_FORM = '-torch.log(-torch.log('
RIGHT_FORM = '-torch.log((-torch.log('


def gumbel_noise(shape, generator=None):
    uniform = torch.rand(shape, generator=generator).clamp_min(1e-10)
    return -torch.log((-torch.log(uniform)).clamp_min(1e-10))


def test_gumbel_noise_is_finite():
    noise = gumbel_noise((10_000,))
    assert torch.isfinite(noise).all()


def test_the_wrong_form_really_is_nan():
    """Documents the failure mode, so the guard below is obviously worth having."""
    uniform = torch.rand(1000).clamp_min(1e-10)
    wrong = -torch.log(-torch.log(uniform).clamp_min(1e-10))
    assert torch.isnan(wrong).all()
    assert torch.argmax(torch.randn(4, 5) + wrong[:5]).item() == 0


def test_gumbel_max_reproduces_the_softmax_distribution():
    torch.manual_seed(0)
    tau = 0.03
    q_values = torch.tensor([0.06, 0.05, 0.12, 0.10])
    samples = torch.argmax(
        q_values / tau + gumbel_noise((200_000, 4)), dim=-1)

    empirical = torch.bincount(samples, minlength=4).double() / samples.numel()
    expected = torch.softmax(q_values / tau, dim=-1).double()
    np.testing.assert_allclose(empirical.numpy(), expected.numpy(), atol=0.005)


@pytest.mark.parametrize('name', sorted(
    path.name for path in pathlib.Path(CLEANRL_DIR).glob('*.py')))
def test_no_trainer_uses_the_wrong_gumbel_form(name):
    source = (pathlib.Path(CLEANRL_DIR) / name).read_text()
    if 'gumbel' not in source:
        pytest.skip('no Gumbel-max sampling in this trainer')
    if WRONG_FORM not in source and RIGHT_FORM not in source:
        pytest.skip('uses F.gumbel_softmax rather than hand-rolled Gumbel noise')
    assert RIGHT_FORM in source, f'{name} builds Gumbel noise without the inner parentheses'
    for line in source.splitlines():
        stripped = line.strip()
        if stripped.startswith('gumbel'):
            assert stripped.startswith(f'gumbel = {RIGHT_FORM}'), (
                f'{name}: {stripped!r} would evaluate to NaN')
