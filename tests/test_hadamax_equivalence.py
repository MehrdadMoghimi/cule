"""Hadamax: numerical equivalence against the official Flax encoder.

The reference in `tests/reference/hadamax_reference.py` is written from Flax/JAX
semantics in NumPy, independently of the PyTorch port, so it genuinely catches
the transcription traps in this encoder: SAME padding for even kernels,
channel-only LayerNorm, Flax's 1e-6 LayerNorm epsilon, and `jax.nn.gelu`'s tanh
default.
"""

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from conftest import DiscreteEnvStub, load_trainer
from reference import hadamax_reference as ref

TRAINERS = ['hadamax_pqn_atari_envpool', 'hadamax_pqn_atari_envpool_torchcompile']


def _block_params(block):
    """Pull one HadamaxBlock's weights out in the reference's Flax layouts."""
    to_hwio = lambda w: w.detach().double().numpy().transpose(2, 3, 1, 0)  # OIHW -> HWIO
    vec = lambda t: t.detach().double().numpy()
    return {
        'w1': to_hwio(block.conv1.weight), 'b1': vec(block.conv1.bias),
        'w2': to_hwio(block.conv2.weight), 'b2': vec(block.conv2.bias),
        'g1': vec(block.norm1.norm.weight), 'be1': vec(block.norm1.norm.bias),
        'g2': vec(block.norm2.norm.weight), 'be2': vec(block.norm2.norm.bias),
    }


@pytest.fixture(params=TRAINERS)
def network(request):
    torch.manual_seed(0)
    module = load_trainer(request.param)
    net = module.QNetwork(DiscreteEnvStub(6)).double().eval()
    # Random non-trivial norm affine parameters: an all-ones scale / all-zeros
    # bias would hide a wrong epsilon or a wrong reduction axis.
    for parameter in net.parameters():
        if parameter.dim() == 1:
            torch.nn.init.normal_(parameter, std=0.3)
    return net


def test_encoder_matches_flax_reference(network):
    observations = torch.randint(0, 256, (2, 4, 84, 84)).double()

    trunk = network.encoder[:3]  # three HadamaxBlocks, before the Flatten
    ours = trunk(observations / 255.0)

    reference = ref.hadamax_encoder(
        (observations / 255.0).permute(0, 2, 3, 1).numpy(),  # NCHW -> NHWC
        [_block_params(block) for block in trunk],
    )

    assert ours.shape == (2, 64, 11, 11)
    np.testing.assert_allclose(
        ours.permute(0, 2, 3, 1).detach().numpy(), reference, rtol=1e-10, atol=1e-10)


def test_block_output_shapes_match_flax_same_padding(network):
    """84 -> 21 -> 11 -> 11, i.e. 7744 features into the projection."""
    x = torch.zeros(1, 4, 84, 84, dtype=torch.float64)
    expected = [(1, 32, 21, 21), (1, 64, 11, 11), (1, 64, 11, 11)]
    for block, shape in zip(network.encoder[:3], expected):
        x = block(x)
        assert tuple(x.shape) == shape
    assert network.projection.in_features == 64 * 11 * 11 == 7744


def test_projection_head_matches_flax_reference(network):
    flat = torch.randn(3, 7744, dtype=torch.float64)

    ours = F.gelu(network.projection_norm(network.projection(flat)), approximate='tanh')
    reference = ref.hadamax_projection(
        flat.numpy(),
        network.projection.weight.detach().double().numpy().T,  # torch (out, in) -> flax (in, out)
        network.projection.bias.detach().double().numpy(),
        network.projection_norm.weight.detach().double().numpy(),
        network.projection_norm.bias.detach().double().numpy(),
    )
    np.testing.assert_allclose(ours.detach().numpy(), reference, rtol=1e-11, atol=1e-11)


def test_gelu_uses_the_tanh_approximation(network):
    """`jax.nn.gelu` defaults to approximate=True; PyTorch's F.gelu does not.

    Guards against silently reverting to the exact erf GELU.
    """
    x = torch.linspace(-3, 3, 41, dtype=torch.float64)
    np.testing.assert_allclose(
        F.gelu(x, approximate='tanh').numpy(), ref.gelu_tanh(x.numpy()), rtol=0, atol=1e-12)
    # The two GELUs differ enough that mixing them up would be a real error.
    assert torch.max(torch.abs(F.gelu(x) - F.gelu(x, approximate='tanh'))) > 1e-4


def test_layer_norm_epsilon_matches_flax(network):
    for block in network.encoder[:3]:
        assert block.norm1.norm.eps == ref.FLAX_LAYER_NORM_EPS
        assert block.norm2.norm.eps == ref.FLAX_LAYER_NORM_EPS
    assert network.projection_norm.eps == ref.FLAX_LAYER_NORM_EPS


def test_channel_layer_norm_reduces_channels_only(network):
    """Flax normalizes the trailing axis; the PQN parent normalizes [C, H, W]."""
    module = load_trainer(TRAINERS[0])
    norm = module.ChannelLayerNorm(8).double()
    torch.nn.init.ones_(norm.norm.weight)
    torch.nn.init.zeros_(norm.norm.bias)
    x = torch.randn(2, 8, 5, 5, dtype=torch.float64)
    out = norm(x)
    # Each spatial position is standardized independently across channels.  The
    # variance lands just under 1 because of the 1e-6 epsilon: the shrinkage is
    # exactly var / (var + eps), which is itself worth pinning down.
    np.testing.assert_allclose(out.mean(dim=1).detach().numpy(), 0.0, atol=1e-9)
    variance = x.var(dim=1, unbiased=False)
    expected = (variance / (variance + ref.FLAX_LAYER_NORM_EPS)).detach().numpy()
    np.testing.assert_allclose(
        out.var(dim=1, unbiased=False).detach().numpy(), expected, rtol=1e-9, atol=1e-9)


def test_hyperparameters_are_inherited_from_pqn():
    """Hadamax changes the encoder only; every learning hyperparameter is PQN's."""
    for name in TRAINERS:
        hadamax = load_trainer(name).Args()
        parent = load_trainer(name.replace('hadamax_pqn', 'pqn')).Args()
        for field in ('learning_rate', 'gamma', 'q_lambda', 'num_steps', 'num_envs',
                      'num_minibatches', 'update_epochs', 'max_grad_norm',
                      'start_e', 'end_e', 'exploration_fraction', 'anneal_lr'):
            assert getattr(hadamax, field) == getattr(parent, field), field


def test_both_variants_define_the_same_encoder():
    eager = load_trainer(TRAINERS[0])
    compiled = load_trainer(TRAINERS[1])
    torch.manual_seed(4)
    a = eager.QNetwork(DiscreteEnvStub(6))
    torch.manual_seed(4)
    b = compiled.QNetwork(DiscreteEnvStub(6))
    assert [k for k in a.state_dict()] == [k for k in b.state_dict()]
    x = torch.randint(0, 256, (2, 4, 84, 84)).float()
    torch.testing.assert_close(a(x), b(x), rtol=0, atol=0)
