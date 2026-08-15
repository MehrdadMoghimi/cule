"""DiscoRL: the discovered update rule, checked against the reference.

The published rule *is* its weights, so the strongest available check is that
`cleanrl/disco_atari.py` reproduces the Haiku computation numerically with the
released `disco_103.npz` loaded. Two independent references are used:

  * `tests/reference/disco_reference.py`, a NumPy transcription of the Haiku
    modules driven by the published input-option table (always available);
  * the real JAX/Haiku implementation, if jax, haiku and rlax are installed
    (skipped otherwise).

The weights file is not vendored. These tests use ~/.cache/cule-disco or
$DISCO_META_WEIGHTS and skip if neither exists; run any disco trainer once to
populate the cache.
"""

import os

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from conftest import load_trainer
from reference import disco_reference as ref

TRAINERS = ['disco_atari', 'disco_atari_torchcompile']
HORIZON, BATCH, N_ACTIONS = 6, 3, 5
PUBLISHED_PARAM_COUNT = 754778
PUBLISHED_ARRAY_COUNT = 42


@pytest.fixture(params=TRAINERS)
def module(request):
    return load_trainer(request.param)


def weights_path():
    override = os.environ.get('DISCO_META_WEIGHTS')
    if override and os.path.exists(override):
        return override
    cached = os.path.join(os.path.expanduser('~'), '.cache', 'cule-disco', 'disco_103.npz')
    if os.path.exists(cached):
        return cached
    return None


published = pytest.mark.skipif(
    weights_path() is None,
    reason='disco_103.npz not cached; run a disco trainer once to download it')


@pytest.fixture(scope='module')
def arrays():
    path = weights_path()
    if path is None:
        pytest.skip('disco_103.npz not cached')
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def make_inputs(seed=0, horizon=HORIZON, batch=BATCH, n_actions=N_ACTIONS,
                prediction_size=600, terminal_at=None):
    """Random but shape-correct meta-network inputs, as NumPy float64."""
    rng = np.random.default_rng(seed)
    steps = horizon + 1

    def normal(*shape):
        return rng.standard_normal(shape)

    is_terminal = np.zeros((horizon, batch))
    if terminal_at is not None:
        is_terminal[terminal_at, 0] = 1.0

    return dict(
        actions=rng.integers(n_actions, size=(steps, batch)),
        rewards=normal(horizon, batch),
        is_terminal=is_terminal,
        logits=normal(steps, batch, n_actions),
        behaviour_logits=normal(steps, batch, n_actions),
        target_logits=normal(steps, batch, n_actions),
        y=normal(steps, batch, prediction_size),
        target_y=normal(steps, batch, prediction_size),
        z=normal(steps, batch, n_actions, prediction_size),
        target_z=normal(steps, batch, n_actions, prediction_size),
        v_scalar=normal(steps, batch),
        adv=normal(horizon, batch),
        normalized_adv=normal(horizon, batch),
        q=normal(steps, batch, n_actions),
        qv_adv=normal(steps, batch, n_actions),
        normalized_qv_adv=normal(steps, batch, n_actions),
    )


def as_reference_tree(inputs):
    """Reshape the flat dict into the nested paths the input option addresses."""
    return dict(
        actions=inputs['actions'],
        rewards=inputs['rewards'],
        is_terminal=inputs['is_terminal'],
        agent_out=dict(logits=inputs['logits'], y=inputs['y'], z=inputs['z']),
        behaviour_agent_out=dict(logits=inputs['behaviour_logits']),
        extra_from_rule=dict(
            v_scalar=inputs['v_scalar'],
            adv=inputs['adv'],
            normalized_adv=inputs['normalized_adv'],
            q=inputs['q'],
            qv_adv=inputs['qv_adv'],
            normalized_qv_adv=inputs['normalized_qv_adv'],
            target_out=dict(
                logits=inputs['target_logits'], y=inputs['target_y'], z=inputs['target_z']),
        ),
    )


def as_meta_inputs(module, inputs):
    fields = {}
    for name, value in inputs.items():
        dtype = torch.int64 if name == 'actions' else torch.float64
        fields[name] = torch.as_tensor(value, dtype=dtype)
    return module.MetaInputs(**fields)


def build_meta_net(module, arrays):
    net = module.DiscoMetaNet().double()
    net.load_published_weights(arrays)
    return net.eval().requires_grad_(False)


# --------------------------------------------------------------------------
# The published weights
# --------------------------------------------------------------------------

@published
def test_every_published_array_is_consumed_and_the_count_matches(module, arrays):
    """A mis-specified architecture cannot absorb 42 arrays of fixed shape."""
    assert len(arrays) == PUBLISHED_ARRAY_COUNT
    assert sum(value.size for value in arrays.values()) == PUBLISHED_PARAM_COUNT

    net = module.DiscoMetaNet()
    assert sum(p.numel() for p in net.parameters()) == PUBLISHED_PARAM_COUNT
    # load_published_weights indexes every key by name; a missing one raises.
    net.load_published_weights(arrays)


@published
def test_published_shapes_pin_the_architecture(module, arrays):
    """Each kernel's shape fixes one architectural constant."""
    prediction, hidden, meta_hidden = 600, 256, 128
    assert arrays['lstm/lstm/linear/w'].shape == (module.META_INPUT_SIZE + hidden, 4 * hidden)
    assert arrays['lstm/linear/w'].shape == (meta_hidden, hidden)
    assert arrays['lstm/linear_1/w'].shape == (hidden, 1)
    assert arrays['lstm/linear_2/w'].shape == (hidden, prediction)
    assert arrays['lstm/linear_3/w'].shape == (hidden, prediction)
    # The action-conditional conv sees 9 features, doubled by the mean pooling.
    assert arrays['lstm/sequential/conv1_d/w'].shape == (
        1, 2 * module.ACTION_CONDITIONAL_INPUT_SIZE, 16)
    assert arrays['lstm/sequential/conv1_d_1/w'].shape == (1, 32, 2)
    # The policy-target conv sees the trajectory state plus the 2-channel policy
    # embedding, again doubled: 2 * (256 + 2) == 516.
    assert arrays['lstm/sequential_1/conv1_d/w'].shape == (1, 2 * (hidden + 2), 16)
    assert arrays['lstm/linear_4/w'].shape == (16, 1)
    assert arrays['lstm/mlp/~/linear_0/w'].shape == (prediction, 16)
    # 27 base features + meta_input_emb + the 1-wide y embedding == 29.
    assert arrays['lstm/~/meta_lstm/~unroll/mlp_2/~/linear_0/w'].shape == (
        module.META_INPUT_SIZE + 2, 16)
    assert arrays['lstm/~/meta_lstm/~unroll/lstm/linear/w'].shape == (
        16 + meta_hidden, 4 * meta_hidden)
    assert module.PREDICTION_SIZE == prediction


@published
def test_meta_net_matches_the_numpy_reference(module, arrays):
    """The whole discovered rule, weights and all, to 1e-9."""
    inputs = make_inputs(seed=1)
    net = build_meta_net(module, arrays)

    pi_hat, y_hat, z_hat = net(as_meta_inputs(module, inputs))
    expected = ref.meta_net_forward(
        as_reference_tree(inputs), arrays,
        np.zeros(128), np.zeros(128))

    np.testing.assert_allclose(pi_hat.numpy(), expected[0], rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(y_hat.numpy(), expected[1], rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(z_hat.numpy(), expected[2], rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(net.meta_hidden.numpy(), expected[3], rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(net.meta_cell.numpy(), expected[4], rtol=1e-9, atol=1e-9)


@published
def test_meta_net_matches_the_reference_across_consecutive_calls(module, arrays):
    """The lifetime LSTM state must advance in step with the reference."""
    net = build_meta_net(module, arrays)
    hidden, cell = np.zeros(128), np.zeros(128)
    for step in range(3):
        inputs = make_inputs(seed=10 + step, terminal_at=step)
        pi_hat, _, _ = net(as_meta_inputs(module, inputs))
        expected_pi, _, _, hidden, cell = ref.meta_net_forward(
            as_reference_tree(inputs), arrays, hidden, cell)
        np.testing.assert_allclose(pi_hat.numpy(), expected_pi, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(net.meta_hidden.numpy(), hidden, rtol=1e-9, atol=1e-9)


@published
def test_the_rule_produces_a_real_policy_target(module, arrays):
    """Sanity: pi_hat is finite and is not just a copy of the agent's policy."""
    inputs = make_inputs(seed=2)
    net = build_meta_net(module, arrays)
    pi_hat, y_hat, z_hat = net(as_meta_inputs(module, inputs))
    for output in (pi_hat, y_hat, z_hat):
        assert torch.isfinite(output).all()
        assert output.abs().sum() > 0
    agent_policy = torch.softmax(torch.as_tensor(inputs['logits'][:-1]), dim=-1)
    assert not torch.allclose(torch.softmax(pi_hat, dim=-1), agent_policy, atol=1e-3)


# --------------------------------------------------------------------------
# Meta-network structure
# --------------------------------------------------------------------------

def test_constructed_input_widths(module):
    """23 base features plus two 2-wide poolings is what the LSTM kernel expects."""
    encoder = module.MetaInputEncoder((16, 1), (16, 2)).double()
    inputs = as_meta_inputs(module, make_inputs(seed=3))
    x, policy_emb = encoder(inputs)
    assert x.shape == (HORIZON, BATCH, module.META_INPUT_SIZE)
    assert policy_emb.shape == (HORIZON, BATCH, N_ACTIONS, 2)
    assert encoder.policy_net.layers[0].in_features == 2 * module.ACTION_CONDITIONAL_INPUT_SIZE


def test_the_trajectory_lstm_runs_backwards(module):
    """A forward unroll would make the rule unable to bootstrap at all."""
    torch.manual_seed(0)
    net = module.DiscoMetaNet().double()
    x = torch.randn(HORIZON, BATCH, module.META_INPUT_SIZE, dtype=torch.float64)
    should_reset = torch.zeros(HORIZON, BATCH, dtype=torch.float64)

    base = net.unroll_backwards(x, should_reset)
    perturbed = x.clone()
    perturbed[3] += 10.0
    moved = net.unroll_backwards(perturbed, should_reset)

    difference = (moved - base).abs().flatten(1).sum(1)
    assert (difference[:4] > 0).all(), 'a change at t must reach every earlier step'
    assert (difference[4:] == 0).all(), 'it must not reach any later step'


def test_terminal_transitions_cut_the_backward_trace(module):
    torch.manual_seed(1)
    net = module.DiscoMetaNet().double()
    x = torch.randn(HORIZON, BATCH, module.META_INPUT_SIZE, dtype=torch.float64)
    should_reset = torch.zeros(HORIZON, BATCH, dtype=torch.float64)
    should_reset[3, 0] = 1.0

    base = net.unroll_backwards(x, should_reset)
    perturbed = x.clone()
    perturbed[5, 0] += 10.0
    moved = net.unroll_backwards(perturbed, should_reset)

    difference = (moved - base).abs().sum(-1)
    # Going backwards the perturbation at t=5 reaches t=4, and then the reset at
    # t=3 zeroes the incoming state, so nothing earlier moves at all.
    assert (difference[4:, 0] > 0).all()
    assert (difference[:4, 0] == 0).all(), 'the reset at t=3 must block the trace'
    # Without a reset the same perturbation reaches every earlier step.
    unreset = (net.unroll_backwards(perturbed, torch.zeros_like(should_reset))
               - net.unroll_backwards(x, torch.zeros_like(should_reset))).abs().sum(-1)
    assert (unreset[:, 0] > 0).all()
    # Rows that were not perturbed never move.
    assert difference[:, 1].sum() == 0


def test_multiplicative_interaction_uses_the_previous_lifetime_state(module):
    """The state is read before it is updated; reading it after would leak the
    current rollout into its own targets."""
    torch.manual_seed(2)
    net = module.DiscoMetaNet().double().requires_grad_(False)
    inputs = as_meta_inputs(module, make_inputs(seed=4))

    before = net.meta_hidden.clone()
    y_first = net(inputs)[1]
    after = net.meta_hidden.clone()
    assert not torch.allclose(before, after), 'the lifetime state must advance'

    net.meta_hidden.copy_(before)
    net.meta_cell.zero_()
    y_again = net(inputs)[1]
    torch.testing.assert_close(y_first, y_again, rtol=0, atol=0)


def test_haiku_lstm_gate_order_and_forget_bias(module):
    """PyTorch's LSTMCell orders gates i, f, g, o with no +1; the published
    kernels are Haiku's i, g, f, o with one."""
    cell = module.HaikuLSTMCell(3, 4).double()
    with torch.no_grad():
        cell.linear.weight.zero_()
        cell.linear.bias.zero_()
    x = torch.zeros(1, 3, dtype=torch.float64)
    hidden = torch.zeros(1, 4, dtype=torch.float64)
    memory = torch.ones(1, 4, dtype=torch.float64)

    with torch.no_grad():
        new_hidden, new_cell = cell(x, hidden, memory)
    # Every gate is zero pre-activation, so only the forget bias is visible:
    # c' = sigmoid(0 + 1) * 1, h' = sigmoid(0) * tanh(c'). Without the +1 the
    # cell would decay to 0.5 instead.
    np.testing.assert_allclose(new_cell.numpy(), 1.0 / (1.0 + np.exp(-1.0)), rtol=1e-12)
    np.testing.assert_allclose(
        new_hidden.numpy(), 0.5 * np.tanh(1.0 / (1.0 + np.exp(-1.0))), rtol=1e-12)
    assert cell.linear.out_features == 4 * 4, 'one kernel produces all four gates'


def test_conv1d_block_pools_over_actions(module):
    """Each action sees the mean over actions, which is what makes the block
    permutation equivariant instead of order dependent."""
    torch.manual_seed(3)
    net = module.Conv1DNet(3, (5,)).double()
    x = torch.randn(2, 2, 4, 3, dtype=torch.float64)
    out = net(x)
    assert out.shape == (2, 2, 4, 5)

    permutation = torch.tensor([2, 0, 3, 1])
    torch.testing.assert_close(net(x[:, :, permutation]), out[:, :, permutation],
                               rtol=1e-12, atol=1e-12)
    assert net.layers[0].in_features == 6, 'features are concatenated with their mean'


def test_haiku_mlp_has_no_output_activation(module):
    torch.manual_seed(4)
    mlp = module.HaikuMLP(4, (8, 3)).double()
    x = torch.randn(32, 4, dtype=torch.float64)
    out = mlp(x)
    assert (out < 0).any(), 'a final ReLU would make every output non-negative'
    torch.testing.assert_close(out, mlp.layers[1](F.relu(mlp.layers[0](x))), rtol=1e-12, atol=1e-12)


# --------------------------------------------------------------------------
# Value machinery
# --------------------------------------------------------------------------

def test_retrace_matches_the_numpy_recursion(module):
    rng = np.random.default_rng(5)
    q_t = rng.standard_normal((4, 3))
    v_t = rng.standard_normal((5, 3))
    r_t = rng.standard_normal((5, 3))
    discount_t = rng.random((5, 3))
    c_t = rng.random((4, 3))

    result = module.retrace_from_q_and_v(
        *(torch.as_tensor(a) for a in (q_t, v_t, r_t, discount_t, c_t)))
    expected = ref.retrace_from_q_and_v(q_t, v_t, r_t, discount_t, c_t)
    np.testing.assert_allclose(result.numpy(), expected, rtol=1e-12, atol=1e-12)
    assert result.shape == (5, 3)


def test_retrace_reduces_to_one_step_when_traces_are_cut(module):
    """c = 0 leaves G_t = r_t + d_t * v_t, the plain one-step target."""
    rng = np.random.default_rng(6)
    q_t = torch.as_tensor(rng.standard_normal((3, 2)))
    v_t = torch.as_tensor(rng.standard_normal((4, 2)))
    r_t = torch.as_tensor(rng.standard_normal((4, 2)))
    discount_t = torch.full((4, 2), 0.99, dtype=torch.float64)

    result = module.retrace_from_q_and_v(q_t, v_t, r_t, discount_t, torch.zeros(3, 2, dtype=torch.float64))
    torch.testing.assert_close(result, r_t + discount_t * v_t, rtol=1e-12, atol=1e-12)


def test_retrace_reduces_to_the_n_step_return_on_policy(module):
    """With c = 1 the correction cancels and Retrace becomes the n-step return."""
    horizon = 4
    q_t = torch.zeros(horizon - 1, 1, dtype=torch.float64)
    v_t = torch.zeros(horizon, 1, dtype=torch.float64)
    r_t = torch.arange(1.0, horizon + 1.0, dtype=torch.float64).reshape(horizon, 1)
    gamma = 0.9
    discount_t = torch.full((horizon, 1), gamma, dtype=torch.float64)

    result = module.retrace_from_q_and_v(
        q_t, v_t, r_t, discount_t, torch.ones(horizon - 1, 1, dtype=torch.float64))
    expected = sum(gamma**k * r_t[k] for k in range(horizon))
    torch.testing.assert_close(result[0], expected, rtol=1e-12, atol=1e-12)


def test_two_hot_round_trip(module):
    scalars = torch.tensor([[0.0, 1.5, -7.25, 300.0, -300.0, 1000.0]], dtype=torch.float64)
    probs = module.transform_to_2hot(scalars, -300.0, 300.0, 601)
    np.testing.assert_allclose(probs.sum(-1).numpy(), 1.0, atol=1e-9)

    decoded = module.transform_from_2hot(probs, -300.0, 300.0, 601)
    # The round trip is exact up to rlax's 1e-5 guard in the interpolation
    # denominator; 1000 saturates at the top bin.
    np.testing.assert_allclose(decoded.numpy()[0, :5], scalars.numpy()[0, :5], atol=1e-4)
    np.testing.assert_allclose(decoded.numpy()[0, 5], 300.0, atol=1e-4)

    expected = ref.transform_to_2hot(scalars.numpy(), -300.0, 300.0, 601)
    np.testing.assert_allclose(probs.numpy(), expected, atol=1e-9)


def test_signed_hyperbolic_is_inverted_by_signed_parabolic(module):
    x = torch.tensor([-5000.0, -1.0, 0.0, 0.25, 17.0, 12345.0], dtype=torch.float64)
    torch.testing.assert_close(
        module.signed_parabolic(module.signed_hyperbolic(x)), x, rtol=1e-8, atol=1e-6)
    np.testing.assert_allclose(
        module.signed_hyperbolic(x).numpy(), ref.signed_hyperbolic(x.numpy()), rtol=1e-12)


def test_value_logits_to_scalar_undoes_the_transform(module):
    """The head predicts h(G); reading it must return G, not h(G)."""
    returns = torch.tensor([[0.0, 4.0, -30.0, 250.0]], dtype=torch.float64)
    probs = module.transform_to_2hot(
        module.signed_hyperbolic(returns), -300.0, 300.0, 601)
    logits = torch.log(probs.clamp_min(1e-30))
    decoded = module.value_logits_to_scalar(logits, 300.0)
    # The two-hot guard's 1e-5 is amplified by the inverse squash, which grows
    # quadratically, so large returns carry a proportionally larger error.
    np.testing.assert_allclose(decoded.numpy(), returns.numpy(), rtol=1e-5, atol=1e-4)


def test_state_value_is_the_policy_weighted_q(module):
    """There is no separate value head: V comes out of Q and pi."""
    torch.manual_seed(6)
    steps, batch, actions, bins = 4, 2, 3, 601
    agent_out = {
        'logits': torch.randn(steps, batch, actions, dtype=torch.float64),
        'q': torch.randn(steps, batch, actions, bins, dtype=torch.float64),
        'y': torch.randn(steps, batch, 600, dtype=torch.float64),
        'z': torch.randn(steps, batch, actions, 600, dtype=torch.float64),
    }
    target_out = {key: value.clone() for key, value in agent_out.items()}
    adv_ema = module.MovingAverage(0.99, 1e-6).double()
    td_ema = module.MovingAverage(0.99, 1e-6).double()

    value_outs, _ = module.compute_value_outs(
        agent_out, target_out,
        agent_out['logits'],
        torch.randn(steps - 1, batch, dtype=torch.float64),
        torch.zeros(steps - 1, batch, dtype=torch.float64),
        torch.randint(actions, (steps, batch)),
        0.997, 0.95, 300.0, adv_ema, td_ema)

    q_values = module.value_logits_to_scalar(agent_out['q'], 300.0)
    expected = (torch.softmax(agent_out['logits'], -1) * q_values).sum(2)
    torch.testing.assert_close(value_outs.value, expected, rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(value_outs.target_value, expected, rtol=1e-10, atol=1e-10)
    # Both networks are identical here, so the advantage is measured against the
    # same values the TD error is, and q_td == q_target - Q(s, a).
    torch.testing.assert_close(value_outs.adv, value_outs.q_target - expected[:-1],
                               rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(value_outs.qv_adv, q_values - expected.unsqueeze(-1),
                               rtol=1e-10, atol=1e-10)


def test_importance_weight_is_one_on_policy(module):
    torch.manual_seed(7)
    logits = torch.randn(5, 3, 4, dtype=torch.float64)
    actions = torch.randint(4, (5, 3))
    rho = module.importance_weight(logits, logits, actions)
    np.testing.assert_allclose(rho.numpy(), 1.0, atol=1e-12)

    other = torch.randn(5, 3, 4, dtype=torch.float64)
    rho = module.importance_weight(logits, other, actions)
    expected = (torch.softmax(logits, -1) / torch.softmax(other, -1)).gather(
        -1, actions.unsqueeze(-1)).squeeze(-1)
    torch.testing.assert_close(rho, expected, rtol=1e-10, atol=1e-10)


def test_moving_average_debiases_like_adam(module):
    ema = module.MovingAverage(0.99, 1e-6).double()
    values = torch.full((4, 4), 3.0, dtype=torch.float64)
    for _ in range(5):
        ema.update(values)
    # A constant stream has zero variance, so the debiased mean is exact.
    normalized = ema.normalize(values)
    np.testing.assert_allclose(normalized.numpy(), 0.0, atol=1e-6)
    np.testing.assert_allclose((ema.moment1 / (1 - ema.decay_product)).item(), 3.0, rtol=1e-9)

    without_mean = ema.normalize(values, subtract_mean=False)
    assert without_mean.mean().item() > 1e5, 'zero variance means a huge scale'


# --------------------------------------------------------------------------
# The discovered loss
# --------------------------------------------------------------------------

def make_agent_out(seed, steps=HORIZON + 1, batch=BATCH, actions=N_ACTIONS, bins=601):
    torch.manual_seed(seed)
    return {
        'logits': torch.randn(steps, batch, actions, dtype=torch.float64, requires_grad=True),
        'y': torch.randn(steps, batch, 600, dtype=torch.float64, requires_grad=True),
        'z': torch.randn(steps, batch, actions, 600, dtype=torch.float64, requires_grad=True),
        'aux_pi': torch.randn(steps, batch, actions, actions, dtype=torch.float64,
                              requires_grad=True),
        'q': torch.randn(steps, batch, actions, bins, dtype=torch.float64, requires_grad=True),
    }


def test_agent_loss_is_the_sum_of_its_weighted_parts(module):
    agent_out = make_agent_out(8)
    actions = torch.randint(N_ACTIONS, (HORIZON + 1, BATCH))
    is_terminal = torch.zeros(HORIZON, BATCH, dtype=torch.float64)
    pi_hat = torch.randn(HORIZON, BATCH, N_ACTIONS, dtype=torch.float64)
    y_hat = torch.randn(HORIZON, BATCH, 600, dtype=torch.float64)
    z_hat = torch.randn(HORIZON, BATCH, 600, dtype=torch.float64)
    q_td = torch.randn(HORIZON, BATCH, dtype=torch.float64)

    total, log = module.disco_agent_loss(
        agent_out, actions, is_terminal, pi_hat, y_hat, z_hat, q_td,
        300.0, 1.0, 1.0, 1.0, 1.0, 0.2)

    expected = (
        module.categorical_kl_divergence(pi_hat, agent_out['logits'][:-1])
        + module.categorical_kl_divergence(y_hat, agent_out['y'][:-1])
        + module.categorical_kl_divergence(
            z_hat, module.batch_lookup(agent_out['z'][:-1], actions[:-1]))
        + module.categorical_kl_divergence(
            agent_out['logits'][1:].detach(),
            module.batch_lookup(agent_out['aux_pi'][:-1], actions[:-1]))
    )
    value_only, _ = module.disco_agent_loss(
        agent_out, actions, is_terminal, pi_hat, y_hat, z_hat, q_td,
        300.0, 0.0, 0.0, 0.0, 0.0, 0.2)
    torch.testing.assert_close(total, expected + value_only, rtol=1e-10, atol=1e-10)
    assert set(log) == {'pi_loss', 'y_loss', 'z_loss', 'aux_loss', 'value_loss'}


def test_kl_direction_is_target_to_prediction(module):
    """KL(pi_hat || logits): the discovered target is the first argument, so the
    agent is pulled onto it and not the other way round."""
    p = torch.tensor([[2.0, 0.0, -1.0]], dtype=torch.float64)
    q = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64)
    forward = module.categorical_kl_divergence(p, q)
    backward = module.categorical_kl_divergence(q, p)
    assert not torch.allclose(forward, backward)
    np.testing.assert_allclose(forward.numpy(), ref.categorical_kl_divergence(p.numpy(), q.numpy()),
                               rtol=1e-12)
    np.testing.assert_allclose(module.categorical_kl_divergence(p, p).numpy(), 0.0, atol=1e-12)


def test_auxiliary_policy_loss_is_masked_at_terminals(module):
    """The next observation belongs to a new episode there, so predicting its
    policy from this episode's model would be nonsense."""
    agent_out = make_agent_out(9)
    actions = torch.randint(N_ACTIONS, (HORIZON + 1, BATCH))
    pi_hat = torch.zeros(HORIZON, BATCH, N_ACTIONS, dtype=torch.float64)
    y_hat = torch.zeros(HORIZON, BATCH, 600, dtype=torch.float64)
    z_hat = torch.zeros(HORIZON, BATCH, 600, dtype=torch.float64)
    q_td = torch.zeros(HORIZON, BATCH, dtype=torch.float64)

    clean = torch.zeros(HORIZON, BATCH, dtype=torch.float64)
    terminal = clean.clone()
    terminal[2, 1] = 1.0
    args = (pi_hat, y_hat, z_hat, q_td, 300.0, 0.0, 0.0, 0.0, 1.0, 0.0)

    without, _ = module.disco_agent_loss(agent_out, actions, clean, *args)
    with_terminal, _ = module.disco_agent_loss(agent_out, actions, terminal, *args)
    assert with_terminal[2, 1].item() == 0.0
    assert without[2, 1].item() > 0.0
    difference = (without - with_terminal).abs()
    assert difference.sum().item() == pytest.approx(without[2, 1].item())


def test_value_loss_targets_the_meta_nets_td_error(module):
    """The value head is regressed onto h(V + TD): the meta-network supplies the
    TD error, nothing here computes one."""
    agent_out = make_agent_out(10)
    actions = torch.randint(N_ACTIONS, (HORIZON + 1, BATCH))
    zeros_pi = torch.zeros(HORIZON, BATCH, N_ACTIONS, dtype=torch.float64)
    zeros_y = torch.zeros(HORIZON, BATCH, 600, dtype=torch.float64)
    q_td = torch.randn(HORIZON, BATCH, dtype=torch.float64)

    total, _ = module.disco_agent_loss(
        agent_out, actions, torch.zeros(HORIZON, BATCH, dtype=torch.float64),
        zeros_pi, zeros_y, zeros_y, q_td, 300.0, 0.0, 0.0, 0.0, 0.0, 1.0)

    q_a = module.batch_lookup(agent_out['q'], actions)[:-1]
    values = module.value_logits_to_scalar(q_a, 300.0)
    target = module.transform_to_2hot(
        module.signed_hyperbolic(values + q_td), -300.0, 300.0, 601)
    expected = -(target * F.log_softmax(q_a, dim=-1)).sum(-1)
    torch.testing.assert_close(total, expected, rtol=1e-10, atol=1e-10)

    # With zero TD the target's mean is exactly the head's own current value:
    # the update then only sharpens the distribution, it does not move it.
    zero_td_target = module.transform_to_2hot(
        module.signed_hyperbolic(values), -300.0, 300.0, 601)
    decoded = module.signed_parabolic(
        module.transform_from_2hot(zero_td_target, -300.0, 300.0, 601))
    torch.testing.assert_close(decoded, values, rtol=1e-5, atol=1e-4)


def test_meta_targets_do_not_receive_gradient(module):
    """The rule is frozen here; a gradient reaching pi_hat would be meta-training."""
    agent_out = make_agent_out(11)
    actions = torch.randint(N_ACTIONS, (HORIZON + 1, BATCH))
    pi_hat = torch.randn(HORIZON, BATCH, N_ACTIONS, dtype=torch.float64, requires_grad=True)
    y_hat = torch.randn(HORIZON, BATCH, 600, dtype=torch.float64)
    z_hat = torch.randn(HORIZON, BATCH, 600, dtype=torch.float64)

    total, _ = module.disco_agent_loss(
        agent_out, actions, torch.zeros(HORIZON, BATCH, dtype=torch.float64),
        pi_hat, y_hat, z_hat, torch.zeros(HORIZON, BATCH, dtype=torch.float64),
        300.0, 1.0, 1.0, 1.0, 1.0, 0.2)
    total.sum().backward()
    assert agent_out['logits'].grad.abs().sum() > 0
    # pi_hat is an input to the KL, so it does get a gradient here; in the
    # trainer it is produced under no_grad, which is what freezes the rule.
    assert 'no_grad' in open(module.__file__).read()


# --------------------------------------------------------------------------
# Agent, optimizer, configuration
# --------------------------------------------------------------------------

def test_agent_outputs_match_the_update_rule_specs(module):
    """flat: logits, y. model (per action): z, aux_pi, q."""
    net = module.DiscoAgentNet(N_ACTIONS, num_bins=601)
    out = net.unroll(torch.zeros(3, 2, 4, 84, 84, dtype=torch.uint8))
    assert out['logits'].shape == (3, 2, N_ACTIONS)
    assert out['y'].shape == (3, 2, module.PREDICTION_SIZE)
    assert out['z'].shape == (3, 2, N_ACTIONS, module.PREDICTION_SIZE)
    assert out['aux_pi'].shape == (3, 2, N_ACTIONS, N_ACTIONS)
    assert out['q'].shape == (3, 2, N_ACTIONS, 601)


def test_action_model_expands_every_action_from_one_root(module):
    """One torso pass, then a single LSTM step per action -- Muesli style."""
    torch.manual_seed(12)
    net = module.DiscoAgentNet(N_ACTIONS, num_bins=11).double()
    observations = (torch.rand(4, 4, 84, 84, dtype=torch.float64) * 255)
    out = net(observations)

    torso = net.torso(net.conv(observations / 255.0))
    cell = net.root(torso)
    for row in range(4):
        for action in range(N_ACTIONS):
            hidden, _ = net.action_lstm(
                F.one_hot(torch.tensor([action]), N_ACTIONS).to(torso.dtype),
                torch.tanh(cell[row:row + 1]), cell[row:row + 1])
            torch.testing.assert_close(
                net.q_head(hidden)[0], out['q'][row, action], rtol=1e-10, atol=1e-10)


def test_clipped_adam_bounds_every_parameter_step(module):
    """optax.clip on the *update*: no parameter can move more than lr per step,
    whatever the gradient is."""
    parameter = torch.nn.Parameter(torch.zeros(3))
    optimizer = module.ClippedAdam([parameter], lr=0.1, max_abs_update=1.0)
    parameter.grad = torch.tensor([1e6, -1e6, 1e-9])
    optimizer.step()
    assert parameter.abs().max().item() <= 0.1 + 1e-6
    np.testing.assert_allclose(parameter.detach().numpy()[:2], [-0.1, 0.1], atol=1e-6)

    # A tighter clip scales the step down proportionally.
    other = torch.nn.Parameter(torch.zeros(1))
    tight = module.ClippedAdam([other], lr=0.1, max_abs_update=0.25)
    other.grad = torch.tensor([1e6])
    tight.step()
    np.testing.assert_allclose(other.item(), -0.025, atol=1e-7)


def test_hyperparameters_match_the_published_disco_settings(module):
    args = module.Args()
    assert args.learning_rate == 3e-4
    assert args.max_abs_update == 1.0
    assert args.discount == 0.997
    assert args.td_lambda == 0.95
    assert args.num_bins == 601
    assert args.max_abs_value == 300.0
    assert args.target_params_coeff == 0.9
    assert args.pi_cost == 1.0
    assert args.y_cost == 1.0
    assert args.z_cost == 1.0
    assert args.aux_policy_cost == 1.0
    assert args.value_cost == 0.2
    assert args.moving_average_decay == 0.99
    assert args.moving_average_eps == 1e-6
    assert args.head_w_init_std == 1e-2
    assert args.num_steps == 29, 'the reference evaluation loop uses rollouts of 29'


def test_meta_net_defaults_match_the_published_config(module):
    net = module.DiscoMetaNet()
    assert net.hidden_size == 256
    assert net.meta_hidden.shape == (128,)
    assert net.trajectory_lstm.linear.in_features == module.META_INPUT_SIZE + 256
    assert net.y_head.out_features == net.z_head.out_features == module.PREDICTION_SIZE
    assert net.policy_target_net.out_channels == 16
    assert net.encoder.policy_net.out_channels == 2


def test_both_variants_define_the_same_networks():
    eager, compiled = (load_trainer(name) for name in TRAINERS)
    for build in (lambda m: m.DiscoMetaNet(), lambda m: m.DiscoAgentNet(N_ACTIONS)):
        torch.manual_seed(13)
        a = build(eager)
        torch.manual_seed(13)
        b = build(compiled)
        for (name_a, pa), (name_b, pb) in zip(a.named_parameters(), b.named_parameters()):
            assert name_a == name_b
            torch.testing.assert_close(pa, pb, rtol=0, atol=0)


# --------------------------------------------------------------------------
# Cross-check against the real JAX implementation, when it is installed
# --------------------------------------------------------------------------

@published
def test_matches_jax_rlax_primitives(module, arrays):
    rlax = pytest.importorskip('rlax', reason='JAX stack not installed')
    import jax.numpy as jnp

    x = np.array([-5000.0, -1.0, 0.0, 0.25, 17.0, 12345.0], dtype=np.float32)
    np.testing.assert_allclose(
        module.signed_hyperbolic(torch.as_tensor(x)).numpy(),
        np.asarray(rlax.signed_hyperbolic(jnp.asarray(x))), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(
        module.signed_parabolic(torch.as_tensor(x)).numpy(),
        np.asarray(rlax.signed_parabolic(jnp.asarray(x))), rtol=1e-4, atol=1e-3)
    np.testing.assert_allclose(
        module.signed_logp1(torch.as_tensor(x)).numpy(),
        np.asarray(rlax.signed_logp1(jnp.asarray(x))), rtol=1e-5, atol=1e-5)

    scalars = np.array([0.0, 1.5, -7.25, 300.0], dtype=np.float32)
    np.testing.assert_allclose(
        module.transform_to_2hot(torch.as_tensor(scalars), -300.0, 300.0, 601).numpy(),
        np.asarray(rlax.transform_to_2hot(jnp.asarray(scalars), -300.0, 300.0, 601)),
        atol=1e-6)
