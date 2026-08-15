"""NumPy transcription of DiscoRL's Haiku meta-network, for equivalence tests.

Transcribed from the official implementation
(https://github.com/google-deepmind/disco_rl, Apache-2.0), specifically
`disco_rl/networks/meta_nets.py`, `disco_rl/update_rules/input_transforms.py`
and `disco_rl/update_rules/disco.py`, plus the rlax primitives they call. It is
written to be structurally *different* from `cleanrl/disco_atari.py` -- the
input construction here is driven by a literal transcription of the published
`get_input_option()` table rather than by explicit code -- so that the two
agreeing is evidence, not tautology.

Only what the tests need is transcribed: the forward pass with published
weights. Nothing here is imported by the trainers.

Haiku parameter paths, and how they map onto the network, follow module creation
order inside `meta_nets.LSTM.__call__`:

    lstm/mlp, lstm/mlp_1                    y_net, z_net          (outer)
    lstm/sequential                         policy_net            (outer)
    lstm/lstm                               per-trajectory LSTM (reverse)
    lstm/linear                             lifetime-state projection
    lstm/linear_1                           meta_input_emb
    lstm/linear_2, lstm/linear_3            y_hat, z_hat
    lstm/sequential_1, lstm/linear_4        policy-target conv stack, pi_hat
    lstm/~/meta_lstm/~unroll/...            everything inside MetaLSTM.unroll
"""

import numpy as np

# The published Disco103 input option, transcribed verbatim from
# `disco.get_input_option()`. `source` is a path into the inputs mapping.
BASE_INPUTS = (
    ('agent_out/logits', ('drop_last', 'softmax', 'stop_grad', 'select_a')),
    ('behaviour_agent_out/logits', ('drop_last', 'softmax', 'stop_grad', 'select_a')),
    ('rewards', ('sign_log',)),
    ('is_terminal', ('masks_to_discounts',)),
    ('extra_from_rule/v_scalar', ('sign_log', 'td_pair', 'stop_grad')),
    ('extra_from_rule/adv', ('sign_log', 'stop_grad')),
    ('extra_from_rule/normalized_adv', ('stop_grad',)),
    ('extra_from_rule/target_out/logits', ('drop_last', 'softmax', 'stop_grad', 'select_a')),
    ('agent_out/y', ('softmax', 'y_net', 'td_pair')),
    ('extra_from_rule/target_out/y', ('softmax', 'y_net', 'td_pair')),
    ('agent_out/z', ('drop_last', 'softmax', 'z_net', 'select_a')),
    ('agent_out/z', ('softmax', 'z_net', 'pi_weighted_avg', 'td_pair')),
    ('agent_out/z', ('softmax', 'z_net', 'max_a', 'td_pair')),
    ('extra_from_rule/target_out/z', ('drop_last', 'softmax', 'z_net', 'select_a')),
    ('extra_from_rule/target_out/z', ('softmax', 'z_net', 'pi_weighted_avg', 'td_pair')),
    ('extra_from_rule/target_out/z', ('softmax', 'z_net', 'max_a', 'td_pair')),
)

ACTION_CONDITIONAL_INPUTS = (
    ('agent_out/logits', ('drop_last', 'softmax', 'stop_grad')),
    ('behaviour_agent_out/logits', ('drop_last', 'softmax', 'stop_grad')),
    ('extra_from_rule/target_out/logits', ('drop_last', 'softmax', 'stop_grad')),
    ('agent_out/z', ('drop_last', 'softmax', 'z_net')),
    ('extra_from_rule/target_out/z', ('drop_last', 'softmax', 'z_net')),
    ('extra_from_rule/q', ('sign_log', 'drop_last', 'stop_grad')),
    ('extra_from_rule/qv_adv', ('sign_log', 'drop_last', 'stop_grad')),
    ('extra_from_rule/normalized_qv_adv', ('drop_last', 'stop_grad')),
)

# `_construct_input` expands these to a trailing singleton before transforming.
EXPAND_LAST = ('v_scalar', 'adv', 'normalized_adv', 'q', 'qv_adv', 'normalized_qv_adv')


def softmax(x, axis=-1):
    shifted = x - x.max(axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=axis, keepdims=True)


def log_softmax(x, axis=-1):
    shifted = x - x.max(axis=axis, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=axis, keepdims=True))


def signed_logp1(x):
    return np.sign(x) * np.log1p(np.abs(x))


def signed_hyperbolic(x, eps=1e-3):
    return np.sign(x) * (np.sqrt(np.abs(x) + 1.0) - 1.0) + eps * x


def signed_parabolic(x, eps=1e-3):
    root = (np.sqrt(1.0 + 4.0 * eps * (np.abs(x) + 1.0 + eps)) - 1.0) / (2.0 * eps)
    return np.sign(x) * (root**2 - 1.0)


def transform_to_2hot(scalar, min_value, max_value, num_bins):
    scalar = np.clip(scalar, min_value, max_value)
    position = (scalar - min_value) / (max_value - min_value) * (num_bins - 1)
    lower, upper = np.floor(position), np.ceil(position)
    lower_value = lower / (num_bins - 1.0) * (max_value - min_value) + min_value
    upper_value = upper / (num_bins - 1.0) * (max_value - min_value) + min_value
    p_lower = (upper_value - scalar) / (upper_value - lower_value + 1e-5)
    p_upper = 1.0 - p_lower

    probs = np.zeros((*scalar.shape, num_bins), dtype=np.float64)
    flat = probs.reshape(-1, num_bins)
    for index, (low, high, weight_low, weight_high) in enumerate(
        zip(lower.reshape(-1).astype(int), upper.reshape(-1).astype(int),
            p_lower.reshape(-1), p_upper.reshape(-1))
    ):
        flat[index, low] += weight_low
        flat[index, high] += weight_high
    return probs


def transform_from_2hot(probs, min_value, max_value, num_bins):
    return (probs * np.linspace(min_value, max_value, num_bins)).sum(-1)


def categorical_kl_divergence(p_logits, q_logits):
    p = softmax(p_logits)
    kl = (p * (log_softmax(p_logits) - log_softmax(q_logits))).sum(-1)
    return np.maximum(kl, 0.0)


def batch_lookup(table, actions):
    """Index axis 2 of [T, B, A, ...] with [T, B]."""
    index = actions.reshape(*actions.shape, 1, *([1] * (table.ndim - 3)))
    index = np.broadcast_to(index, (*actions.shape, 1, *table.shape[3:]))
    return np.take_along_axis(table, index, axis=2)[:, :, 0]


def retrace_from_q_and_v(q_t, v_t, r_t, discount_t, c_t):
    """G_t = r_t + d_t * (v_t - c_t q_t + c_t G_{t+1}); rlax's recursion."""
    returns = [r_t[-1] + discount_t[-1] * v_t[-1]]
    for index in reversed(range(q_t.shape[0])):
        returns.insert(
            0,
            r_t[index]
            + discount_t[index] * (v_t[index] - c_t[index] * q_t[index] + c_t[index] * returns[0]),
        )
    return np.stack(returns, axis=0)


# --------------------------------------------------------------------------
# Haiku modules
# --------------------------------------------------------------------------

def linear(x, arrays, prefix):
    weight = arrays[f'{prefix}/w']
    if weight.ndim == 3:  # hk.Conv1D, kernel_shape=1
        weight = weight[0]
    return x @ weight.astype(np.float64) + arrays[f'{prefix}/b'].astype(np.float64)


def mlp(x, arrays, prefix, depth):
    for index in range(depth):
        x = linear(x, arrays, f'{prefix}/~/linear_{index}')
        if index < depth - 1:
            x = np.maximum(x, 0.0)
    return x


def conv1d_net(x, arrays, prefix, depth):
    """Each block concatenates the action-mean, then a 1x1 conv and a ReLU."""
    for index in range(depth):
        pooled = np.broadcast_to(x.mean(axis=2, keepdims=True), x.shape)
        suffix = '' if index == 0 else f'_{index}'
        x = np.maximum(
            linear(np.concatenate([x, pooled], axis=-1), arrays, f'{prefix}/conv1_d{suffix}'), 0.0
        )
    return x


def lstm_step(x, hidden, cell, arrays, prefix):
    """hk.LSTM: gates ordered i, g, f, o, with a +1 forget bias."""
    gates = linear(np.concatenate([x, hidden], axis=-1), arrays, prefix)
    size = gates.shape[-1] // 4
    i, g, f, o = (gates[..., k * size:(k + 1) * size] for k in range(4))
    sigmoid = lambda t: 1.0 / (1.0 + np.exp(-t))
    new_cell = sigmoid(f + 1.0) * cell + sigmoid(i) * np.tanh(g)
    return sigmoid(o) * np.tanh(new_cell), new_cell


# --------------------------------------------------------------------------
# Meta-network forward pass
# --------------------------------------------------------------------------

def _extract(inputs, source):
    node = inputs
    for key in source.split('/'):
        node = node[key]
    return np.asarray(node, dtype=np.float64)


def construct_input(inputs, arrays, prefix, mlp_names, seq_name, embedding_depth, policy_depth):
    """`meta_nets._construct_input`, driven by the transcribed option table."""
    actions = np.asarray(inputs['actions'])[:-1]
    policy = softmax(_extract(inputs, 'agent_out/logits'))
    horizon, batch = np.asarray(inputs['rewards']).shape
    n_actions = policy.shape[2]

    def apply(source, transforms, prefix_shape):
        x = _extract(inputs, source)
        leaf = source.split('/')[-1]
        if source.startswith('extra_from_rule') and leaf in EXPAND_LAST:
            x = x[..., None]
        for name in transforms:
            if name == 'y_net':
                x = mlp(x, arrays, f'{prefix}/{mlp_names[0]}', embedding_depth)
            elif name == 'z_net':
                x = mlp(x, arrays, f'{prefix}/{mlp_names[1]}', embedding_depth)
            elif name == 'softmax':
                x = softmax(x)
            elif name == 'drop_last':
                x = x[:-1]
            elif name == 'stop_grad':
                pass
            elif name == 'select_a':
                x = batch_lookup(x, actions)
            elif name == 'sign_log':
                x = signed_logp1(x)
            elif name == 'masks_to_discounts':
                x = 1.0 - x
            elif name == 'td_pair':
                x = np.concatenate([x[:-1], x[1:]], axis=-1)
            elif name == 'pi_weighted_avg':
                x = (x * policy[..., None]).sum(axis=2)
            elif name == 'max_a':
                x = x.max(axis=2)
            else:
                raise KeyError(name)
        return x.reshape(*prefix_shape, -1)

    base = [apply(source, txs, (horizon, batch)) for source, txs in BASE_INPUTS]
    action_conditional = [
        apply(source, txs, (horizon, batch, n_actions))
        for source, txs in ACTION_CONDITIONAL_INPUTS
    ]
    one_hot = np.eye(n_actions, dtype=np.float64)[actions][..., None]
    action_conditional.append(one_hot)

    policy_emb = conv1d_net(
        np.concatenate(action_conditional, axis=-1), arrays, f'{prefix}/{seq_name}', policy_depth
    )
    base.append(policy_emb.mean(axis=2))
    base.append(batch_lookup(policy_emb, actions))
    return np.concatenate(base, axis=-1), policy_emb


def meta_net_forward(inputs, arrays, meta_hidden, meta_cell, hidden_size=256):
    """Returns (pi_hat, y_hat, z_hat, new_meta_hidden, new_meta_cell)."""
    x, policy_emb = construct_input(
        inputs, arrays, 'lstm', ('mlp', 'mlp_1'), 'sequential',
        embedding_depth=2, policy_depth=2)
    horizon, batch, _ = x.shape
    is_terminal = np.asarray(inputs['is_terminal'], dtype=np.float64)

    # hk.ResetCore inside hk.dynamic_unroll(reverse=True).
    hidden = np.zeros((batch, hidden_size))
    cell = np.zeros((batch, hidden_size))
    outputs = [None] * horizon
    for step in reversed(range(horizon)):
        keep = (1.0 - is_terminal[step])[:, None]
        hidden, cell = lstm_step(x[step], hidden * keep, cell * keep, arrays, 'lstm/lstm/linear')
        outputs[step] = hidden
    x = np.stack(outputs, axis=0)

    # Multiplicative interaction with the lifetime state as it was before.
    x = x * linear(meta_hidden, arrays, 'lstm/linear')

    meta_input_emb = linear(x, arrays, 'lstm/linear_1')
    y_hat = linear(x, arrays, 'lstm/linear_2')
    z_hat = linear(x, arrays, 'lstm/linear_3')

    n_actions = policy_emb.shape[2]
    w = np.broadcast_to(x[:, :, None, :], (*x.shape[:2], n_actions, x.shape[-1]))
    w = np.concatenate([w, policy_emb], axis=-1)
    w = conv1d_net(w, arrays, 'lstm/sequential_1', 1)
    pi_hat = linear(w, arrays, 'lstm/linear_4')[..., 0]

    meta_prefix = 'lstm/~/meta_lstm/~unroll'
    meta_inputs, _ = construct_input(
        inputs, arrays, meta_prefix, ('mlp', 'mlp_1'), 'sequential',
        embedding_depth=2, policy_depth=2)
    pooled = np.concatenate(
        [meta_inputs, meta_input_emb, mlp(softmax(y_hat), arrays, f'{meta_prefix}/mlp', 2)], axis=-1)
    pooled = mlp(pooled, arrays, f'{meta_prefix}/mlp_2', 1).mean(axis=(0, 1))
    new_hidden, new_cell = lstm_step(
        pooled, meta_hidden, meta_cell, arrays, f'{meta_prefix}/lstm/linear')
    return pi_hat, y_hat, z_hat, new_hidden, new_cell
