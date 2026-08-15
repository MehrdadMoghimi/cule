"""R2D2: the port checked against the paper's specification.

DeepMind released no R2D2 code, so there is nothing to diff against. These tests
pin the pieces the paper specifies exactly and that a reimplementation gets
wrong quietly:

  * the invertible value rescaling h and its inverse (a round-trip bug here just
    biases every target, it never raises);
  * the rescaled n-step double-Q target, against a hand-rolled NumPy version;
  * the mixed max/mean sequence priority;
  * the Ape-X per-actor epsilon ladder;
  * that burn-in really does keep gradients out of the first steps while leaving
    the forward pass identical;
  * that the sequence buffer serves contiguous slices with the recurrent state
    that was actually stored at the sampled row.
"""

import numpy as np
import pytest
import torch

from conftest import DiscreteEnvStub, load_trainer

TRAINERS = ['r2d2_atari', 'r2d2_atari_torchcompile']


@pytest.fixture(params=TRAINERS)
def module(request):
    return load_trainer(request.param)


def test_value_rescaling_matches_the_paper_formula(module):
    x = torch.linspace(-500, 500, 2001, dtype=torch.float64)
    eps = 1e-3
    expected = np.sign(x.numpy()) * (np.sqrt(np.abs(x.numpy()) + 1) - 1) + eps * x.numpy()
    np.testing.assert_allclose(
        module.signed_hyperbolic(x, eps).numpy(), expected, rtol=1e-12, atol=1e-12)


def test_value_rescaling_is_invertible(module):
    x = torch.linspace(-1000, 1000, 4001, dtype=torch.float64)
    for eps in (1e-3, 1e-2, 0.0):
        compressed = module.signed_hyperbolic(x, eps)
        np.testing.assert_allclose(
            module.signed_parabolic(compressed, eps).numpy(), x.numpy(), rtol=1e-8, atol=1e-8)


def test_value_rescaling_fixed_points(module):
    """h(0) = 0 and h is monotone increasing, or the ordering of values changes."""
    assert module.signed_hyperbolic(torch.zeros(1, dtype=torch.float64), 1e-3).item() == 0.0
    x = torch.linspace(-50, 50, 501, dtype=torch.float64)
    compressed = module.signed_hyperbolic(x, 1e-3)
    assert torch.all(compressed[1:] > compressed[:-1])
    # It really compresses: h(400) is order 20, not order 400.
    assert module.signed_hyperbolic(torch.tensor([400.0], dtype=torch.float64), 1e-3).item() < 25


def test_actor_epsilon_ladder_matches_apex(module):
    epsilons = module.actor_epsilons(8, 0.4, 7.0)
    expected = [0.4 ** (1 + i / 7 * 7.0) for i in range(8)]
    np.testing.assert_allclose(epsilons, expected, rtol=1e-6)
    # Actor 0 explores least, the last actor most; the ladder is monotone.
    assert np.all(np.diff(epsilons) < 0)
    np.testing.assert_allclose(epsilons[0], 0.4, rtol=1e-6)
    # A single actor gets the most exploratory setting, not the least.
    np.testing.assert_allclose(module.actor_epsilons(1, 0.4, 7.0)[0], 0.4 ** 8, rtol=1e-6)


def test_rescaled_n_step_double_q_target(module):
    """Against an independent NumPy computation of the paper's target."""
    torch.manual_seed(0)
    batch, n_actions = 3, 4
    burn_in, seq_len, n_step = 2, 5, 3
    horizon = burn_in + seq_len + n_step
    gamma, eps = 0.997, 1e-3

    q_online = torch.randn(batch, horizon, n_actions, dtype=torch.float64)
    q_target = torch.randn(batch, horizon, n_actions, dtype=torch.float64)
    actions = torch.randint(0, n_actions, (batch, horizon))
    rewards = torch.randn(batch, horizon, dtype=torch.float64)
    dones = (torch.rand(batch, horizon) < 0.15).double()

    targets, predictions = module.r2d2_targets(
        q_online, q_target, actions, rewards, dones,
        burn_in, seq_len, n_step, gamma, eps)

    def h(x):
        return np.sign(x) * (np.sqrt(np.abs(x) + 1) - 1) + eps * x

    def h_inverse(x):
        root = (np.sqrt(1 + 4 * eps * (np.abs(x) + 1 + eps)) - 1) / (2 * eps)
        return np.sign(x) * (root ** 2 - 1)

    q_online_np, q_target_np = q_online.numpy(), q_target.numpy()
    rewards_np, dones_np = rewards.numpy(), dones.numpy()
    for b in range(batch):
        for offset in range(seq_len):
            t = burn_in + offset
            accumulated, alive, discount = 0.0, 1.0, 1.0
            for k in range(n_step):
                accumulated += discount * rewards_np[b, t + k] * alive
                alive *= 1.0 - dones_np[b, t + k]
                discount *= gamma
            # Double Q: online net argmax, target net value.
            best = int(np.argmax(q_online_np[b, t + n_step]))
            bootstrap = h_inverse(q_target_np[b, t + n_step, best])
            expected = h(accumulated + discount * alive * bootstrap)
            np.testing.assert_allclose(targets[b, offset].item(), expected, rtol=1e-9, atol=1e-9)
            assert predictions[b, offset].item() == q_online_np[b, t, actions[b, t].item()]


def test_mixed_priority_formula(module):
    """p = eta * max_t |delta_t| + (1 - eta) * mean_t |delta_t|."""
    eta = 0.9
    td_errors = torch.tensor([[1.0, -4.0, 2.0], [0.5, 0.5, 0.5]], dtype=torch.float64)
    absolute = td_errors.abs()
    priorities = eta * absolute.max(dim=1).values + (1 - eta) * absolute.mean(dim=1)
    np.testing.assert_allclose(
        priorities.numpy(), [0.9 * 4.0 + 0.1 * (7 / 3), 0.9 * 0.5 + 0.1 * 0.5], rtol=1e-12)
    # The two summaries disagree on a spiky sequence, which is the whole point.
    assert absolute[0].max() != pytest.approx(absolute[0].mean())
    assert load_trainer(TRAINERS[0]).Args().priority_eta == eta


def test_burn_in_blocks_gradients_but_not_the_forward_pass(module):
    """Burn-in must change only the graph, never the values."""
    torch.manual_seed(1)
    net = module.R2D2Network(DiscreteEnvStub(4), lstm_size=16).double()
    batch, horizon = 2, 6
    observations = torch.randint(0, 256, (batch, horizon, 4, 84, 84)).double()
    previous_actions = torch.randint(0, 4, (batch, horizon))
    previous_rewards = torch.randn(batch, horizon, dtype=torch.float64)
    resets = torch.zeros(batch, horizon, dtype=torch.float64)
    state = net.initial_state(batch, torch.device('cpu'))

    full, _ = net.unroll(observations, previous_actions, previous_rewards, resets, state, grad_from=0)
    burned, _ = net.unroll(observations, previous_actions, previous_rewards, resets, state, grad_from=3)
    torch.testing.assert_close(full, burned, rtol=1e-9, atol=1e-9)

    # Only the graded tail is attached to the graph.
    net.zero_grad()
    burned[:, :3].sum().backward(retain_graph=True)
    assert all(p.grad is None or p.grad.abs().sum() == 0 for p in net.parameters())

    net.zero_grad()
    burned[:, 3:].sum().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in net.parameters())


def test_unroll_resets_the_recurrent_state_at_dones(module):
    """A step flagged as a reset must start from a zero state."""
    torch.manual_seed(2)
    net = module.R2D2Network(DiscreteEnvStub(3), lstm_size=8).double()
    batch, horizon = 1, 4
    observations = torch.randint(0, 256, (batch, horizon, 4, 84, 84)).double()
    previous_actions = torch.randint(0, 3, (batch, horizon))
    previous_rewards = torch.randn(batch, horizon, dtype=torch.float64)

    nonzero_state = (torch.randn(batch, 8, dtype=torch.float64),
                     torch.randn(batch, 8, dtype=torch.float64))
    resets = torch.zeros(batch, horizon, dtype=torch.float64)
    resets[:, 0] = 1.0
    with_reset, _ = net.unroll(
        observations, previous_actions, previous_rewards, resets, nonzero_state)

    zero_state = net.initial_state(batch, torch.device('cpu'))
    from_zero, _ = net.unroll(
        observations, previous_actions, previous_rewards,
        torch.zeros(batch, horizon, dtype=torch.float64), zero_state)

    torch.testing.assert_close(with_reset, from_zero, rtol=1e-9, atol=1e-9)


def test_lstm_sees_previous_action_and_reward(module):
    """R2D2 feeds a_{t-1} and r_{t-1} to the core; dropping them is a silent
    change of algorithm, so check they actually move the output."""
    torch.manual_seed(3)
    net = module.R2D2Network(DiscreteEnvStub(4), lstm_size=8).double()
    observations = torch.randint(0, 256, (2, 4, 84, 84)).double()
    state = net.initial_state(2, torch.device('cpu'))
    base, _ = net.step(observations, torch.zeros(2, dtype=torch.long),
                       torch.zeros(2, dtype=torch.float64), state)
    other_action, _ = net.step(observations, torch.full((2,), 3, dtype=torch.long),
                               torch.zeros(2, dtype=torch.float64), state)
    other_reward, _ = net.step(observations, torch.zeros(2, dtype=torch.long),
                               torch.full((2,), 5.0, dtype=torch.float64), state)
    assert not torch.allclose(base, other_action)
    assert not torch.allclose(base, other_reward)
    assert net.lstm.input_size == 512 + 4 + 1


def test_sequence_buffer_serves_contiguous_slices_and_stored_state(module):
    """The replayed sequence must be a real slice of one env's trajectory, and
    the state must be the one recorded at that row."""
    torch.manual_seed(4)
    n_envs, burn_in, seq_len, n_step, lstm_size = 3, 2, 4, 2, 5
    space = type('Box', (), {'shape': (4, 84, 84)})()
    action_space = type('Discrete', (), {'n': 4})()
    buffer = module.R2D2SequenceBuffer(
        4000, space, action_space, torch.device('cpu'), n_envs=n_envs,
        n_step=n_step, gamma=0.99, alpha=0.9, beta=0.6, eps=1e-6,
        burn_in=burn_in, seq_len=seq_len, lstm_size=lstm_size)
    assert buffer.sample_horizon == burn_in + seq_len + n_step

    observations = torch.zeros(n_envs, 4, 84, 84, dtype=torch.uint8)
    buffer.initialize(observations)
    for step in range(60):
        # A distinctive per-step state so it can be matched back after sampling.
        hidden = torch.full((n_envs, lstm_size), float(step))
        cell = torch.full((n_envs, lstm_size), -float(step))
        actions = torch.full((n_envs,), step % 4, dtype=torch.long)
        rewards = torch.full((n_envs,), float(step))
        dones = torch.zeros(n_envs, dtype=torch.bool)
        buffer.add(observations, actions, rewards, dones, hidden, cell)

    data = buffer.sample_sequences(16)
    horizon = burn_in + seq_len + n_step
    assert data['observations'].shape == (16, horizon, 4, 84, 84)
    assert data['actions'].shape == (16, horizon)
    assert data['hidden'].shape == (16, lstm_size)

    # rewards were written as the step index, so a contiguous slice increments
    # by exactly one each timestep.
    rewards = data['rewards'].numpy()
    np.testing.assert_allclose(np.diff(rewards, axis=1), 1.0)
    # The stored state at the sequence start matches that first step index.
    np.testing.assert_allclose(data['hidden'].numpy()[:, 0], rewards[:, 0])
    np.testing.assert_allclose(data['cell'].numpy()[:, 0], -rewards[:, 0])
    # previous_* are the step before the sequence starts, except where no such
    # transition exists (a sequence starting at step 0), which is treated as a
    # reset with a zeroed previous action and reward.
    previous = data['previous_rewards'].numpy()[:, 0]
    starts_at_zero = rewards[:, 0] == 0
    np.testing.assert_allclose(previous[~starts_at_zero], rewards[~starts_at_zero, 0] - 1.0)
    np.testing.assert_allclose(previous[starts_at_zero], 0.0)
    np.testing.assert_allclose(data['previous_dones'].numpy()[starts_at_zero, 0], 1.0)


def test_hyperparameters_match_paper(module):
    args = module.Args()
    assert args.gamma == 0.997
    assert args.n_step == 5
    assert args.learning_rate == 1e-4
    assert args.adam_eps == 1e-3, 'the paper uses an unusually large Adam epsilon'
    assert args.prioritized_replay_alpha == 0.9
    assert args.prioritized_replay_beta == 0.6
    assert args.priority_eta == 0.9
    assert args.rescale_eps == 1e-3
    assert args.actor_epsilon_base == 0.4 and args.actor_epsilon_alpha == 7.0
    # Rescaling replaces reward clipping.
    assert args.clip_rewards is False


def test_both_variants_agree():
    eager, compiled = (load_trainer(name) for name in TRAINERS)
    torch.manual_seed(7)
    a = eager.R2D2Network(DiscreteEnvStub(5), lstm_size=16)
    torch.manual_seed(7)
    b = compiled.R2D2Network(DiscreteEnvStub(5), lstm_size=16)
    observations = torch.randint(0, 256, (2, 4, 84, 84)).float()
    state = a.initial_state(2, torch.device('cpu'))
    previous_actions = torch.zeros(2, dtype=torch.long)
    previous_rewards = torch.zeros(2)
    torch.testing.assert_close(
        a.step(observations, previous_actions, previous_rewards, state)[0],
        b.step(observations, previous_actions, previous_rewards, state)[0],
        rtol=0, atol=0)

    x = torch.linspace(-100, 100, 201, dtype=torch.float64)
    torch.testing.assert_close(
        eager.signed_hyperbolic(x), compiled.signed_hyperbolic(x), rtol=0, atol=0)
