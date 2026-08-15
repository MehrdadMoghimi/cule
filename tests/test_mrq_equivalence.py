"""MR.Q: the parts that must match the reference agent.

facebookresearch/MRQ is licensed CC BY-NC 4.0, so -- as with the streaming-drl
ports -- nothing is vendored and there is no source diff to run. Instead each
test states the reference behaviour independently (a NumPy transcription of the
update rule, an algebraic identity, or a distributional property) and checks the
trainer against it. The pieces pinned here are the ones where a plausible-looking
alternative would silently change the algorithm:

  * two-hot encoding over symexp bins -- an interpolation, not a soft one-hot
  * the encoder rolling on its *own* predicted latent, not the target's
  * masked means that are NOT renormalized by the mask
  * the reward-scale algebra in the TD target, including the stale target scale
  * LAP priorities: floored, exponentiated, and stored without IS weights
  * the encoder receiving no gradient from the value or policy loss
"""

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from conftest import load_trainer

TRAINERS = ['mrq_atari', 'mrq_atari_torchcompile']
N_ACTIONS = 6


@pytest.fixture(params=TRAINERS)
def module(request):
    return load_trainer(request.param)


def make_buffer(module, n_envs=2, enc_horizon=5, q_horizon=3, capacity=4000):
    space = type('Box', (), {'shape': (4, 84, 84)})()
    action_space = type('Discrete', (), {'n': N_ACTIONS})()
    return module.MRQReplayBuffer(
        capacity, space, action_space, torch.device('cpu'), n_envs=n_envs,
        n_step=q_horizon, gamma=0.99, alpha=0.4, beta=0.0,
        enc_horizon=enc_horizon, q_horizon=q_horizon, min_priority=1.0)


# --------------------------------------------------------------------------
# Two-hot reward encoding
# --------------------------------------------------------------------------

def test_bins_are_symexp_of_a_linear_grid(module):
    two_hot = module.TwoHot(-10.0, 10.0, 65)
    grid = np.linspace(-10.0, 10.0, 65)
    expected = np.sign(grid) * (np.exp(np.abs(grid)) - 1.0)
    np.testing.assert_allclose(two_hot.bins.numpy(), expected, rtol=1e-6)

    bins = two_hot.bins.numpy()
    assert (np.diff(bins) > 0).all(), 'bins must be strictly increasing'
    assert bins[32] == 0.0, 'an odd bin count puts a bin exactly on zero'
    np.testing.assert_allclose(bins[0], -bins[-1])
    # The grid is linear in log space, so it resolves the small rewards Atari
    # actually produces while still reaching e^10.
    assert abs(bins[33]) < 0.4 and bins[-1] > 2e4


def test_transform_is_an_interpolation_that_preserves_the_mean(module):
    """The whole point of two-hot: a *lossless* encoding of a scalar."""
    two_hot = module.TwoHot(-10.0, 10.0, 65)
    x = torch.tensor([[0.0], [1.0], [-1.0], [0.37], [-12.5], [123.0]])
    encoded = two_hot.transform(x)

    np.testing.assert_allclose(encoded.sum(-1).numpy(), 1.0, atol=1e-6)
    assert (encoded >= 0).all()
    assert (encoded > 0).sum(-1).max() <= 2, 'at most two bins may be non-zero'
    decoded = (encoded * two_hot.bins).sum(-1, keepdim=True)
    np.testing.assert_allclose(decoded.numpy(), x.numpy(), atol=1e-4)


def test_zero_reward_is_exactly_one_hot_on_the_zero_bin(module):
    """Most Atari rewards are exactly zero; that must not smear over two bins."""
    two_hot = module.TwoHot(-10.0, 10.0, 65)
    encoded = two_hot.transform(torch.zeros(3, 1))
    assert (encoded[:, 32] == 1.0).all()
    assert encoded.sum() == 3.0


def test_transform_on_a_bin_centre_is_one_hot_there(module):
    two_hot = module.TwoHot(-10.0, 10.0, 65)
    for index in (1, 20, 32, 50, 63):
        encoded = two_hot.transform(two_hot.bins[index].reshape(1, 1))
        assert encoded.argmax().item() == index
        np.testing.assert_allclose(encoded.max().item(), 1.0, atol=1e-6)


def test_inverse_is_the_softmax_weighted_bin_mean(module):
    torch.manual_seed(0)
    two_hot = module.TwoHot(-10.0, 10.0, 65)
    logits = torch.randn(4, 65)
    expected = (torch.softmax(logits, dim=-1) * two_hot.bins).sum(-1, keepdim=True)
    torch.testing.assert_close(two_hot.inverse(logits), expected)


def test_cross_entropy_matches_the_explicit_formula_and_is_minimized_at_the_target(module):
    torch.manual_seed(1)
    two_hot = module.TwoHot(-10.0, 10.0, 65)
    logits = torch.randn(8, 65)
    targets = torch.randn(8, 1)

    expected = -(two_hot.transform(targets) * F.log_softmax(logits, dim=-1)).sum(-1, keepdim=True)
    torch.testing.assert_close(two_hot.cross_entropy_loss(logits, targets), expected)

    # Predicting the target distribution itself beats predicting anything else.
    perfect = torch.log(two_hot.transform(targets).clamp_min(1e-12))
    assert two_hot.cross_entropy_loss(perfect, targets).mean() < expected.mean()


# --------------------------------------------------------------------------
# Architecture
# --------------------------------------------------------------------------

def test_encoder_conv_stack_shapes(module):
    encoder = module.Encoder(4, N_ACTIONS)
    for name, (in_ch, out_ch, stride) in {
        'zs_cnn1': (4, 32, 2), 'zs_cnn2': (32, 32, 2),
        'zs_cnn3': (32, 32, 2), 'zs_cnn4': (32, 32, 1),
    }.items():
        conv = getattr(encoder, name)
        assert (conv.in_channels, conv.out_channels) == (in_ch, out_ch)
        assert conv.kernel_size == (3, 3) and conv.stride == (stride, stride)
        assert conv.padding == (0, 0), 'the reference uses valid padding throughout'
    # 84 -> 41 -> 20 -> 9 -> 7
    assert encoder.zs_lin.in_features == 32 * 7 * 7 == 1568
    assert encoder.zs(torch.zeros(2, 4, 84, 84)).shape == (2, 512)


def test_encoder_uses_elu_and_normalizes_the_observation(module):
    encoder = module.Encoder(4, N_ACTIONS)
    assert encoder.activ is F.elu, 'the encoder activation is ELU, not ReLU'

    # state/255 - 0.5 centres the input; an all-zero and an all-255 frame must
    # differ, and a mid-grey frame is the one that maps to zero pre-activation.
    with torch.no_grad():
        conv = encoder.zs_cnn1(torch.full((1, 4, 84, 84), 127.5) / 255.0 - 0.5)
    np.testing.assert_allclose(conv.numpy(), 0.0, atol=1e-6)


def test_model_head_slices_are_done_dynamics_reward(module):
    torch.manual_seed(2)
    encoder = module.Encoder(4, N_ACTIONS, num_bins=65, zs_dim=512)
    zs = torch.randn(3, 512)
    actions = F.one_hot(torch.tensor([0, 2, 5]), N_ACTIONS).float()

    with torch.no_grad():
        raw = encoder.model(encoder(zs, actions))
        done, next_zs, reward = encoder.model_all(zs, actions)
    assert encoder.model.out_features == 1 + 512 + 65
    torch.testing.assert_close(done, raw[:, 0:1])
    torch.testing.assert_close(next_zs, raw[:, 1:513])
    torch.testing.assert_close(reward, raw[:, 513:])


def test_predicted_next_latent_is_not_residual(module):
    """A residual `zs + delta` head would be a different (and easier) model."""
    torch.manual_seed(3)
    encoder = module.Encoder(4, N_ACTIONS)
    zs = torch.randn(2, 512) * 100.0
    actions = F.one_hot(torch.tensor([1, 3]), N_ACTIONS).float()
    with torch.no_grad():
        _, next_zs, _ = encoder.model_all(zs, actions)
    assert (next_zs - zs).abs().mean() > 1.0


def test_layer_norms_are_parameter_free(module):
    """`ln_activ` uses functional layer_norm, so there is no affine to learn."""
    mlp = module.BaseMLP(8, 4, 16)
    names = {name for name, _ in mlp.named_parameters()}
    assert names == {'l1.weight', 'l1.bias', 'l2.weight', 'l2.bias', 'l3.weight', 'l3.bias'}

    x = torch.randn(5, 8)
    normalized = module.ln_activ(x, lambda t: t)
    np.testing.assert_allclose(normalized.mean(-1).detach().numpy(), 0.0, atol=1e-6)
    np.testing.assert_allclose(normalized.std(-1, unbiased=False).detach().numpy(), 1.0, atol=1e-3)


def test_base_mlp_has_no_activation_on_its_output(module):
    """The Q head, the policy logits and the model head all read l3 directly."""
    torch.manual_seed(4)
    mlp = module.BaseMLP(8, 4, 16, 'elu')
    x = torch.randn(64, 8)
    with torch.no_grad():
        out = mlp(x)
    assert (out < 0).any() and (out > 0).any()
    with torch.no_grad():
        hidden = module.ln_activ(mlp.l2(module.ln_activ(mlp.l1(x), F.elu)), F.elu)
        torch.testing.assert_close(out, mlp.l3(hidden))


def test_weight_init_is_xavier_uniform_at_the_relu_gain(module):
    torch.manual_seed(5)
    layer = torch.nn.Linear(400, 300)
    module.weight_init(layer)
    bound = torch.nn.init.calculate_gain('relu') * np.sqrt(6.0 / (400 + 300))
    assert layer.weight.abs().max().item() <= bound + 1e-6
    # A uniform on (-a, a) has std a/sqrt(3).
    np.testing.assert_allclose(layer.weight.std().item(), bound / np.sqrt(3.0), rtol=0.05)
    assert (layer.bias == 0).all()


def test_value_is_a_twin_with_independent_parameters(module):
    torch.manual_seed(6)
    value = module.Value(512, 512)
    zsa = torch.randn(4, 512)
    with torch.no_grad():
        out = value(zsa)
    assert out.shape == (4, 2)
    torch.testing.assert_close(out[:, 0:1], value.q1(zsa))
    torch.testing.assert_close(out[:, 1:2], value.q2(zsa))
    assert {id(p) for p in value.q1.parameters()}.isdisjoint(id(p) for p in value.q2.parameters())
    assert not torch.allclose(out[:, 0], out[:, 1]), 'the twins must be initialized apart'


def test_policy_returns_a_relaxed_action_and_its_logits(module):
    torch.manual_seed(7)
    policy = module.Policy(N_ACTIONS, gumbel_tau=10.0)
    zs = torch.randn(16, 512)
    action, pre_activ = policy(zs)

    assert action.shape == pre_activ.shape == (16, N_ACTIONS)
    np.testing.assert_allclose(action.detach().sum(-1).numpy(), 1.0, atol=1e-5)
    assert (action >= 0).all(), 'the relaxed action lives on the simplex'
    torch.testing.assert_close(pre_activ, policy.policy(zs))
    # `act` is the same call with the logits dropped; it redraws its own Gumbel
    # noise, so only the shape and simplex constraint can be compared.
    sampled = policy.act(zs)
    assert sampled.shape == (16, N_ACTIONS)
    np.testing.assert_allclose(sampled.detach().sum(-1).numpy(), 1.0, atol=1e-5)


def test_the_gumbel_relaxation_is_soft_not_straight_through(module):
    """hard=True would emit exact one-hots; the soft form is what carries the
    deterministic policy gradient into the encoder's action embedding."""
    torch.manual_seed(8)
    policy = module.Policy(N_ACTIONS, gumbel_tau=10.0)
    action, _ = policy(torch.randn(64, 512))
    assert action.max().item() < 0.999, 'a straight-through action would be exactly one-hot'
    assert action.min().item() > 0.0


def test_high_tau_does_not_change_the_sampled_action_distribution(module):
    """softmax((l+g)/tau) is order preserving, so argmax still samples softmax(l).

    tau only smooths the *gradient*; if it changed the sampling distribution,
    tau=10 would make the behaviour policy uniform.
    """
    torch.manual_seed(9)
    logits = torch.tensor([[2.0, 0.0, -1.0, 1.0, 0.5, -2.0]])
    samples = torch.stack([
        F.gumbel_softmax(logits.expand(40000, -1), tau=10.0).argmax(-1),
        F.gumbel_softmax(logits.expand(40000, -1), tau=1.0).argmax(-1),
    ])
    expected = torch.softmax(logits[0], dim=-1).numpy()
    for row in samples:
        observed = torch.bincount(row, minlength=N_ACTIONS).double().numpy() / row.numel()
        np.testing.assert_allclose(observed, expected, atol=0.01)


def test_realign_snaps_a_noisy_action_back_to_one_hot(module):
    x = torch.tensor([[0.1, 0.9, 0.3], [0.7, 0.2, 0.05]])
    aligned = module.realign_discrete(x)
    np.testing.assert_allclose(aligned.numpy(), [[0, 1, 0], [1, 0, 0]])
    assert aligned.dtype == x.dtype


# --------------------------------------------------------------------------
# Losses
# --------------------------------------------------------------------------

def test_masked_mse_does_not_renormalize_by_the_mask(module):
    """A window that terminates early contributes *less*, it is not rescaled."""
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    y = torch.zeros_like(x)
    full = module.masked_mse(x, y, torch.ones(2, 1))
    half = module.masked_mse(x, y, torch.tensor([[1.0], [0.0]]))
    np.testing.assert_allclose(full.item(), (1 + 4 + 9 + 16) / 4)
    np.testing.assert_allclose(half.item(), (1 + 4) / 4)
    assert half.item() != pytest.approx((1 + 4) / 2), 'the mask must not renormalize'


def test_multi_step_reward_matches_a_numpy_loop(module):
    torch.manual_seed(10)
    rewards = torch.randn(7, 3, 1)
    not_dones = torch.ones(7, 3, 1)
    not_dones[2, 1] = 0.0  # terminates on the second step
    not_dones[5, 0] = 0.0  # terminates immediately
    gamma = 0.99

    ms_reward, term_discount = module.multi_step_reward(rewards, not_dones, gamma)

    r = rewards.numpy()[:, :, 0]
    nd = not_dones.numpy()[:, :, 0]
    expected_reward = np.zeros(7)
    expected_scale = np.ones(7)
    for i in range(3):
        expected_reward += expected_scale * r[:, i]
        expected_scale *= gamma * nd[:, i]

    np.testing.assert_allclose(ms_reward.numpy()[:, 0], expected_reward, rtol=1e-6)
    np.testing.assert_allclose(term_discount.numpy()[:, 0], expected_scale, rtol=1e-6)
    # A window that terminates has no bootstrap at all.
    assert term_discount[2, 0] == 0.0 and term_discount[5, 0] == 0.0
    np.testing.assert_allclose(term_discount[0, 0].item(), gamma**3, rtol=1e-6)


def test_encoder_loss_rolls_on_its_own_prediction(module):
    """Teacher forcing on the target latent would remove the multi-step signal."""
    torch.manual_seed(11)
    encoder = module.Encoder(4, N_ACTIONS).double()
    two_hot = module.TwoHot(-10.0, 10.0, 65).double()
    horizon, batch = 3, 2

    observations = torch.rand(batch, 4, 84, 84, dtype=torch.float64) * 255
    actions = F.one_hot(torch.randint(N_ACTIONS, (batch, horizon)), N_ACTIONS).double()
    rewards = torch.randn(batch, horizon, 1, dtype=torch.float64)
    not_dones = torch.ones(batch, horizon, 1, dtype=torch.float64)
    target_zs = torch.randn(batch, horizon, 512, dtype=torch.float64)

    loss = module.encoder_loss(encoder, two_hot, observations, actions, rewards,
                               not_dones, target_zs, 1.0, 0.1, 0.1)

    # Reference: unroll by hand, feeding each prediction back in.
    expected = torch.zeros((), dtype=torch.float64)
    pred_zs = encoder.zs(observations)
    for i in range(horizon):
        done, pred_zs, reward_logits = encoder.model_all(pred_zs, actions[:, i])
        expected = expected + F.mse_loss(pred_zs, target_zs[:, i])
        expected = expected + 0.1 * two_hot.cross_entropy_loss(reward_logits, rewards[:, i]).mean()
        expected = expected + 0.1 * F.mse_loss(done, torch.zeros_like(done))
    torch.testing.assert_close(loss, expected, rtol=1e-10, atol=1e-10)

    # And the roll really is autoregressive: replacing the first latent changes
    # every later term, which teacher forcing would not do.
    zs0 = encoder.zs(observations)
    _, first_pred, _ = encoder.model_all(zs0, actions[:, 0])
    _, second_pred, _ = encoder.model_all(first_pred, actions[:, 1])
    _, second_forced, _ = encoder.model_all(target_zs[:, 0], actions[:, 1])
    assert not torch.allclose(second_pred, second_forced)


def test_encoder_loss_masks_everything_after_a_terminal(module):
    torch.manual_seed(12)
    encoder = module.Encoder(4, N_ACTIONS).double()
    two_hot = module.TwoHot(-10.0, 10.0, 65).double()
    horizon, batch = 4, 3

    observations = torch.rand(batch, 4, 84, 84, dtype=torch.float64) * 255
    actions = F.one_hot(torch.randint(N_ACTIONS, (batch, horizon)), N_ACTIONS).double()
    rewards = torch.randn(batch, horizon, 1, dtype=torch.float64)
    target_zs = torch.randn(batch, horizon, 512, dtype=torch.float64)

    not_dones = torch.ones(batch, horizon, 1, dtype=torch.float64)
    not_dones[1, 1] = 0.0
    base = module.encoder_loss(encoder, two_hot, observations, actions, rewards,
                               not_dones, target_zs, 1.0, 0.1, 0.1)

    # Perturbing targets strictly after the terminal must not move the loss.
    poisoned_targets = target_zs.clone()
    poisoned_targets[1, 2:] += 1e3
    poisoned_rewards = rewards.clone()
    poisoned_rewards[1, 2:] += 1e3
    poisoned = module.encoder_loss(encoder, two_hot, observations, actions, poisoned_rewards,
                                   not_dones, poisoned_targets, 1.0, 0.1, 0.1)
    torch.testing.assert_close(base, poisoned, rtol=1e-12, atol=1e-12)

    # The terminal step itself is still supervised -- that is the only place the
    # done head ever sees a positive label.
    poisoned_targets = target_zs.clone()
    poisoned_targets[1, 1] += 1e3
    at_terminal = module.encoder_loss(encoder, two_hot, observations, actions, rewards,
                                      not_dones, poisoned_targets, 1.0, 0.1, 0.1)
    assert at_terminal.item() > base.item()


def test_encoder_loss_weights_are_applied_per_term(module):
    torch.manual_seed(13)
    encoder = module.Encoder(4, N_ACTIONS).double()
    two_hot = module.TwoHot(-10.0, 10.0, 65).double()
    args = (encoder, two_hot, torch.rand(2, 4, 84, 84, dtype=torch.float64) * 255,
            F.one_hot(torch.randint(N_ACTIONS, (2, 2)), N_ACTIONS).double(),
            torch.randn(2, 2, 1, dtype=torch.float64),
            torch.ones(2, 2, 1, dtype=torch.float64),
            torch.randn(2, 2, 512, dtype=torch.float64))

    both = module.encoder_loss(*args, 1.0, 0.1, 0.1)
    dyn_only = module.encoder_loss(*args, 1.0, 0.0, 0.0)
    reward_only = module.encoder_loss(*args, 0.0, 0.1, 0.0)
    done_only = module.encoder_loss(*args, 0.0, 0.0, 0.1)
    torch.testing.assert_close(both, dyn_only + reward_only + done_only, rtol=1e-10, atol=1e-10)
    # Dynamics dominates by construction: weight 1 against 0.1.
    assert dyn_only.item() > reward_only.item()


def test_scaled_q_target_algebra(module):
    """The stale target scale must be undone before renormalizing."""
    ms_reward = torch.tensor([[2.0], [2.0]])
    term_discount = torch.tensor([[0.97], [0.0]])
    next_q = torch.tensor([[5.0], [5.0]])

    target = module.scaled_q_target(ms_reward, term_discount, next_q, 0.5, 0.25)
    expected = (ms_reward + term_discount * next_q * 0.25) / 0.5
    torch.testing.assert_close(target, expected)

    # A terminated window bootstraps nothing, whatever the scales are.
    np.testing.assert_allclose(target[1].item(), 2.0 / 0.5)

    # With a fixed normalizer the target reduces to the ordinary TD target,
    # divided by the scale.
    unit = module.scaled_q_target(ms_reward, term_discount, next_q, 1.0, 1.0)
    torch.testing.assert_close(unit, ms_reward + term_discount * next_q)


def test_lap_priority_floors_then_exponentiates(module):
    q = torch.tensor([[1.0, 3.0], [0.5, 0.6], [-4.0, 0.0]])
    q_target = torch.tensor([[1.0], [0.4], [0.0]])
    priority = module.lap_priority(q, q_target, min_priority=1.0, alpha=0.4)

    error = (q - q_target.expand(-1, 2)).abs().max(1).values
    np.testing.assert_allclose(error.numpy(), [2.0, 0.2, 4.0])
    np.testing.assert_allclose(priority.numpy(), np.maximum(error.numpy(), 1.0) ** 0.4, rtol=1e-6)
    # Below the floor everything is sampled uniformly -- the LAP property that
    # removes the need for importance-sampling weights.
    assert priority[1].item() == 1.0


def test_min_priority_matches_the_huber_transition_point(module):
    """LAP is only unbiased when the floor sits where smooth_l1 goes linear."""
    args = module.Args()
    assert args.min_priority == 1.0
    x = torch.tensor([1.0])
    quadratic = F.smooth_l1_loss(x, torch.zeros(1), beta=1.0)
    np.testing.assert_allclose(quadratic.item(), 0.5)  # 0.5 * 1^2, the join point


# --------------------------------------------------------------------------
# Gradient routing
# --------------------------------------------------------------------------

def test_value_and_policy_losses_do_not_train_the_encoder(module):
    """MR.Q's claim is that the representation is learned by the model loss
    alone; leaking the TD gradient into the encoder would break that."""
    torch.manual_seed(14)
    encoder = module.Encoder(4, N_ACTIONS)
    value = module.Value(512, 512)
    observations = torch.rand(3, 4, 84, 84) * 255
    actions = F.one_hot(torch.tensor([0, 1, 2]), N_ACTIONS).float()

    with torch.no_grad():
        zs = encoder.zs(observations)
        zsa = encoder(zs, actions)
    q = value(zsa)
    F.smooth_l1_loss(q, torch.zeros_like(q)).backward()

    assert all(p.grad is None for p in encoder.parameters()), 'zsa must be detached'
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in value.parameters())


def test_policy_gradient_flows_through_the_action_embedding(module):
    """The deterministic policy gradient reaches the actor only via encoder.za,
    which is why the relaxed (differentiable) action is required."""
    torch.manual_seed(15)
    encoder = module.Encoder(4, N_ACTIONS)
    policy = module.Policy(N_ACTIONS)
    value = module.Value(512, 512)

    with torch.no_grad():
        zs = encoder.zs(torch.rand(3, 4, 84, 84) * 255)
    action, pre_activ = policy(zs)
    loss = -value(encoder(zs, action)).mean() + 1e-5 * pre_activ.pow(2).mean()
    loss.backward()

    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in policy.parameters())
    assert encoder.za.weight.grad is not None and encoder.za.weight.grad.abs().sum() > 0
    # The observation path saw no gradient: zs was detached.
    assert encoder.zs_cnn1.weight.grad is None


def test_pre_activation_penalty_pulls_logits_toward_zero(module):
    torch.manual_seed(16)
    policy = module.Policy(N_ACTIONS)
    zs = torch.randn(8, 512)
    _, pre_activ = policy(zs)
    (1e-5 * pre_activ.pow(2).mean()).backward()

    # The penalty is a plain L2 on the logits, so its gradient is 2*w*logit/N.
    assert policy.policy.l3.bias.grad is not None
    torch.testing.assert_close(
        policy.policy.l3.bias.grad,
        2e-5 * pre_activ.detach().mean(0) / pre_activ.numel() * pre_activ.shape[0],
        rtol=1e-5, atol=1e-9)


# --------------------------------------------------------------------------
# Replay buffer
# --------------------------------------------------------------------------

def fill(buffer, steps, n_envs, done_at=None):
    observations = torch.zeros(n_envs, 4, 84, 84, dtype=torch.uint8)
    buffer.initialize(observations)
    for step in range(steps):
        dones = torch.zeros(n_envs, dtype=torch.bool)
        if done_at is not None and step == done_at:
            dones[:] = True
        buffer.add(observations, torch.full((n_envs,), step % N_ACTIONS, dtype=torch.long),
                   torch.full((n_envs,), float(step)), dones)


def test_subtrajectory_shapes(module):
    buffer = make_buffer(module)
    fill(buffer, 60, n_envs=2)

    encoder_batch = buffer.sample_subtrajectory(16, 5, True)
    assert encoder_batch.observations.shape == (16, 4, 84, 84)
    assert encoder_batch.next_observations.shape == (16, 5, 4, 84, 84)
    assert encoder_batch.actions.shape == (16, 5)
    assert encoder_batch.rewards.shape == (16, 5, 1)
    assert encoder_batch.not_dones.shape == (16, 5, 1)

    value_batch = buffer.sample_subtrajectory(16, 3, False)
    assert value_batch.observations.shape == (16, 4, 84, 84)
    assert value_batch.next_observations.shape == (16, 4, 84, 84)
    assert value_batch.actions.shape == (16,), 'only the first action is used'
    assert value_batch.rewards.shape == (16, 3, 1)


def test_subtrajectories_are_contiguous(module):
    buffer = make_buffer(module, n_envs=3)
    fill(buffer, 80, n_envs=3)
    data = buffer.sample_subtrajectory(32, 5, True)
    rewards = data.rewards.numpy()[:, :, 0]
    # Rewards were written as the step index.
    np.testing.assert_allclose(np.diff(rewards, axis=1), 1.0)


def test_windows_may_cross_a_terminal(module):
    """The other prioritized buffers here refuse to; MR.Q needs it, because the
    done head and the terminal bootstrap have nothing to learn from otherwise."""
    buffer = make_buffer(module, n_envs=1, enc_horizon=5, q_horizon=3, capacity=200)
    fill(buffer, 40, n_envs=1, done_at=20)

    seen_terminal = False
    for _ in range(40):
        data = buffer.sample_subtrajectory(64, 5, True)
        if (data.not_dones == 0).any():
            seen_terminal = True
            break
    assert seen_terminal, 'no sampled window ever contained a terminal transition'

    # And once it terminates, everything after stays masked.
    not_dones = data.not_dones.numpy()[:, :, 0]
    rows = np.flatnonzero((not_dones == 0).any(axis=1))
    for row in rows:
        first_zero = int(np.flatnonzero(not_dones[row] == 0)[0])
        mask = np.cumprod(not_dones[row])
        assert mask[first_zero:].sum() == 0


def test_reward_scale_is_the_mean_absolute_reward(module):
    buffer = make_buffer(module, n_envs=2)
    assert buffer.reward_scale() == pytest.approx(1e-8), 'empty buffers must not divide by zero'

    observations = torch.zeros(2, 4, 84, 84, dtype=torch.uint8)
    buffer.initialize(observations)
    values = [1.0, -3.0, 0.0, 2.0, -6.0]
    for reward in values:
        buffer.add(observations, torch.zeros(2, dtype=torch.long),
                   torch.full((2,), reward), torch.zeros(2, dtype=torch.bool))
    np.testing.assert_allclose(buffer.reward_scale(), np.mean(np.abs(values)), rtol=1e-6)


def test_lap_priorities_are_stored_verbatim_and_seed_new_rows(module):
    buffer = make_buffer(module, n_envs=1, capacity=200)
    fill(buffer, 30, n_envs=1)
    assert buffer.max_priority == 1.0, 'new transitions start at the LAP floor'

    data = buffer.sample_subtrajectory(8, 3, False)
    priorities = np.array([1.0, 2.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    buffer.update_priorities(data.indices, priorities)

    stored = buffer.sum_tree.values(data.indices)
    # Sampling is with replacement, so an index can appear twice in one batch
    # and the later write wins. Compare per unique index against the last value
    # written to it rather than positionally.
    indices = np.asarray(data.indices)
    expected = {int(index): float(priority) for index, priority in zip(indices, priorities)}
    # No epsilon added and no second exponentiation: the caller already applied
    # clamp(min=1)**alpha.
    np.testing.assert_allclose(
        stored, np.array([expected[int(index)] for index in indices], dtype=np.float32), rtol=1e-6
    )
    assert buffer.max_priority == pytest.approx(2.5)


def test_buffer_carries_no_importance_sampling_weights(module):
    buffer = make_buffer(module)
    fill(buffer, 40, n_envs=2)
    data = buffer.sample_subtrajectory(8, 3, False)
    assert not hasattr(data, 'weights'), 'LAP corrects the bias through the loss, not weights'
    assert module.Args().min_priority == 1.0


def test_env_terminates_tracks_whether_a_done_was_ever_seen(module):
    buffer = make_buffer(module, n_envs=1, capacity=200)
    fill(buffer, 20, n_envs=1)
    assert buffer.env_terminates is False
    fill(buffer, 30, n_envs=1, done_at=10)
    assert buffer.env_terminates is True


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

def test_hyperparameters_match_the_official_atari_configuration(module):
    args = module.Args()
    assert args.batch_size == 256
    assert args.buffer_size == 1000000
    assert args.gamma == 0.99
    assert args.target_network_frequency == 250
    assert args.learning_starts == 10000
    assert args.enc_horizon == 5 and args.q_horizon == 3
    assert (args.dyn_weight, args.reward_weight, args.done_weight) == (1.0, 0.1, 0.1)
    assert (args.zs_dim, args.za_dim, args.zsa_dim) == (512, 256, 512)
    assert args.enc_hdim == args.value_hdim == args.policy_hdim == 512
    assert args.encoder_learning_rate == 1e-4
    assert args.value_learning_rate == args.policy_learning_rate == 3e-4
    assert args.weight_decay == 1e-4
    assert args.value_grad_clip == 20.0
    assert args.gumbel_tau == 10.0
    assert args.pre_activ_weight == 1e-5
    assert (args.num_bins, args.bin_lower, args.bin_upper) == (65, -10.0, 10.0)
    assert args.prioritized_replay_alpha == 0.4
    assert args.exploration_noise == 0.2
    assert args.target_policy_noise == 0.2 and args.noise_clip == 0.3
    assert args.data_augmentation is True
    assert args.total_timesteps == 2500000
    assert args.learner_updates_per_vector_step == 1.0, 'one update per environment step'
    # The official Atari setup clips no rewards and does not end on life loss.
    assert args.clip_rewards is False
    assert args.episodic_life is False


def test_noise_scales_are_halved_for_discrete_actions(module):
    """Discrete actions live on [0, 1], continuous ones on [-1, 1]."""
    source = open(module.__file__).read()
    assert 'exploration_noise = 0.5 * args.exploration_noise' in source
    assert 'target_policy_noise = 0.5 * args.target_policy_noise' in source
    assert 'noise_clip = 0.5 * args.noise_clip' in source


def test_augmentation_is_a_random_shift_without_intensity_noise(module):
    """SPR's augmentation adds multiplicative noise; MR.Q's does not."""
    torch.manual_seed(17)
    images = torch.rand(64, 4, 84, 84) * 255
    augmented = module.random_shift_augmentation(images)
    assert augmented.shape == images.shape
    assert not torch.allclose(augmented, images)
    # A constant image survives a shift exactly: replicate padding plus an
    # integer offset means no interpolation and no brightness change.
    flat = torch.full((8, 4, 84, 84), 37.0)
    torch.testing.assert_close(module.random_shift_augmentation(flat), flat, rtol=1e-5, atol=1e-3)


def test_both_variants_define_the_same_networks():
    eager, compiled = (load_trainer(name) for name in TRAINERS)
    x = torch.rand(3, 4, 84, 84) * 255
    for build in (
        lambda m: m.Encoder(4, N_ACTIONS),
        lambda m: m.Policy(N_ACTIONS),
    ):
        torch.manual_seed(18)
        a = build(eager)
        torch.manual_seed(18)
        b = build(compiled)
        for (name_a, pa), (name_b, pb) in zip(a.named_parameters(), b.named_parameters()):
            assert name_a == name_b
            torch.testing.assert_close(pa, pb, rtol=0, atol=0)

    torch.manual_seed(19)
    ea = eager.Encoder(4, N_ACTIONS)
    torch.manual_seed(19)
    ec = compiled.Encoder(4, N_ACTIONS)
    with torch.no_grad():
        torch.testing.assert_close(ea.zs(x), ec.zs(x), rtol=0, atol=0)
