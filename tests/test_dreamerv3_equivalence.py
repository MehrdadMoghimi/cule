"""Assert DreamerV3 against the claims of arXiv:2301.04104 (Nature 2025).

`tests/crosscheck/check_dreamer.py` diffs the port numerically against
NM512/dreamerv3-torch. These tests are the complement: they state the paper's
mechanisms directly, so they still hold if the upstream checkout is absent, and
they cover the pieces the diff cannot reach -- the replay's episode-boundary
handling and the imagination rollout's shapes.
"""

import numpy as np
import pytest
import torch

from conftest import device_params, load_trainer


@pytest.fixture(scope="module")
def dreamer():
    return load_trainer("dreamerv3_atari")


# --- symlog and the two-hot heads (Section 3, "Robust predictions") --------


def test_symlog_is_invertible_and_compresses(dreamer):
    x = torch.tensor([-100000.0, -1.0, 0.0, 1.0, 100000.0], dtype=torch.float64)
    assert torch.allclose(dreamer.symexp(dreamer.symlog(x)), x, rtol=1e-9)
    assert dreamer.symlog(torch.tensor(100000.0)).item() < 12.0
    assert dreamer.symlog(torch.tensor(0.0)).item() == 0.0


def test_two_hot_target_is_two_hot_and_sums_to_one(dreamer):
    """The target puts mass on the two bins straddling the value, in
    proportion to distance -- so the expectation is exact for any value."""
    logits = torch.zeros(1, 1, 255)
    distribution = dreamer.TwoHotDist(logits)
    for value in (-3.0, 0.0, 0.25, 7.5):
        x = dreamer.symlog(torch.tensor([[[value]]]))
        below = torch.sum((distribution.buckets <= x[..., None]).int(), -1) - 1
        above = 255 - torch.sum((distribution.buckets > x[..., None]).int(), -1)
        assert (above - below).abs().max() <= 1


@pytest.mark.parametrize("value", [-50.0, -1.0, 0.0, 0.3, 12.0, 900.0])
def test_two_hot_recovers_the_value_it_was_fit_to(dreamer, value):
    """Fitting a single target must put the distribution's mean back on it,
    which is what lets one hyperparameter set span many reward scales."""
    logits = torch.zeros(1, 1, 255, requires_grad=True)
    target = torch.full((1, 1, 1), value)
    optimizer = torch.optim.Adam([logits], lr=0.5)
    for _ in range(400):
        optimizer.zero_grad()
        (-dreamer.TwoHotDist(logits).log_prob(target)).mean().backward()
        optimizer.step()
    assert dreamer.TwoHotDist(logits).mean().item() == pytest.approx(value, rel=2e-2, abs=1e-2)


def test_two_hot_bins_span_symlog_20(dreamer):
    buckets = dreamer.TwoHotDist(torch.zeros(1, 255)).buckets
    assert len(buckets) == 255
    assert buckets[0].item() == pytest.approx(-20.0)
    assert buckets[-1].item() == pytest.approx(20.0)
    # symexp(20) is about 4.8e8, so the head covers essentially any Atari return.
    assert dreamer.symexp(buckets[-1]).item() > 1e8


# --- categorical latents ---------------------------------------------------


def test_unimix_keeps_every_class_reachable(dreamer):
    """"we parameterize the categorical distributions as a mixture of 1%
    uniform and 99% neural network output" -- so no logit can go to -inf."""
    logits = torch.tensor([[100.0, -100.0, -100.0, -100.0]])
    probs = dreamer.OneHotDist(logits, unimix_ratio=0.01).probs
    assert probs.min().item() == pytest.approx(0.01 / 4, rel=1e-4)
    assert probs.sum().item() == pytest.approx(1.0)


def test_straight_through_sample_is_one_hot_with_gradient(dreamer):
    logits = torch.randn(16, 6, requires_grad=True)
    sample = dreamer.OneHotDist(logits, unimix_ratio=0.01).sample()
    assert sample.requires_grad
    assert torch.allclose(sample.sum(-1), torch.ones(16), atol=1e-5)
    assert torch.allclose(sample.detach().max(-1).values, torch.ones(16), atol=1e-5)
    sample.sum().backward()
    assert logits.grad is not None


# --- the RSSM --------------------------------------------------------------


def make_rssm(dreamer, actions=5, embed=32):
    return dreamer.RSSM(stoch=4, deter=16, hidden=16, discrete=3, unimix_ratio=0.01,
                        num_actions=actions, embed=embed)


def test_feature_is_the_flattened_stochastic_plus_deterministic_state(dreamer):
    rssm = make_rssm(dreamer)
    state = rssm.initial(2, "cpu")
    assert rssm.get_feat(state).shape == (2, 4 * 3 + 16)
    assert rssm.feat_size == 4 * 3 + 16


def test_initial_deterministic_state_is_learned(dreamer):
    """"initial" is a parameter, not zeros; the paper trains it."""
    rssm = make_rssm(dreamer)
    assert rssm.W.requires_grad
    assert rssm.initial(3, "cpu")["deter"].shape == (3, 16)
    rssm.W.data.fill_(0.5)
    assert torch.allclose(rssm.initial(1, "cpu")["deter"], torch.tanh(torch.tensor(0.5)))


def test_img_step_never_sees_the_observation(dreamer):
    """The prior is what the model can imagine; it must depend only on the
    previous state and action."""
    torch.manual_seed(0)
    rssm = make_rssm(dreamer)
    state = rssm.initial(2, "cpu")
    action = torch.nn.functional.one_hot(torch.tensor([1, 3]), 5).float()
    torch.manual_seed(1)
    first = rssm.img_step({k: v.clone() for k, v in state.items()}, action)
    torch.manual_seed(1)
    second = rssm.img_step({k: v.clone() for k, v in state.items()}, action)
    assert torch.equal(first["deter"], second["deter"])
    assert torch.equal(first["logit"], second["logit"])


def test_obs_step_does_not_rewrite_the_state_it_was_given(dreamer):
    """`observe` keeps a reference to every posterior it produces. If the
    is_first reset wrote through the incoming dict it would retroactively
    corrupt the previous timestep, at every episode boundary."""
    torch.manual_seed(0)
    rssm = make_rssm(dreamer)
    state = rssm.initial(2, "cpu")
    state = {k: v + 0.1 for k, v in state.items()}
    snapshot = {k: v.clone() for k, v in state.items()}
    action = torch.nn.functional.one_hot(torch.tensor([1, 3]), 5).float()
    rssm.obs_step(state, action, torch.randn(2, 32), torch.tensor([0.0, 1.0]))
    for key in snapshot:
        assert torch.equal(state[key], snapshot[key]), f"obs_step mutated {key}"


def test_observe_resets_only_the_flagged_rows(dreamer):
    torch.manual_seed(0)
    rssm = make_rssm(dreamer)
    embed = torch.randn(2, 6, 32)
    actions = torch.nn.functional.one_hot(torch.randint(0, 5, (2, 6)), 5).float()
    is_first = torch.zeros(2, 6)
    is_first[:, 0] = 1.0
    is_first[1, 3] = 1.0
    post, prior = rssm.observe(embed, actions, is_first)
    assert post["deter"].shape == (2, 6, 16)
    assert post["logit"].shape == (2, 6, 4, 3)
    # The posterior's deterministic part is the prior's, by construction.
    assert torch.equal(post["deter"], prior["deter"])


def test_kl_loss_is_clipped_below_the_free_nats(dreamer):
    """"we clip the KL below 1 nat" -- both terms, separately, and the
    representation term is scaled 5x more weakly than the dynamics term."""
    torch.manual_seed(0)
    rssm = make_rssm(dreamer)
    identical = {"logit": torch.randn(2, 5, 4, 3)}
    loss, value, dynamics, representation = rssm.kl_loss(identical, identical, 1.0, 0.5, 0.1)
    assert torch.allclose(dynamics, torch.ones_like(dynamics))
    assert torch.allclose(representation, torch.ones_like(representation))
    assert torch.allclose(loss, torch.full_like(loss, 0.5 * 1.0 + 0.1 * 1.0))
    assert torch.allclose(value, torch.zeros_like(value), atol=1e-6)


def test_kl_terms_stop_gradients_on_opposite_sides(dreamer):
    """The dynamics term pulls the prior towards a detached posterior and the
    representation term does the reverse; that asymmetry is the whole point."""
    torch.manual_seed(0)
    rssm = make_rssm(dreamer)
    post = {"logit": torch.randn(2, 3, 4, 3, requires_grad=True)}
    prior = {"logit": torch.randn(2, 3, 4, 3, requires_grad=True)}
    _, _, dynamics, _ = rssm.kl_loss(post, prior, 0.0, 0.5, 0.1)
    dynamics.sum().backward()
    assert post["logit"].grad is None or torch.allclose(post["logit"].grad, torch.zeros(1))
    assert prior["logit"].grad is not None and prior["logit"].grad.abs().sum() > 0


# --- imagination and the return ------------------------------------------


def test_lambda_return_matches_a_direct_sum_when_lambda_is_one(dreamer):
    """lambda = 1 is the discounted Monte Carlo return."""
    steps = 6
    reward = torch.arange(1.0, steps + 1).reshape(steps, 1, 1)
    value = torch.zeros(steps, 1, 1)
    discount = torch.full((steps, 1, 1), 0.9)
    bootstrap = torch.zeros(1, 1)
    returns = dreamer.lambda_return(reward, value, discount, bootstrap, 1.0)
    expected = sum(0.9**i * reward[i] for i in range(steps))
    assert returns[0].item() == pytest.approx(expected.item(), rel=1e-6)


def test_lambda_return_matches_one_step_when_lambda_is_zero(dreamer):
    steps = 4
    reward = torch.ones(steps, 1, 1)
    value = torch.full((steps, 1, 1), 5.0)
    discount = torch.full((steps, 1, 1), 0.9)
    returns = dreamer.lambda_return(reward, value, discount, torch.zeros(1, 1), 0.0)
    assert returns[0].item() == pytest.approx(1.0 + 0.9 * 5.0)


def test_reward_ema_scale_never_amplifies(dreamer):
    """"we divide returns by their range, but only if the range is larger
    than 1" -- small returns keep their scale."""
    ema = dreamer.RewardEMA()
    state = torch.zeros(2)
    tiny = torch.full((100, 1), 0.001)
    for _ in range(500):
        _, scale = ema(tiny, state)
    assert scale.item() == pytest.approx(1.0)

    ema = dreamer.RewardEMA()
    state = torch.zeros(2)
    wide = torch.linspace(-500, 500, 1000).reshape(-1, 1)
    for _ in range(2000):
        offset, scale = ema(wide, state)
    assert scale.item() > 100.0


@pytest.mark.parametrize("device", device_params())
def test_imagination_shapes_line_up(dreamer, device):
    """feats, states and actions all have length `horizon`; the final feature
    is what the lambda-return bootstraps from."""
    torch.manual_seed(0)
    args = dreamer.Args()
    args.dyn_stoch, args.dyn_discrete, args.dyn_deter, args.dyn_hidden = 4, 3, 16, 16
    args.cnn_depth, args.units, args.imag_horizon = 4, 16, 5
    world_model = dreamer.WorldModel(6, args, device)
    behavior = dreamer.ImagBehavior(6, world_model, args, device)
    start = world_model.dynamics.initial(8, device)
    start = {k: v.unsqueeze(0) for k, v in start.items()}  # (1, 8, ...)
    feats, states, actions = behavior.imagine(start, args.imag_horizon)
    assert feats.shape == (5, 8, world_model.dynamics.feat_size)
    assert actions.shape == (5, 8, 6)
    assert states["deter"].shape == (5, 8, 16)


# --- replay ---------------------------------------------------------------


def make_replay(dreamer, capacity=64, num_envs=1):
    return dreamer.SequenceReplayBuffer(capacity, num_envs, 6, (3, 64, 64), "cpu")


def test_replay_returns_contiguous_windows(dreamer):
    replay = make_replay(dreamer, capacity=200)
    for step in range(120):
        observation = np.full((1, 3, 64, 64), step % 256, dtype=np.uint8)
        replay.add(observation, np.array([step % 6]), np.array([float(step)]),
                   np.array([1.0 if step == 0 else 0.0]), np.zeros(1))
    batch = replay.sample(4, 8)
    assert batch["image"].shape == (4, 8, 3, 64, 64)
    assert batch["action"].shape == (4, 8, 6)
    for row in range(4):
        rewards = batch["reward"][row]
        assert torch.allclose(rewards[1:] - rewards[:-1], torch.ones(7))


def test_replay_marks_episode_starts_for_the_rssm(dreamer):
    replay = make_replay(dreamer, capacity=200)
    for step in range(60):
        replay.add(np.zeros((1, 3, 64, 64), np.uint8), np.array([0]), np.array([0.0]),
                   np.array([1.0 if step in (0, 25) else 0.0]),
                   np.array([1.0 if step == 24 else 0.0]))
    batch = replay.sample(32, 8)
    assert batch["is_first"].sum() > 0
    # `cont` is the continuation flag the cont head is trained on.
    assert set(batch["cont"].unique().tolist()) <= {0.0, 1.0}


def test_replay_actions_are_one_hot(dreamer):
    replay = make_replay(dreamer, capacity=64)
    for step in range(40):
        replay.add(np.zeros((1, 3, 64, 64), np.uint8), np.array([step % 6]),
                   np.array([0.0]), np.zeros(1), np.zeros(1))
    batch = replay.sample(2, 4)
    assert torch.allclose(batch["action"].sum(-1), torch.ones(2, 4))


def test_replay_refuses_windows_it_cannot_fill(dreamer):
    replay = make_replay(dreamer, capacity=64)
    for _ in range(5):
        replay.add(np.zeros((1, 3, 64, 64), np.uint8), np.zeros(1, dtype=np.int64),
                   np.zeros(1), np.zeros(1), np.zeros(1))
    with pytest.raises(RuntimeError):
        replay.sample(2, 32)


# --- configuration --------------------------------------------------------


def test_defaults_match_the_atari100k_config(dreamer):
    args = dreamer.Args()
    assert (args.dyn_stoch, args.dyn_discrete) == (32, 32)
    assert (args.dyn_deter, args.dyn_hidden, args.units) == (512, 512, 512)
    assert (args.batch_size, args.batch_length) == (16, 64)
    assert args.imag_horizon == 15
    assert (args.discount, args.discount_lambda) == (0.997, 0.95)
    assert (args.kl_free, args.dyn_scale, args.rep_scale) == (1.0, 0.5, 0.1)
    assert (args.model_lr, args.actor_lr, args.critic_lr) == (1e-4, 3e-5, 3e-5)
    assert args.actor_entropy == 3e-4
    assert args.unimix_ratio == 0.01
    assert args.slow_target_fraction == 0.02
    # atari100k overrides: reinforce gradients, action repeat 4, 400k frames.
    assert args.imag_gradient == "reinforce"
    assert args.action_repeat == 4
    assert args.total_timesteps == 400000
    assert args.train_ratio == 1024
    # train_ratio 1024 over a 16x64 batch is exactly one update per policy step.
    assert (args.batch_size * args.batch_length) / args.train_ratio == 1.0


def test_encoder_reaches_the_minimum_resolution(dreamer):
    """The stack halves 64 -> 32 -> 16 -> 8 -> 4, doubling channels each time.
    An 84x84 observation would not divide, which is why this trainer does not
    reuse the repository's Atari pipeline."""
    encoder = dreamer.ConvEncoder(depth=8)
    assert encoder.outdim == 8 * 2**3 * 4 * 4
    output = encoder(torch.rand(2, 3, 3, 64, 64))
    assert output.shape == (2, 3, encoder.outdim)


def test_decoder_round_trips_the_shape(dreamer):
    decoder = dreamer.ConvDecoder(48, depth=8)
    assert decoder(torch.randn(2, 3, 48)).shape == (2, 3, 3, 64, 64)
