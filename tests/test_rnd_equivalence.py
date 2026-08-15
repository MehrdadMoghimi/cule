"""PPO + RND: equivalence against openai/random-network-distillation and CleanRL.

The official implementation is TensorFlow 1, so the reference for the *formulas*
is a transcription of `policies/cnn_policy_param_matched.py` with line numbers
quoted. The reference for the *PyTorch scaffolding* is CleanRL's
`ppo_rnd_envpool.py`, which is executable and is loaded straight from the clone
at `~/cleanrl` where present.

The interesting content of this file is the third section: RND has four
mechanisms that are individually easy to implement and individually easy to get
backwards, and all four are checked as behaviour rather than as arithmetic —
non-episodic intrinsic returns, the running-return (not running-reward)
normalisation, predictor subsampling that does not scale the gradient, and a
frozen target.
"""

import os

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from conftest import REPO_ROOT, DiscreteEnvStub, load_trainer

TRAINER = 'ppo_rnd_atari'


@pytest.fixture(scope='module')
def rnd():
    return load_trainer(TRAINER)


# ---------------------------------------------------------------------------
# the intrinsic reward
# ---------------------------------------------------------------------------

def test_intrinsic_reward_is_the_reference_mean(rnd):
    """`tf.reduce_mean(tf.square(stop_gradient(X_r) - X_r_hat), axis=-1)`.

    `cnn_policy_param_matched.py:167`.
    """
    torch.manual_seed(0)
    predict = torch.randn(64, 512, dtype=torch.float64)
    target = torch.randn(64, 512, dtype=torch.float64)

    got = rnd.intrinsic_reward(predict, target, 'mean')
    want = (target - predict).pow(2).mean(-1)
    assert torch.allclose(got, want, rtol=0, atol=1e-14)
    assert got.shape == (64,)


def test_cleanrl_variant_differs_only_by_a_constant(rnd):
    """CleanRL uses `sum(1) / 2`; the reference uses `mean(-1)`.

    The two are a fixed multiple of each other, which is why the disagreement is
    inert once the intrinsic reward is divided by its own running std.
    """
    torch.manual_seed(0)
    feature_dim = 512
    predict = torch.randn(128, feature_dim, dtype=torch.float64)
    target = torch.randn(128, feature_dim, dtype=torch.float64)

    reference = rnd.intrinsic_reward(predict, target, 'mean')
    cleanrl = rnd.intrinsic_reward(predict, target, 'sum_half')
    assert torch.allclose(cleanrl, reference * (feature_dim / 2), rtol=0, atol=1e-12)

    # ...and after whitening by their own standard deviations they coincide.
    assert torch.allclose(reference / reference.std(), cleanrl / cleanrl.std(), rtol=0, atol=1e-11)


def test_unknown_reduction_rejected(rnd):
    with pytest.raises(ValueError, match='unsupported intrinsic_reward_reduction'):
        rnd.intrinsic_reward(torch.zeros(2, 4), torch.zeros(2, 4), 'median')


def test_intrinsic_reward_is_zero_for_a_perfect_predictor(rnd):
    features = torch.randn(8, 512)
    assert torch.allclose(rnd.intrinsic_reward(features, features), torch.zeros(8), atol=1e-12)


# ---------------------------------------------------------------------------
# the predictor loss
# ---------------------------------------------------------------------------

def test_predictor_loss_matches_the_reference_masking(rnd):
    """`reduce_sum(mask * aux_loss) / max(reduce_sum(mask), 1)`.

    `cnn_policy_param_matched.py:172-175`.
    """
    torch.manual_seed(0)
    predict = torch.randn(32, 512, dtype=torch.float64, requires_grad=True)
    target = torch.randn(32, 512, dtype=torch.float64)
    mask = (torch.rand(32) < 0.25).double()

    got = rnd.rnd_predictor_loss(predict, target, mask)
    per_sample = F.mse_loss(predict, target, reduction='none').mean(-1)
    want = (per_sample * mask).sum() / torch.clamp(mask.sum(), min=1.0)
    assert np.isclose(got.item(), want.item(), rtol=0, atol=1e-13)


def test_predictor_loss_survives_an_all_zero_mask(rnd):
    """`max(sum(mask), 1)` is what stops a 0/0 when nothing is sampled."""
    predict = torch.randn(8, 16, dtype=torch.float64, requires_grad=True)
    target = torch.randn(8, 16, dtype=torch.float64)
    loss = rnd.rnd_predictor_loss(predict, target, torch.zeros(8, dtype=torch.float64))
    assert loss.item() == 0.0
    loss.backward()
    assert torch.count_nonzero(predict.grad) == 0


def test_subsampling_does_not_shrink_the_gradient(rnd):
    """Dividing by `sum(mask)` rather than by the batch size is the point.

    Lowering `update_proportion` must slow the predictor down by giving it fewer
    *samples*, not by shrinking each step. With a mask that happens to select
    everything, the loss equals the plain mean.
    """
    torch.manual_seed(0)
    predict = torch.randn(64, 128, dtype=torch.float64, requires_grad=True)
    target = torch.randn(64, 128, dtype=torch.float64)

    full = rnd.rnd_predictor_loss(predict, target, torch.ones(64, dtype=torch.float64))
    plain = F.mse_loss(predict, target, reduction='none').mean(-1).mean()
    assert np.isclose(full.item(), plain.item(), rtol=0, atol=1e-13)

    # A quarter mask keeps the same scale (a mean over the selected quarter),
    # so the gradient magnitude per selected sample is unchanged.
    mask = torch.zeros(64, dtype=torch.float64)
    mask[:16] = 1.0
    quarter = rnd.rnd_predictor_loss(predict, target, mask)
    quarter_direct = F.mse_loss(predict[:16], target[:16], reduction='none').mean(-1).mean()
    assert np.isclose(quarter.item(), quarter_direct.item(), rtol=0, atol=1e-13)


def test_target_receives_no_gradient(rnd):
    predict = torch.randn(8, 16, dtype=torch.float64, requires_grad=True)
    target = torch.randn(8, 16, dtype=torch.float64, requires_grad=True)
    rnd.rnd_predictor_loss(predict, target, torch.ones(8, dtype=torch.float64)).backward()
    assert target.grad is None
    assert predict.grad is not None


# ---------------------------------------------------------------------------
# observation normalisation
# ---------------------------------------------------------------------------

def test_normalize_rnd_obs_whitens_and_clips(rnd):
    frame = torch.tensor([[[[0.0, 10.0, 100.0, -100.0]]]])
    mean = torch.tensor([[[[5.0, 5.0, 5.0, 5.0]]]])
    var = torch.tensor([[[[4.0, 4.0, 4.0, 4.0]]]])
    got = rnd.normalize_rnd_obs(frame, mean, var, 5.0)
    # (0-5)/2 = -2.5 ; (10-5)/2 = 2.5 ; (100-5)/2 = 47.5 -> 5 ; (-100-5)/2 -> -5
    assert torch.allclose(got, torch.tensor([[[[-2.5, 2.5, 5.0, -5.0]]]]), atol=1e-6)


def test_normalize_rnd_obs_divides_by_sqrt_var_with_no_epsilon(rnd):
    """The reference has no `+ 1e-8`; var starts at 1, so it never divides by 0."""
    frame = torch.ones(1, 1, 2, 2)
    got = rnd.normalize_rnd_obs(frame, torch.zeros(1, 1, 2, 2), torch.full((1, 1, 2, 2), 4.0), 5.0)
    assert torch.allclose(got, torch.full((1, 1, 2, 2), 0.5), atol=1e-7)


def test_running_mean_std_matches_numpy_on_the_whole_stream(rnd):
    """Chan's parallel update must agree with a single-pass computation."""
    generator = np.random.default_rng(0)
    batches = [generator.normal(3.0, 2.0, size=(37, 5)) for _ in range(9)]
    running = rnd.RunningMeanStd(shape=(5,))
    for batch in batches:
        running.update(batch)

    stacked = np.concatenate(batches, axis=0)
    # `epsilon=1e-4` of pseudo-count at mean 0, var 1 biases it very slightly.
    assert np.allclose(running.mean, stacked.mean(axis=0), rtol=0, atol=1e-4)
    assert np.allclose(running.var, stacked.var(axis=0), rtol=0, atol=1e-3)


def test_running_mean_std_starts_at_unit_variance(rnd):
    """Before any update the RND input is whitened by var=1, i.e. left alone."""
    running = rnd.RunningMeanStd(shape=(1, 1, 4, 4))
    assert np.all(running.mean == 0.0)
    assert np.all(running.var == 1.0)


# ---------------------------------------------------------------------------
# the reward forward filter
# ---------------------------------------------------------------------------

def test_reward_forward_filter_accumulates_a_discounted_return(rnd):
    """`rewems = rewems * gamma + r` — a *forward* discounted sum, no reset."""
    filt = rnd.RewardForwardFilter(0.99)
    first = filt.update(np.array([1.0, 2.0]))
    assert np.allclose(first, [1.0, 2.0])
    second = filt.update(np.array([1.0, 0.0]))
    assert np.allclose(second, [1.0 * 0.99 + 1.0, 2.0 * 0.99 + 0.0])
    third = filt.update(np.array([0.0, 0.0]))
    assert np.allclose(third, [second[0] * 0.99, second[1] * 0.99])


def test_reward_forward_filter_never_resets_on_episode_end(rnd):
    """It has no `done` argument at all — the intrinsic stream is non-episodic.

    Pinned as an interface property, because "add a done mask here" is the most
    natural-looking wrong change to make to this class.
    """
    import inspect
    signature = inspect.signature(rnd.RewardForwardFilter.update)
    assert list(signature.parameters) == ['self', 'rews']


def test_normalisation_uses_the_return_not_the_reward(rnd):
    """Dividing by std(reward) instead of std(discounted return) is a silent bug.

    For a constant bonus stream the discounted return grows towards
    `r / (1 - gamma)` and its spread across environments differs from the
    reward's, so the two normalisers are not interchangeable.
    """
    gamma = 0.99
    rewards = np.stack([np.full(50, 1.0), np.full(50, 3.0)], axis=1)  # [T, 2 envs]
    filt = rnd.RewardForwardFilter(gamma)
    returns = np.array([filt.update(step) for step in rewards])
    assert returns.std() > rewards.std() * 5


# ---------------------------------------------------------------------------
# the two advantage streams
# ---------------------------------------------------------------------------

def make_gae_batch(seed=0, num_steps=16, num_envs=4):
    generator = torch.Generator().manual_seed(seed)
    return dict(
        rewards=torch.randn(num_steps, num_envs, generator=generator, dtype=torch.float64),
        curiosity_rewards=torch.rand(num_steps, num_envs, generator=generator, dtype=torch.float64),
        ext_values=torch.randn(num_steps, num_envs, generator=generator, dtype=torch.float64),
        int_values=torch.randn(num_steps, num_envs, generator=generator, dtype=torch.float64),
        dones=(torch.rand(num_steps, num_envs, generator=generator) < 0.2).double(),
        next_done=(torch.rand(num_envs, generator=generator) < 0.2).double(),
        next_value_ext=torch.randn(num_envs, generator=generator, dtype=torch.float64),
        next_value_int=torch.randn(num_envs, generator=generator, dtype=torch.float64),
    )


@pytest.mark.parametrize('seed', range(4))
def test_dual_gae_matches_cleanrl_reference(rnd, seed):
    """Transcription of `ppo_rnd_envpool.py`'s advantage loop."""
    batch = make_gae_batch(seed)
    gamma, int_gamma, lam = 0.999, 0.99, 0.95

    got_ext, got_int = rnd.dual_gae(gamma=gamma, int_gamma=int_gamma, gae_lambda=lam, **batch)

    num_steps = batch['rewards'].shape[0]
    ext_advantages = torch.zeros_like(batch['rewards'])
    int_advantages = torch.zeros_like(batch['curiosity_rewards'])
    ext_lastgaelam = 0
    int_lastgaelam = 0
    for t in reversed(range(num_steps)):
        if t == num_steps - 1:
            ext_nextnonterminal = 1.0 - batch['next_done']
            int_nextnonterminal = 1.0
            ext_nextvalues = batch['next_value_ext']
            int_nextvalues = batch['next_value_int']
        else:
            ext_nextnonterminal = 1.0 - batch['dones'][t + 1]
            int_nextnonterminal = 1.0
            ext_nextvalues = batch['ext_values'][t + 1]
            int_nextvalues = batch['int_values'][t + 1]
        ext_delta = batch['rewards'][t] + gamma * ext_nextvalues * ext_nextnonterminal - batch['ext_values'][t]
        int_delta = (batch['curiosity_rewards'][t] + int_gamma * int_nextvalues * int_nextnonterminal
                     - batch['int_values'][t])
        ext_advantages[t] = ext_lastgaelam = ext_delta + gamma * lam * ext_nextnonterminal * ext_lastgaelam
        int_advantages[t] = int_lastgaelam = int_delta + int_gamma * lam * int_nextnonterminal * int_lastgaelam

    assert torch.allclose(got_ext, ext_advantages, rtol=0, atol=1e-13)
    assert torch.allclose(got_int, int_advantages, rtol=0, atol=1e-13)


def test_intrinsic_stream_ignores_dones_entirely(rnd):
    """The defining asymmetry: flipping every `done` must not move `A_int`."""
    batch = make_gae_batch(1)
    gamma, int_gamma, lam = 0.999, 0.99, 0.95

    ext_a, int_a = rnd.dual_gae(gamma=gamma, int_gamma=int_gamma, gae_lambda=lam, **batch)

    flipped = dict(batch)
    flipped['dones'] = 1.0 - batch['dones']
    flipped['next_done'] = 1.0 - batch['next_done']
    ext_b, int_b = rnd.dual_gae(gamma=gamma, int_gamma=int_gamma, gae_lambda=lam, **flipped)

    assert torch.allclose(int_a, int_b, rtol=0, atol=1e-14)
    assert not torch.allclose(ext_a, ext_b, rtol=1e-3, atol=1e-3)


def test_the_two_streams_use_different_discounts(rnd):
    """gamma_ext = 0.999 and gamma_int = 0.99 are not interchangeable."""
    num_steps, num_envs = 30, 1
    ones = torch.ones(num_steps, num_envs, dtype=torch.float64)
    zeros = torch.zeros(num_steps, num_envs, dtype=torch.float64)
    ext_a, int_a = rnd.dual_gae(
        rewards=ones, curiosity_rewards=ones, ext_values=zeros, int_values=zeros,
        dones=zeros, next_done=torch.zeros(num_envs, dtype=torch.float64),
        next_value_ext=torch.zeros(num_envs, dtype=torch.float64),
        next_value_int=torch.zeros(num_envs, dtype=torch.float64),
        gamma=0.999, int_gamma=0.99, gae_lambda=1.0)
    # Identical rewards, identical values, no dones -> the only difference left
    # is the discount, so the two advantages must diverge.
    assert ext_a[0].item() > int_a[0].item()


# ---------------------------------------------------------------------------
# the networks
# ---------------------------------------------------------------------------

def test_rnd_target_is_frozen(rnd):
    model = rnd.RNDModel()
    assert all(not p.requires_grad for p in model.target.parameters())
    assert all(p.requires_grad for p in model.predictor.parameters())


def test_predictor_is_strictly_deeper_than_the_target(rnd):
    """The reference gives the predictor two extra hidden layers on purpose."""
    model = rnd.RNDModel()
    predictor_linear = [m for m in model.predictor if isinstance(m, torch.nn.Linear)]
    target_linear = [m for m in model.target if isinstance(m, torch.nn.Linear)]
    assert len(predictor_linear) == 3
    assert len(target_linear) == 1
    assert predictor_linear[-1].out_features == target_linear[-1].out_features


def test_rnd_networks_take_a_single_frame(rnd):
    """One 84x84 channel, not the 4-frame stack the policy sees."""
    model = rnd.RNDModel()
    for network in (model.predictor, model.target):
        first_conv = next(m for m in network if isinstance(m, torch.nn.Conv2d))
        assert first_conv.in_channels == 1

    frame = torch.randn(5, 1, 84, 84)
    predict, target = model(frame)
    assert predict.shape == (5, 512) and target.shape == (5, 512)


def test_rnd_target_stays_frozen_through_an_optimizer_step(rnd):
    """The end-to-end version: only the predictor may move."""
    torch.manual_seed(0)
    model = rnd.RNDModel(feature_dim=32)
    before = {name: p.detach().clone() for name, p in model.target.named_parameters()}
    optimizer = torch.optim.Adam(model.predictor.parameters(), lr=1e-2)

    frame = torch.randn(8, 1, 84, 84)
    predict, target = model(frame)
    loss = rnd.rnd_predictor_loss(predict, target, torch.ones(8))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    for name, parameter in model.target.named_parameters():
        assert torch.equal(parameter, before[name]), name
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.predictor.parameters())


def test_predictor_error_falls_on_a_repeated_observation(rnd):
    """The mechanism, end to end: novelty must decay with exposure."""
    torch.manual_seed(0)
    model = rnd.RNDModel(feature_dim=64)
    optimizer = torch.optim.Adam(model.predictor.parameters(), lr=1e-3)
    frame = torch.randn(4, 1, 84, 84)

    with torch.no_grad():
        predict, target = model(frame)
        before = rnd.intrinsic_reward(predict, target).mean().item()
    for _ in range(30):
        predict, target = model(frame)
        loss = rnd.rnd_predictor_loss(predict, target, torch.ones(4))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        predict, target = model(frame)
        after = rnd.intrinsic_reward(predict, target).mean().item()

    assert after < before * 0.5

    # ...and an unseen observation is still novel.
    with torch.no_grad():
        predict, target = model(torch.randn(4, 1, 84, 84))
        unseen = rnd.intrinsic_reward(predict, target).mean().item()
    assert unseen > after


def test_agent_has_two_value_heads_over_a_shared_trunk(rnd):
    torch.manual_seed(0)
    agent = rnd.Agent(DiscreteEnvStub(18))
    obs = torch.randint(0, 255, (6, 4, 84, 84), dtype=torch.uint8).float()
    action, logprob, entropy, value_ext, value_int = agent.get_action_and_value(obs)
    assert action.shape == (6,) and logprob.shape == (6,) and entropy.shape == (6,)
    assert value_ext.shape == (6, 1) and value_int.shape == (6, 1)
    assert not torch.allclose(value_ext, value_int)

    got_ext, got_int = agent.get_value(obs)
    assert torch.allclose(got_ext, value_ext) and torch.allclose(got_int, value_int)


def test_agent_trunk_is_448_wide_with_the_residual_extra_layer(rnd):
    """`param matched`: 3136 -> 256 -> 448, plus `critic(features + hidden)`."""
    agent = rnd.Agent(DiscreteEnvStub(18))
    linear = [m for m in agent.network if isinstance(m, torch.nn.Linear)]
    assert [(m.in_features, m.out_features) for m in linear] == [(3136, 256), (256, 448)]
    assert agent.critic_ext.in_features == 448
    assert agent.critic_int.in_features == 448
    assert isinstance(agent.extra_layer[0], torch.nn.Linear)
    assert agent.extra_layer[0].in_features == agent.extra_layer[0].out_features == 448


# ---------------------------------------------------------------------------
# against CleanRL's executable reference, where it is present
# ---------------------------------------------------------------------------

def test_agent_matches_cleanrl_ppo_rnd_agent():
    """Same architecture and same initialisation, layer for layer."""
    path = os.path.join(os.path.expanduser('~'), 'cleanrl', 'cleanrl', 'ppo_rnd_envpool.py')
    if not os.path.exists(path):
        pytest.skip('cleanrl checkout not present at ~/cleanrl')

    import ast
    import textwrap
    with open(path) as handle:
        source = handle.read()
    tree = ast.parse(source)

    namespace = {'nn': torch.nn, 'torch': torch, 'np': np}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == 'layer_init':
            exec(textwrap.dedent(ast.get_source_segment(source, node)), namespace)
        if isinstance(node, ast.ClassDef) and node.name in ('Agent', 'RNDModel'):
            namespace['Categorical'] = __import__(
                'torch.distributions.categorical', fromlist=['Categorical']).Categorical
            exec(textwrap.dedent(ast.get_source_segment(source, node)), namespace)

    module = load_trainer(TRAINER)
    envs = DiscreteEnvStub(18)

    torch.manual_seed(0)
    ours = module.Agent(envs)
    torch.manual_seed(0)
    theirs = namespace['Agent'](envs)
    our_state, their_state = ours.state_dict(), theirs.state_dict()
    assert our_state.keys() == their_state.keys()
    for key in our_state:
        assert torch.equal(our_state[key], their_state[key]), key

    torch.manual_seed(0)
    our_rnd = module.RNDModel(512)
    torch.manual_seed(0)
    their_rnd = namespace['RNDModel'](1, 512)
    our_state, their_state = our_rnd.state_dict(), their_rnd.state_dict()
    assert our_state.keys() == their_state.keys()
    for key in our_state:
        assert torch.equal(our_state[key], their_state[key]), key
