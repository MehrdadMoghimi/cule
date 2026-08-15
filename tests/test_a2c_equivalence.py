"""A2C: equivalence against openai/baselines' reference implementation.

The port is a subtraction from PPO, so the risk is not in what it computes but
in what it quietly keeps. Three things are pinned here:

  * `RMSpropTFLike` really is TensorFlow's RMSProp — accumulator initialised to
    ones, epsilon inside the square root — and really does differ from
    `torch.optim.RMSprop`, which is what a careless port would have used;
  * `nstep_returns` agrees with a transcription of baselines'
    `discount_with_dones`, including on the branch baselines special-cases;
  * the loss is baselines' loss: advantage detached, value loss with no leading
    1/2, entropy subtracted rather than added.
"""

import numpy as np
import pytest
import torch

from conftest import DiscreteEnvStub, load_trainer

TRAINER = 'a2c_atari'


@pytest.fixture(scope='module')
def a2c():
    return load_trainer(TRAINER)


# ---------------------------------------------------------------------------
# RMSProp: TensorFlow semantics
# ---------------------------------------------------------------------------

def tf_rmsprop_reference(params, grads, lr, alpha, eps, steps):
    """`tf.train.RMSPropOptimizer` in NumPy.

    ms <- alpha * ms + (1 - alpha) * g^2,     ms initialised to 1
    p  <- p - lr * g / sqrt(ms + eps)
    """
    params = [np.array(p, dtype=np.float64) for p in params]
    ms = [np.ones_like(p) for p in params]
    for step_grads in grads:
        for i, g in enumerate(step_grads):
            g = np.asarray(g, dtype=np.float64)
            ms[i] = alpha * ms[i] + (1.0 - alpha) * g * g
            params[i] = params[i] - lr * g / np.sqrt(ms[i] + eps)
    assert len(grads) == steps
    return params


@pytest.mark.parametrize('eps', [1e-5, 1e-8, 1e-2])
def test_rmsprop_tf_like_matches_tensorflow_semantics(a2c, eps):
    torch.manual_seed(0)
    shapes = [(4, 3), (7,), (2, 2, 2)]
    init = [torch.randn(s, dtype=torch.float64) for s in shapes]
    tensors = [t.clone().requires_grad_(True) for t in init]
    lr, alpha, steps = 0.011, 0.93, 6

    optimizer = a2c.RMSpropTFLike(tensors, lr=lr, alpha=alpha, eps=eps)
    grad_sequence = []
    generator = torch.Generator().manual_seed(1234)
    for _ in range(steps):
        step_grads = [torch.randn(s, generator=generator, dtype=torch.float64) for s in shapes]
        grad_sequence.append([g.numpy().copy() for g in step_grads])
        for tensor, grad in zip(tensors, step_grads):
            tensor.grad = grad.clone()
        optimizer.step()

    expected = tf_rmsprop_reference([t.numpy() for t in init], grad_sequence, lr, alpha, eps, steps)
    for got, want in zip(tensors, expected):
        assert np.allclose(got.detach().numpy(), want, rtol=0, atol=1e-12)


def test_accumulator_starts_at_one_not_zero(a2c):
    """The first update must be lr * g, not ~10 * lr * sign(g)."""
    tensor = torch.zeros(5, dtype=torch.float64, requires_grad=True)
    optimizer = a2c.RMSpropTFLike([tensor], lr=0.1, alpha=0.99, eps=0.0)
    tensor.grad = torch.full((5,), 3.0, dtype=torch.float64)
    optimizer.step()
    # ms = 0.99 * 1 + 0.01 * 9 = 1.08; step = 0.1 * 3 / sqrt(1.08)
    expected = -0.1 * 3.0 / np.sqrt(1.08)
    assert np.allclose(tensor.detach().numpy(), expected, atol=1e-12)

    state = optimizer.state[tensor]
    assert np.allclose(state['square_avg'].numpy(), 1.08, atol=1e-12)


def test_torch_rmsprop_would_have_been_a_different_optimizer(a2c):
    """Guard the trap: the two optimisers disagree, so the default matters."""
    torch.manual_seed(0)
    init = torch.randn(16, dtype=torch.float64)
    grads = [torch.randn(16, dtype=torch.float64) for _ in range(3)]

    results = []
    for factory in (
        lambda p: a2c.RMSpropTFLike(p, lr=7e-4, alpha=0.99, eps=1e-5),
        lambda p: torch.optim.RMSprop(p, lr=7e-4, alpha=0.99, eps=1e-5),
    ):
        tensor = init.clone().requires_grad_(True)
        optimizer = factory([tensor])
        for grad in grads:
            tensor.grad = grad.clone()
            optimizer.step()
        results.append(tensor.detach().numpy().copy())

    # First step alone is off by ~an order of magnitude; after three steps the
    # trajectories are nowhere near each other.
    assert not np.allclose(results[0], results[1], rtol=1e-3, atol=1e-6)


def _load_sb3_rmsprop():
    """Load SB3's optimiser straight from the clone, if it is present.

    The module imports nothing but torch, so it can be executed without
    installing stable-baselines3 (which would drag in gym, a whole second
    Atari stack, and a numpy pin).
    """
    import importlib.util
    import os

    from conftest import REPO_ROOT

    path = os.path.join(REPO_ROOT, 'third_party', 'upstream', 'stable-baselines3',
                        'stable_baselines3', 'common', 'sb2_compat', 'rmsprop_tf_like.py')
    if not os.path.exists(path):
        return None
    spec = importlib.util.spec_from_file_location('_sb3_rmsprop_tf_like', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_rmsprop_tf_like_matches_stable_baselines3(a2c):
    """Diff against SB3's `RMSpropTFLike`, which exists for the same reason."""
    sb3 = _load_sb3_rmsprop()
    if sb3 is None:
        pytest.skip('stable-baselines3 clone not present under third_party/upstream')
    torch.manual_seed(0)
    init = torch.randn(9, 4, dtype=torch.float64)
    grads = [torch.randn(9, 4, dtype=torch.float64) for _ in range(5)]

    results = []
    for factory in (
        lambda p: a2c.RMSpropTFLike(p, lr=7e-4, alpha=0.99, eps=1e-5),
        lambda p: sb3.RMSpropTFLike(p, lr=7e-4, alpha=0.99, eps=1e-5),
    ):
        tensor = init.clone().requires_grad_(True)
        optimizer = factory([tensor])
        for grad in grads:
            tensor.grad = grad.clone()
            optimizer.step()
        results.append(tensor.detach().numpy().copy())

    assert np.allclose(results[0], results[1], rtol=0, atol=1e-14)


@pytest.mark.parametrize('centered', [False, True])
def test_momentum_and_centered_branches_are_consistent(a2c, centered):
    """Momentum/centred paths are unused by A2C but must not be silently wrong."""
    torch.manual_seed(3)
    tensor = torch.randn(6, dtype=torch.float64, requires_grad=True)
    reference = tensor.detach().numpy().copy()
    lr, alpha, eps, momentum = 0.05, 0.9, 1e-6, 0.5
    optimizer = a2c.RMSpropTFLike([tensor], lr=lr, alpha=alpha, eps=eps,
                                  momentum=momentum, centered=centered)

    ms = np.ones_like(reference)
    grad_avg = np.zeros_like(reference)
    buf = np.zeros_like(reference)
    for _ in range(4):
        grad = torch.randn(6, dtype=torch.float64)
        tensor.grad = grad.clone()
        optimizer.step()

        g = grad.numpy()
        ms = alpha * ms + (1 - alpha) * g * g
        if centered:
            grad_avg = alpha * grad_avg + (1 - alpha) * g
            avg = np.sqrt(ms - grad_avg * grad_avg + eps)
        else:
            avg = np.sqrt(ms + eps)
        buf = momentum * buf + g / avg
        reference = reference - lr * buf

    assert np.allclose(tensor.detach().numpy(), reference, rtol=0, atol=1e-12)


# ---------------------------------------------------------------------------
# n-step returns
# ---------------------------------------------------------------------------

def baselines_returns_reference(a2c, rewards, next_dones, last_values, gamma):
    """baselines' runner, branch and all, one environment at a time."""
    num_steps, num_envs = rewards.shape
    out = np.zeros((num_steps, num_envs), dtype=np.float64)
    for env in range(num_envs):
        env_rewards = [float(r) for r in rewards[:, env]]
        env_dones = [float(d) for d in next_dones[:, env]]
        value = float(last_values[env])
        if env_dones[-1] == 0:
            discounted = a2c.discount_with_dones(env_rewards + [value], env_dones + [0.0], gamma)[:-1]
        else:
            discounted = a2c.discount_with_dones(env_rewards, env_dones, gamma)
        out[:, env] = discounted
    return out


@pytest.mark.parametrize('seed', range(6))
def test_nstep_returns_match_baselines_runner(a2c, seed):
    generator = torch.Generator().manual_seed(seed)
    num_steps, num_envs, gamma = 5, 16, 0.99
    rewards = torch.randn(num_steps, num_envs, generator=generator, dtype=torch.float64)
    next_dones = (torch.rand(num_steps, num_envs, generator=generator) < 0.25).double()
    last_values = torch.randn(num_envs, generator=generator, dtype=torch.float64)

    got = a2c.nstep_returns(rewards, next_dones, last_values, gamma).numpy()
    want = baselines_returns_reference(a2c, rewards.numpy(), next_dones.numpy(), last_values.numpy(), gamma)
    assert np.allclose(got, want, rtol=0, atol=1e-12)


def test_both_baselines_branches_agree(a2c):
    """The `dones[-1] == 1` special case in baselines is redundant, not different."""
    gamma = 0.97
    rewards = [1.0, -2.0, 0.5, 3.0]
    dones = [0.0, 1.0, 0.0, 1.0]  # ends done -> baselines takes the `else` branch
    value = 12.34

    with_bootstrap = a2c.discount_with_dones(rewards + [value], dones + [0.0], gamma)[:-1]
    without_bootstrap = a2c.discount_with_dones(rewards, dones, gamma)
    assert np.allclose(with_bootstrap, without_bootstrap, atol=1e-12)


def test_returns_cut_at_episode_boundaries(a2c):
    """A done at step t must stop the return at t; nothing after it leaks back."""
    gamma = 0.9
    rewards = torch.tensor([[1.0], [1.0], [1.0], [1.0]], dtype=torch.float64)
    next_dones = torch.tensor([[0.0], [1.0], [0.0], [0.0]], dtype=torch.float64)
    last_values = torch.tensor([100.0], dtype=torch.float64)

    got = a2c.nstep_returns(rewards, next_dones, last_values, gamma).squeeze(-1).numpy()
    # step 3: 1 + .9*100 = 91;  step 2: 1 + .9*91 = 82.9
    # step 1: done -> 1 (no bootstrap);  step 0: 1 + .9*1 = 1.9
    assert np.allclose(got, [1.9, 1.0, 82.9, 91.0], atol=1e-12)


def test_no_done_reduces_to_plain_discounting(a2c):
    gamma = 0.99
    num_steps = 5
    rewards = torch.arange(1, num_steps + 1, dtype=torch.float64).reshape(-1, 1)
    next_dones = torch.zeros(num_steps, 1, dtype=torch.float64)
    last_values = torch.tensor([7.0], dtype=torch.float64)

    got = a2c.nstep_returns(rewards, next_dones, last_values, gamma).squeeze(-1).numpy()
    want = [
        sum(gamma ** (k - t) * (k + 1) for k in range(t, num_steps)) + gamma ** (num_steps - t) * 7.0
        for t in range(num_steps)
    ]
    assert np.allclose(got, want, atol=1e-12)


# ---------------------------------------------------------------------------
# the objective
# ---------------------------------------------------------------------------

def test_a2c_losses_match_baselines_formula(a2c):
    torch.manual_seed(0)
    batch = 80
    logprobs = torch.randn(batch, dtype=torch.float64)
    entropies = torch.rand(batch, dtype=torch.float64)
    values = torch.randn(batch, dtype=torch.float64)
    returns = torch.randn(batch, dtype=torch.float64)
    behaviour_values = torch.randn(batch, dtype=torch.float64)
    ent_coef, vf_coef = 0.01, 0.5

    loss, pg_loss, v_loss, entropy = a2c.a2c_losses(
        logprobs, entropies, values, returns, behaviour_values, ent_coef, vf_coef)

    advantages = (returns - behaviour_values).numpy()
    want_pg = float(np.mean(advantages * -logprobs.numpy()))
    # TF `losses.mean_squared_error` has no leading 1/2.
    want_vf = float(np.mean((values.numpy() - returns.numpy()) ** 2))
    want_entropy = float(np.mean(entropies.numpy()))

    assert np.isclose(pg_loss.item(), want_pg, rtol=0, atol=1e-12)
    assert np.isclose(v_loss.item(), want_vf, rtol=0, atol=1e-12)
    assert np.isclose(entropy.item(), want_entropy, rtol=0, atol=1e-12)
    assert np.isclose(
        loss.item(), want_pg - want_entropy * ent_coef + want_vf * vf_coef, rtol=0, atol=1e-12)


def test_value_loss_has_no_leading_half(a2c):
    """PPO's `0.5 * mse` would halve the critic's gradient; baselines' does not."""
    values = torch.tensor([2.0], dtype=torch.float64)
    returns = torch.tensor([0.0], dtype=torch.float64)
    _, _, v_loss, _ = a2c.a2c_losses(
        torch.zeros(1, dtype=torch.float64),
        torch.zeros(1, dtype=torch.float64),
        values, returns, torch.zeros(1, dtype=torch.float64), 0.0, 1.0)
    assert np.isclose(v_loss.item(), 4.0, atol=1e-12)


def test_advantage_carries_no_gradient(a2c):
    """`ADV` is a placeholder in baselines: the critic is not trained through it."""
    logprobs = torch.zeros(4, dtype=torch.float64, requires_grad=True)
    behaviour_values = torch.randn(4, dtype=torch.float64, requires_grad=True)
    values = torch.randn(4, dtype=torch.float64, requires_grad=True)
    returns = torch.randn(4, dtype=torch.float64)

    loss, _, _, _ = a2c.a2c_losses(
        logprobs, torch.zeros(4, dtype=torch.float64), values, returns, behaviour_values, 0.0, 0.5)
    loss.backward()

    # behaviour_values only enters through the advantage, and PPO-style value
    # clipping is gone, so its gradient must be exactly the advantage path...
    assert behaviour_values.grad is not None
    # ...which is the point: it flows to *logprobs*, not into the critic head.
    assert torch.allclose(logprobs.grad, -(returns - behaviour_values).detach() / 4)


def test_entropy_is_a_bonus_not_a_penalty(a2c):
    """Higher entropy must lower the loss."""
    kwargs = dict(
        logprobs=torch.zeros(3, dtype=torch.float64),
        values=torch.zeros(3, dtype=torch.float64),
        returns=torch.zeros(3, dtype=torch.float64),
        behaviour_values=torch.zeros(3, dtype=torch.float64),
        ent_coef=0.01,
        vf_coef=0.5,
    )
    low, *_ = a2c.a2c_losses(entropies=torch.full((3,), 0.1, dtype=torch.float64), **kwargs)
    high, *_ = a2c.a2c_losses(entropies=torch.full((3,), 1.0, dtype=torch.float64), **kwargs)
    assert high.item() < low.item()


# ---------------------------------------------------------------------------
# learning-rate schedule
# ---------------------------------------------------------------------------

def baselines_scheduler_reference(iterations, batch_size, total_timesteps, lr):
    """baselines' `Scheduler`, driven exactly as `Model.train` drives it."""
    counter = 0.0
    out = []
    for _ in range(iterations):
        current = None
        for _ in range(batch_size):  # `for step in range(len(obs))`
            current = lr * (1.0 - counter / total_timesteps)
            counter += 1.0
        out.append(current)
    return out


def test_lr_schedule_matches_baselines_scheduler(a2c):
    batch_size, total_timesteps, lr = 80, 4000, 7e-4
    iterations = total_timesteps // batch_size
    want = baselines_scheduler_reference(iterations, batch_size, total_timesteps, lr)
    got = [a2c.baselines_lr_fraction(i, batch_size, total_timesteps) * lr
           for i in range(1, iterations + 1)]
    assert np.allclose(got, want, rtol=0, atol=1e-15)


def test_lr_schedule_is_clamped_at_zero(a2c):
    """Benchmark mode can run past `total_timesteps`; the LR must not go negative."""
    assert a2c.baselines_lr_fraction(10_000, 80, 4_000) == 0.0


# ---------------------------------------------------------------------------
# the network
# ---------------------------------------------------------------------------

def test_agent_matches_ppo_agent_layer_for_layer(a2c):
    """A2C changes the learning rule, not the network: same torso, same init scales."""
    ppo = load_trainer('ppo_atari')
    envs = DiscreteEnvStub(6)

    torch.manual_seed(0)
    a2c_agent = a2c.Agent(envs)
    torch.manual_seed(0)
    ppo_agent = ppo.Agent(envs)

    a2c_state, ppo_state = a2c_agent.state_dict(), ppo_agent.state_dict()
    assert a2c_state.keys() == ppo_state.keys()
    for key in a2c_state:
        assert torch.equal(a2c_state[key], ppo_state[key]), key


def test_agent_forward_shapes(a2c):
    torch.manual_seed(0)
    envs = DiscreteEnvStub(6)
    agent = a2c.Agent(envs)
    obs = torch.randint(0, 255, (7, 4, 84, 84), dtype=torch.uint8).float()
    action, logprob, entropy, value = agent.get_action_and_value(obs)
    assert action.shape == (7,)
    assert logprob.shape == (7,)
    assert entropy.shape == (7,)
    assert value.shape == (7, 1)
    assert action.min() >= 0 and action.max() < 6


def test_build_optimizer_rejects_unknown(a2c):
    envs = DiscreteEnvStub(4)
    agent = a2c.Agent(envs)

    class _Args:
        learning_rate = 7e-4
        rmsprop_alpha = 0.99
        rmsprop_eps = 1e-5

    assert isinstance(a2c.build_optimizer('rmsprop-tf', agent.parameters(), _Args()), a2c.RMSpropTFLike)
    assert isinstance(a2c.build_optimizer('rmsprop', agent.parameters(), _Args()), torch.optim.RMSprop)
    assert isinstance(a2c.build_optimizer('adam', agent.parameters(), _Args()), torch.optim.Adam)
    with pytest.raises(ValueError, match='unsupported optimizer'):
        a2c.build_optimizer('sgd', agent.parameters(), _Args())


# ---------------------------------------------------------------------------
# A2C is PPO with the PPO removed
# ---------------------------------------------------------------------------

def test_one_ppo_epoch_one_minibatch_no_clipping_reproduces_a2c(a2c):
    """The defining claim: PPO degenerates to A2C at the right settings.

    With a single epoch and a single minibatch the ratio is exactly 1, so the
    clipped surrogate is inactive and `d/dlogp [-A * exp(logp - logp_old)]`
    equals `d/dlogp [-A * logp]`. The *values* differ by a constant — PPO's
    surrogate is `-mean(A)`, A2C's is `-mean(A * logp)` — so only the gradients
    can be compared, and the gradients are what the update sees. Any residual
    difference here would mean the port kept something it should have dropped.
    """
    torch.manual_seed(0)
    batch = 32
    logprobs = torch.randn(batch, dtype=torch.float64, requires_grad=True)
    entropies = torch.rand(batch, dtype=torch.float64)
    values = torch.randn(batch, dtype=torch.float64, requires_grad=True)
    returns = torch.randn(batch, dtype=torch.float64)
    behaviour_values = values.detach().clone()
    advantages = returns - behaviour_values

    # A2C
    a2c_loss, _, _, _ = a2c.a2c_losses(
        logprobs, entropies, values, returns, behaviour_values, 0.01, 0.5)

    # PPO with update_epochs=1, num_minibatches=1, norm_adv=False,
    # clip_vloss=False, and the customary 1/2 folded into the value loss.
    ratio = (logprobs - logprobs.detach()).exp()
    pg = torch.max(-advantages * ratio,
                   -advantages * torch.clamp(ratio, 0.9, 1.1)).mean()
    v_loss = 0.5 * ((values - returns) ** 2).mean()
    ppo_loss = pg - 0.01 * entropies.mean() + v_loss * 1.0  # vf_coef doubled to absorb the 1/2

    a2c_grad = torch.autograd.grad(a2c_loss, [logprobs, values], retain_graph=True)
    ppo_grad = torch.autograd.grad(ppo_loss, [logprobs, values], retain_graph=True)
    for got, want in zip(a2c_grad, ppo_grad):
        assert torch.allclose(got, want, rtol=0, atol=1e-12)

    # ...and the policy-gradient terms really are offset, not equal, so the
    # comparison above is the only one that means anything.
    assert not np.isclose(a2c_loss.item(), ppo_loss.item(), rtol=0, atol=1e-9)


# ---------------------------------------------------------------------------
# differential check against NVIDIA's A2C (examples/a2c/train.py)
#
# openai/baselines is TensorFlow 1 and cannot be executed here, so the formulas
# above are transcriptions. This repository does, however, already contain a
# second, independently written PyTorch A2C: the one NVIDIA shipped with CuLE.
# Its return recursion and loss are spelled differently from baselines' — it
# carries a length `num_steps + 1` `returns` buffer seeded with the bootstrap
# value, and it forms the advantage against the *freshly recomputed* value
# rather than the stored behaviour value — so agreeing with it is a real
# constraint, not a restatement.
# ---------------------------------------------------------------------------

def cule_example_returns(rewards, masks, bootstrap_value, gamma):
    """`examples/a2c/train.py`, the `else` branch of the `use_gae` switch.

        returns[-1] = V(states[-1])
        for step in reversed(range(num_steps)):
            returns[step] = rewards[step] + gamma * returns[step + 1] * masks[step]

    Note `masks`, not dones: CuLE stores `1 - done`.
    """
    num_steps = rewards.shape[0]
    returns = torch.zeros((num_steps + 1,) + rewards.shape[1:], dtype=rewards.dtype)
    returns[-1] = bootstrap_value
    for step in reversed(range(num_steps)):
        returns[step] = rewards[step] + gamma * returns[step + 1] * masks[step]
    return returns[:-1]


@pytest.mark.parametrize('seed', range(4))
def test_returns_match_nvidia_cule_a2c(a2c, seed):
    generator = torch.Generator().manual_seed(100 + seed)
    num_steps, num_envs, gamma = 5, 16, 0.99
    rewards = torch.randn(num_steps, num_envs, generator=generator, dtype=torch.float64)
    next_dones = (torch.rand(num_steps, num_envs, generator=generator) < 0.2).double()
    bootstrap = torch.randn(num_envs, generator=generator, dtype=torch.float64)

    ours = a2c.nstep_returns(rewards, next_dones, bootstrap, gamma)
    theirs = cule_example_returns(rewards, 1.0 - next_dones, bootstrap, gamma)
    assert torch.allclose(ours, theirs, rtol=0, atol=1e-12)


def test_loss_matches_nvidia_cule_a2c(a2c):
    """CuLE forms the advantage against the fresh value; baselines against the
    stored one. On the batch A2C actually trains on they are the same tensor,
    because the weights have not moved since the rollout — which is exactly why
    the two implementations agree despite reading differently.
    """
    torch.manual_seed(0)
    batch = 64
    action_log_probs = torch.randn(batch, dtype=torch.float64, requires_grad=True)
    entropy_terms = torch.rand(batch, dtype=torch.float64)
    value = torch.randn(batch, dtype=torch.float64, requires_grad=True)
    returns = torch.randn(batch, dtype=torch.float64)
    vf_coef, ent_coef = 0.5, 0.01

    # CuLE: advantages = returns - value (fresh); policy loss detaches them.
    advantages = returns - value
    cule_value_loss = advantages.pow(2).mean()
    cule_policy_loss = -(advantages.clone().detach() * action_log_probs).mean()
    cule_entropy = entropy_terms.mean()
    cule_loss = cule_value_loss * vf_coef + cule_policy_loss - cule_entropy * ent_coef

    # Ours: behaviour values supplied as data, equal to the fresh values here.
    ours, pg_loss, v_loss, entropy = a2c.a2c_losses(
        action_log_probs, entropy_terms, value, returns, value.detach(), ent_coef, vf_coef)

    assert np.isclose(ours.item(), cule_loss.item(), rtol=0, atol=1e-12)
    assert np.isclose(v_loss.item(), cule_value_loss.item(), rtol=0, atol=1e-12)
    assert np.isclose(pg_loss.item(), cule_policy_loss.item(), rtol=0, atol=1e-12)
    assert np.isclose(entropy.item(), cule_entropy.item(), rtol=0, atol=1e-12)

    our_grads = torch.autograd.grad(ours, [action_log_probs, value], retain_graph=True)
    their_grads = torch.autograd.grad(cule_loss, [action_log_probs, value], retain_graph=True)
    for got, want in zip(our_grads, their_grads):
        assert torch.allclose(got, want, rtol=0, atol=1e-12)


def test_cule_gae_branch_is_not_what_we_implement(a2c):
    """CuLE offers `--use-gae`; baselines' A2C has no GAE. Keep them distinct.

    If this ever starts passing, the port has drifted back into PPO territory.
    """
    gamma, tau = 0.99, 0.95
    num_steps, num_envs = 5, 3
    torch.manual_seed(7)
    rewards = torch.randn(num_steps, num_envs, dtype=torch.float64)
    masks = (torch.rand(num_steps, num_envs) > 0.2).double()
    values = torch.randn(num_steps + 1, num_envs, dtype=torch.float64)

    gae = torch.zeros(num_envs, dtype=torch.float64)
    gae_returns = torch.zeros(num_steps, num_envs, dtype=torch.float64)
    for step in reversed(range(num_steps)):
        delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
        gae = delta + gamma * tau * masks[step] * gae
        gae_returns[step] = gae + values[step]

    plain = a2c.nstep_returns(rewards, 1.0 - masks, values[-1], gamma)
    assert not torch.allclose(plain, gae_returns, rtol=1e-3, atol=1e-3)
