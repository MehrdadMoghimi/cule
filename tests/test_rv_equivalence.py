"""Assert Relative Value Learning against the theory of arXiv:2607.21120.

"Relative Value Learning", Hoeftmann, Robine & Harmeling, ICLR 2026.

`tests/crosscheck/check_rv.py` diffs the port numerically against the official
implementation at github.com/Hauf3n/relative-value-learning. These tests are
the complement: they state the paper's theorems directly, so they still hold if
the upstream checkout is absent.
"""

import numpy as np
import pytest
import torch

from conftest import DiscreteEnvStub, device_params, load_trainer

GAMMA, LAMBDA = 0.99, 0.95


class BoxEnvStub(DiscreteEnvStub):
    def __init__(self, n_actions=6):
        super().__init__(n_actions)
        self.single_observation_space = type("Box", (), {"shape": (4, 84, 84)})()


@pytest.fixture(scope="module")
def rv():
    return load_trainer("ppo_rv_atari")


# --- the antisymmetric critic (Section 3, Equation 30) ---------------------


@pytest.mark.parametrize("head", ["linear", "mlp_symmetric"])
def test_head_is_exactly_antisymmetric(rv, head):
    """"a single learned vector w ... to ensure antisymmetry ... and
    Delta(s_i, s_i) = 0 by design" (Section 5.1).

    The design claim is about Phi, so it is checked on Phi: given the same two
    encodings, Phi(z_i - z_j) = -Phi(z_j - z_i) bit for bit, and Phi(0) = 0.
    """
    torch.manual_seed(0)
    agent = rv.Agent(BoxEnvStub(6), head).double()
    encoded_i = torch.randn(7, 512, dtype=torch.float64)
    encoded_j = torch.randn(7, 512, dtype=torch.float64)
    assert torch.equal(
        agent.encoding_to_rv(encoded_i, encoded_j), -agent.encoding_to_rv(encoded_j, encoded_i)
    )
    assert torch.equal(
        agent.encoding_to_rv(encoded_i, encoded_i), torch.zeros(7, dtype=torch.float64)
    )


@pytest.mark.parametrize("head", ["linear", "mlp_symmetric"])
def test_get_rv_is_antisymmetric_up_to_batching_noise(rv, head):
    """End to end the property survives, but not bit for bit: get_rv encodes
    the two arguments in one concatenated batch, and a sample's convolution
    output depends slightly on where it sits in that batch."""
    torch.manual_seed(0)
    agent = rv.Agent(BoxEnvStub(6), head).double()
    x = torch.randint(0, 255, (7, 4, 84, 84)).double()
    y = torch.randint(0, 255, (7, 4, 84, 84)).double()
    assert torch.allclose(agent.get_rv(x, y), -agent.get_rv(y, x), atol=1e-12)
    assert torch.allclose(agent.get_rv(x, x), torch.zeros(7, dtype=torch.float64), atol=1e-12)


@pytest.mark.parametrize("head", ["linear", "mlp_symmetric"])
def test_the_head_carries_no_bias(rv, head):
    """A bias anywhere in Phi would break antisymmetry."""
    agent = rv.Agent(BoxEnvStub(6), head)
    for module in agent.rv_head.modules():
        if isinstance(module, torch.nn.Linear):
            assert module.bias is None


def test_unknown_head_is_rejected(rv):
    with pytest.raises(ValueError):
        rv.Agent(BoxEnvStub(6), "quadratic")


# --- Theorem 3.1: the pairwise Bellman operator is a gamma-contraction -----


def _tabular_operator(delta, reward, transition, gamma):
    """(T_pi Delta)(i, j) = r(i) - r(j) + gamma E[Delta(i', j')], Equation 8."""
    successor = transition @ delta @ transition.T
    return (reward[:, None] - reward[None, :]) + gamma * successor


def test_pairwise_operator_is_a_gamma_contraction(rv):
    """Theorem 3.1: ||T d1 - T d2||_inf <= gamma ||d1 - d2||_inf."""
    generator = np.random.default_rng(0)
    states = 6
    transition = generator.random((states, states))
    transition /= transition.sum(1, keepdims=True)
    reward = generator.standard_normal(states)

    for _ in range(50):
        first = generator.standard_normal((states, states))
        first = first - first.T  # antisymmetric, as F requires
        second = generator.standard_normal((states, states))
        second = second - second.T
        applied_first = _tabular_operator(first, reward, transition, GAMMA)
        applied_second = _tabular_operator(second, reward, transition, GAMMA)
        assert np.abs(applied_first - applied_second).max() <= GAMMA * np.abs(first - second).max() + 1e-12


def test_the_operator_preserves_antisymmetry(rv):
    """T maps F into F: the space of bounded antisymmetric functions."""
    generator = np.random.default_rng(1)
    states = 5
    transition = generator.random((states, states))
    transition /= transition.sum(1, keepdims=True)
    reward = generator.standard_normal(states)
    delta = generator.standard_normal((states, states))
    delta = delta - delta.T
    applied = _tabular_operator(delta, reward, transition, GAMMA)
    assert np.abs(applied + applied.T).max() < 1e-12


def test_fixed_point_equals_the_true_value_differences(rv):
    """Theorem 3.1: the unique fixed point is Delta(i, j) = V(i) - V(j)."""
    generator = np.random.default_rng(2)
    states = 6
    transition = generator.random((states, states))
    transition /= transition.sum(1, keepdims=True)
    reward = generator.standard_normal(states)

    delta = np.zeros((states, states))
    for _ in range(4000):
        delta = _tabular_operator(delta, reward, transition, GAMMA)

    values = np.linalg.solve(np.eye(states) - GAMMA * transition, reward)
    assert np.abs(delta - (values[:, None] - values[None, :])).max() < 1e-8


# --- Lemma 3.2 and Corollary 3.3: R-GAE = GAE + a trajectory constant ------


def test_r_gae_equals_gae_plus_the_trajectory_constant(rv):
    """Lemma 3.2: A~_t = A_t + B_t with B_t = (1 - gamma) C sum_l (gamma lambda)^l."""
    torch.manual_seed(3)
    steps, actors = 40, 2
    rewards = torch.randn(steps, actors, dtype=torch.float64)
    next_done = torch.zeros(steps, actors)  # one uninterrupted fragment
    values = torch.randn(steps + 1, actors, dtype=torch.float64)

    # Absolute-value GAE, the textbook recursion.
    absolute = torch.zeros(steps, actors, dtype=torch.float64)
    running = torch.zeros(actors, dtype=torch.float64)
    for t in reversed(range(steps)):
        residual = rewards[t] + GAMMA * values[t + 1] - values[t]
        running = residual + GAMMA * LAMBDA * running
        absolute[t] = running

    # Relative values are the same function anchored at zero: V~ = V - V(s_0).
    constant = values[0].clone()
    delta = values[1:] - values[:-1]
    relative = rv.relative_values(delta, next_done, None)
    assert torch.allclose(relative, values - constant, atol=1e-9)

    advantages = rv.relative_gae(relative, rewards, next_done, GAMMA, LAMBDA)
    powers = torch.tensor(
        [sum((GAMMA * LAMBDA) ** l for l in range(steps - t)) for t in range(steps)],
        dtype=torch.float64,
    )
    expected_offset = (1 - GAMMA) * constant.unsqueeze(0) * powers.unsqueeze(1)
    assert torch.allclose(advantages - absolute, expected_offset, atol=1e-9)


def test_the_offset_is_constant_across_actions_at_each_timestep(rv):
    """Corollary 3.3 turns on B_t not depending on a_t, so E[grad log pi * B_t] = 0."""
    torch.manual_seed(4)
    steps = 30
    rewards = torch.randn(steps, 1, dtype=torch.float64)
    next_done = torch.zeros(steps, 1)
    values = torch.randn(steps + 1, 1, dtype=torch.float64)
    delta = values[1:] - values[:-1]

    baseline = rv.relative_gae(
        rv.relative_values(delta, next_done, None), rewards, next_done, GAMMA, LAMBDA
    )
    # Shift the whole value function by an arbitrary constant: A~ must not move,
    # because the zero anchor removes exactly that degree of freedom.
    shifted_values = values + 7.5
    shifted_delta = shifted_values[1:] - shifted_values[:-1]
    shifted = rv.relative_gae(
        rv.relative_values(shifted_delta, next_done, None), rewards, next_done, GAMMA, LAMBDA
    )
    assert torch.allclose(baseline, shifted, atol=1e-12)


def test_relative_values_reanchor_at_episode_boundaries(rv):
    torch.manual_seed(5)
    steps = 10
    delta = torch.ones(steps, 1, dtype=torch.float64)
    next_done = torch.zeros(steps, 1)
    next_done[4] = 1.0
    values = rv.relative_values(delta, next_done, None)
    assert values[0].item() == 0.0
    assert values[4].item() == 4.0  # the last state of the first episode
    assert values[5].item() == 0.0  # re-anchored
    assert values[6].item() == 1.0


# --- Section 4.1: trajectory ranking --------------------------------------


def test_start_state_offsets_match_the_quadratic_form(rv):
    """The O(N) shortcut is exact for the linear head (Equation 25)."""
    torch.manual_seed(6)
    agent = rv.Agent(BoxEnvStub(6), "linear").double()
    encoded = torch.randn(9, 512, dtype=torch.float64)

    fast = rv.start_state_offsets(encoded, agent)
    n = encoded.shape[0]
    matrix = agent.encoding_to_rv(
        encoded.unsqueeze(1).expand(n, n, -1).reshape(n * n, -1),
        encoded.unsqueeze(0).expand(n, n, -1).reshape(n * n, -1),
    ).view(n, n)
    slow = matrix.mean(dim=1)
    slow = slow - slow.min()
    assert torch.allclose(fast, slow, atol=1e-10)


def test_offsets_are_non_negative_with_a_zero_minimum(rv):
    """"subtract the batch minimum to get non-negative values with ranking"."""
    torch.manual_seed(7)
    agent = rv.Agent(BoxEnvStub(6), "linear").double()
    offsets = rv.start_state_offsets(torch.randn(12, 512, dtype=torch.float64), agent)
    assert (offsets >= 0).all()
    assert offsets.min().item() == pytest.approx(0.0, abs=1e-12)


def test_start_mask_marks_row_zero_and_every_episode_start(rv):
    episode_start = torch.zeros(6, 2)
    episode_start[3, 0] = 1.0
    mask = rv.find_start_states_in_batch(episode_start)
    assert mask[0].all()
    assert bool(mask[3, 0]) and not bool(mask[3, 1])
    assert int(mask.sum()) == 3


# --- Section 3.4: well-posed targets --------------------------------------


class _LookupAgent:
    """Stands in for the critic so the target arithmetic can be checked alone."""

    linear_head = True

    def __init__(self, values):
        self.values = values

    def encoding_to_rv(self, encoded_i, encoded_j):
        return self.values[encoded_i[:, 0].long()] - self.values[encoded_j[:, 0].long()]


def _flat_batch(rv_module, rewards, next_done, gamma):
    import tensordict

    count = rewards.shape[0]
    data = tensordict.TensorDict(
        {
            "rewards": rewards.unsqueeze(1),
            "next_done": next_done.unsqueeze(1),
            "obs": torch.arange(count, dtype=torch.float64).view(count, 1, 1),
            "next_obs": (torch.arange(count, dtype=torch.float64) + 1).view(count, 1, 1),
        },
        batch_size=[count, 1],
    )
    rv_module.prepare_data(data, gamma)
    flat = data.view(-1)
    flat["encoded_obs"] = flat["obs"].view(count, 1)
    flat["encoded_next_obs"] = flat["next_obs"].view(count, 1)
    return flat


def test_one_step_target_without_terminals(rv):
    """y = (r_i - r_j) + gamma Delta(s_i+1, s_j+1), Equations 19-20 case one."""
    rewards = torch.tensor([0.5, -1.0, 2.0, 0.25, 1.5, -0.75], dtype=torch.float64)
    next_done = torch.zeros(6, dtype=torch.float64)
    flat = _flat_batch(rv, rewards, next_done, GAMMA)
    values = torch.arange(8, dtype=torch.float64) * 0.3
    agent = _LookupAgent(values)

    idx_i = torch.tensor([0, 1, 2])
    idx_j = torch.tensor([3, 4, 5])
    target, _ = rv.rv_n_step_target(flat, idx_i, idx_j, agent, GAMMA, 1)
    expected = (rewards[idx_i] - rewards[idx_j]) + GAMMA * (values[idx_i + 1] - values[idx_j + 1])
    assert torch.allclose(target, expected, atol=1e-12)


def test_terminal_cases_avoid_bootstrapping_off_absolute_values(rv):
    """Equation 20: a terminal successor is rewritten with observable rewards."""
    rewards = torch.tensor([0.5, -1.0, 2.0, 0.25, 1.5, -0.75], dtype=torch.float64)
    next_done = torch.zeros(6, dtype=torch.float64)
    next_done[0] = 1.0  # s_i terminates
    next_done[4] = 1.0  # s_j terminates
    flat = _flat_batch(rv, rewards, next_done, GAMMA)
    values = torch.arange(8, dtype=torch.float64) * 0.3
    agent = _LookupAgent(values)

    delta = lambda a, b: values[a] - values[b]

    # d_i = 1, d_j = 0: Delta(s_i, s_j+1) - r_i
    target, _ = rv.rv_n_step_target(flat, torch.tensor([0]), torch.tensor([3]), agent, GAMMA, 1)
    expected = (rewards[0] - rewards[3]) + GAMMA * (delta(0, 4) - rewards[0])
    assert target.item() == pytest.approx(expected.item(), abs=1e-12)

    # d_i = 0, d_j = 1: Delta(s_i+1, s_j) + r_j
    target, _ = rv.rv_n_step_target(flat, torch.tensor([1]), torch.tensor([4]), agent, GAMMA, 1)
    expected = (rewards[1] - rewards[4]) + GAMMA * (delta(2, 4) + rewards[4])
    assert target.item() == pytest.approx(expected.item(), abs=1e-12)

    # both terminal: the variance-reducing default of zero
    target, _ = rv.rv_n_step_target(flat, torch.tensor([0]), torch.tensor([4]), agent, GAMMA, 1)
    assert target.item() == pytest.approx(0.0, abs=1e-12)


def test_n_step_horizon_never_crosses_a_terminal(rv):
    """"with the assumption that neither trajectory terminates within the
    n-step window" -- the horizon is truncated to make that true."""
    rewards = torch.zeros(12, dtype=torch.float64)
    next_done = torch.zeros(12, dtype=torch.float64)
    next_done[5] = 1.0
    flat = _flat_batch(rv, rewards, next_done, GAMMA)
    horizons = flat["max_n_step"]
    assert horizons[0].item() == 6  # six transitions up to and including index 5
    assert horizons[5].item() == 1
    assert horizons[6].item() == 6  # to the end of the buffer


def test_discounted_reward_sums_truncate_at_the_boundary(rv):
    rewards = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float64)
    next_done = torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0], dtype=torch.float64)
    sums, horizons = rv.compute_discounted_reward_sums(rewards, next_done, 0.5)
    assert sums[0, 0].item() == pytest.approx(1.0)
    assert sums[0, 1].item() == pytest.approx(1.0 + 0.5 * 2.0)
    assert sums[0, 2].item() == pytest.approx(1.0 + 0.5 * 2.0 + 0.25 * 3.0)
    # Nothing past the terminal at index 2 contributes.
    assert sums[0, 3].item() == pytest.approx(0.0)
    assert horizons[0].item() == 3


# --- Appendix D: pair sampling --------------------------------------------


def test_pair_sampling_honours_p_same_and_excludes_self(rv):
    offsets = torch.tensor([0, 20, 45, 80])
    anchors = torch.arange(80)
    for p_same in (0.0, 0.33, 1.0):
        same = 0
        trials = 300
        for trial in range(trials):
            generator = torch.Generator().manual_seed(trial)
            partners = rv.get_target_indices(anchors, offsets, p_same=p_same, generator=generator)
            assert (partners != anchors).all(), "self-pairs carry no signal"
            assert (partners >= 0).all() and (partners < 80).all()
            episode_of = lambda index: torch.bucketize(index, offsets[1:], right=True)
            same += int((episode_of(partners) == episode_of(anchors)).sum())
        rate = same / (trials * 80)
        if p_same == 1.0:
            assert rate > 0.99
        elif p_same == 0.0:
            assert rate < 0.45  # the base rate of landing in one's own episode
        else:
            assert 0.35 < rate < 0.65


# --- Section 5: the training objective ------------------------------------


def test_clipped_value_loss_matches_the_ppo_form(rv):
    torch.manual_seed(8)
    predicted = torch.randn(256, dtype=torch.float64)
    target = torch.randn(256, dtype=torch.float64)
    old = predicted + 0.4 * torch.randn(256, dtype=torch.float64)
    loss, fraction = rv.rv_loss(predicted, target, old, True, 0.15)

    clipped = old + (predicted - old).clamp(-0.15, 0.15)
    expected = torch.max((predicted - target) ** 2, (clipped - target) ** 2).mean()
    assert loss.item() == pytest.approx(expected.item(), abs=1e-12)
    assert fraction.item() == pytest.approx(((predicted - old).abs() > 0.15).double().mean().item())
    # The clipped branch can only ever raise the loss, never lower it.
    assert loss.item() >= ((predicted - target) ** 2).mean().item() - 1e-12


def test_hyperparameters_match_the_papers_table(rv):
    """Table 3, plus the two places the official code differs from it."""
    args = rv.Args()
    assert (args.gamma, args.gae_lambda) == (0.99, 0.95)
    assert args.clip_coef == 0.1
    assert args.update_epochs == 5
    assert args.num_steps == 128 and args.num_envs == 8
    assert args.num_minibatches == 8  # 1024 / 8 = the table's minibatch size of 128
    assert args.learning_rate == 2.5e-4
    assert args.rv_coef == 1.25
    assert args.clip_rv == 0.15
    assert args.max_grad_norm == 0.5
    assert args.p_same_episode == 0.33
    # The table says n-step 5; the official code anneals 6 -> 5, which is what
    # this port follows, and the entropy coefficient likewise (0.00875 vs 0.01).
    assert (args.n_step_cutoff, args.n_step_cutoff_minimum) == (6, 5)
    assert args.ent_coef == 0.00875


@pytest.mark.parametrize("device", device_params())
def test_agent_runs_end_to_end_on_device(rv, device):
    torch.manual_seed(9)
    agent = rv.Agent(BoxEnvStub(6), "linear").to(device)
    observations = torch.randint(0, 255, (4, 4, 84, 84), device=device).float()
    action, logprob, entropy = agent.get_action(observations)
    assert action.shape == (4,) and logprob.shape == (4,) and entropy.shape == (4,)
    assert agent.get_rv(observations, observations.flip(0)).shape == (4,)
