"""The closed-form MDP: is the oracle right, and does the env match it?

`benchmarks/verify_chain.py` judges a trainer by comparing its Q against
`diagnostic_env.optimal_q`.  That comparison is only worth anything if the
formula is actually `Q*` for the process the environment actually runs, so this
file checks both halves against each other:

  * the closed form against tabular value iteration, at several discounts and
    chain lengths -- the arithmetic is re-derived, never assumed;
  * the environment's transitions, rewards, termination and frame stacking
    against the transition model that value iteration was given;
  * `canonical_observations` against the observations the environment really
    produces along the optimal trajectory, since those are the inputs at which
    a trainer's Q gets read.
"""

import sys

import numpy as np
import pytest
import torch

from conftest import CLEANRL_DIR, device_params

if CLEANRL_DIR not in sys.path:
    sys.path.insert(0, CLEANRL_DIR)

import diagnostic_env as diag  # noqa: E402


SPECS = [
    diag.ChainSpec(),
    diag.ChainSpec(n_states=2),
    diag.ChainSpec(n_states=12, n_actions=6),
    diag.ChainSpec(success_prob=0.8),
    diag.ChainSpec(n_states=7, success_prob=0.75, reward=1.0),
]
GAMMAS = [0.5, 0.9, 0.95, 0.99]


# ---------------------------------------------------------------------------
# the closed form
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('spec', SPECS)
@pytest.mark.parametrize('gamma', GAMMAS)
def test_closed_form_matches_value_iteration(spec, gamma):
    closed_form = diag.optimal_q(spec, gamma)
    tabular = diag.value_iteration(spec, gamma)
    assert np.abs(closed_form - tabular).max() < 1e-12


@pytest.mark.parametrize('spec', SPECS)
@pytest.mark.parametrize('gamma', GAMMAS)
def test_closed_form_satisfies_the_bellman_optimality_equation(spec, gamma):
    """The direct check, independent of how value iteration is written."""
    q = diag.optimal_q(spec, gamma)
    v = q.max(axis=1)
    # ADVANCE from the goal is terminal, so its value is the mean reward alone.
    assert q[-1, diag.ADVANCE] == pytest.approx(spec.mean_terminal_reward, abs=1e-15)
    for state in range(spec.n_states - 1):
        assert q[state, diag.ADVANCE] == pytest.approx(gamma * v[state + 1], abs=1e-15)
    for state in range(spec.n_states):
        for action in range(1, spec.n_actions):
            assert q[state, action] == pytest.approx(gamma * v[state], abs=1e-15)


@pytest.mark.parametrize('spec', SPECS)
@pytest.mark.parametrize('gamma', GAMMAS)
def test_advance_is_strictly_optimal_everywhere(spec, gamma):
    """The whole design rests on this: STAY must be a real, losing alternative."""
    q = diag.optimal_q(spec, gamma)
    assert (q[:, diag.ADVANCE, None] > q[:, 1:] + 1e-12).all()


@pytest.mark.parametrize('spec', SPECS)
@pytest.mark.parametrize('gamma', GAMMAS)
def test_unmasked_fixed_point_is_self_consistent(spec, gamma):
    """`q_without_terminal_mask` must solve the *buggy* Bellman equation.

    Otherwise the harness would be naming a bug signature that the bug does not
    actually produce.
    """
    q = diag.q_without_terminal_mask(spec, gamma)
    v = q.max(axis=1)
    # The terminal transition pays the reward and then bootstraps off the state
    # the environment resets into, which is state 0.
    assert q[-1, diag.ADVANCE] == pytest.approx(
        spec.mean_terminal_reward + gamma * v[0], rel=1e-12)
    for state in range(spec.n_states - 1):
        assert q[state, diag.ADVANCE] == pytest.approx(gamma * v[state + 1], rel=1e-12)
    assert q[0, diag.ADVANCE] / diag.optimal_q(spec, gamma)[0, diag.ADVANCE] == pytest.approx(
        1.0 / (1.0 - gamma ** spec.n_states), rel=1e-12)


def test_spec_rejects_parameters_with_no_closed_form():
    with pytest.raises(ValueError):
        diag.ChainSpec(success_prob=0.5)   # waiting forever becomes optimal
    with pytest.raises(ValueError):
        diag.ChainSpec(n_states=1)
    with pytest.raises(ValueError):
        diag.ChainSpec(n_actions=1)        # no STAY, so the argmax is trivial
    with pytest.raises(ValueError):
        diag.ChainSpec(n_states=diag.MAX_STATES + 1)  # would not fit the 7x7 grid
    with pytest.raises(ValueError):
        diag.optimal_q(diag.ChainSpec(), gamma=1.0)


@pytest.mark.parametrize('gamma', GAMMAS)
def test_return_distribution_has_the_right_mean(gamma):
    for spec in SPECS:
        for state in range(spec.n_states):
            atoms, weights = diag.optimal_return_distribution(spec, gamma, state)
            assert weights.sum() == pytest.approx(1.0)
            assert float(atoms @ weights) == pytest.approx(
                diag.optimal_v(spec, gamma)[state], rel=1e-12)


# ---------------------------------------------------------------------------
# the environment
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('device', device_params())
def test_advance_walks_the_chain_and_stay_does_not(device):
    spec = diag.ChainSpec(n_states=5, n_actions=4)
    env = diag.DiagnosticVectorEnv(spec, num_envs=2, device=device)
    env.reset()
    assert env._states.tolist() == [0, 0]
    # env 0 advances, env 1 stays.
    for expected in range(1, spec.n_states):
        _, rewards, terminated, _, _ = env.step(torch.tensor([0, 1], device=device))
        assert env._states.tolist() == [expected, 0]
        assert rewards.sum().item() == 0.0
        assert not terminated.any()


@pytest.mark.parametrize('device', device_params())
def test_the_goal_pays_once_terminates_and_resets(device):
    spec = diag.ChainSpec(n_states=3)
    env = diag.DiagnosticVectorEnv(spec, num_envs=4, device=device)
    env.reset()
    advance = torch.zeros(4, dtype=torch.long, device=device)
    for _ in range(spec.n_states - 1):
        _, rewards, terminated, _, _ = env.step(advance)
        assert not terminated.any()
    observations, rewards, terminated, truncations, infos = env.step(advance)
    assert terminated.all()
    assert not truncations.any()
    assert rewards.tolist() == [spec.reward] * 4
    # Auto-reset, and the stale frames are cleared exactly as CuLE clears them.
    assert env._states.tolist() == [0, 0, 0, 0]
    assert (observations[:, :-1] == 0).all()
    assert (observations[:, -1] == env._sprite_table[0]).all()
    episodes = infos['final_info']
    assert all(info['episode'] == {'r': spec.reward, 'l': spec.n_states} for info in episodes)


@pytest.mark.parametrize('device', device_params())
def test_reward_clipping_is_the_identity_here(device):
    """Every trainer sign-clips; the MDP is built so that changes nothing."""
    spec = diag.ChainSpec(n_states=2, success_prob=0.75)
    clipped = diag.DiagnosticVectorEnv(spec, 64, device, seed=7, clip_rewards=True)
    unclipped = diag.DiagnosticVectorEnv(spec, 64, device, seed=7, clip_rewards=False)
    clipped.reset()
    unclipped.reset()
    advance = torch.zeros(64, dtype=torch.long, device=device)
    for _ in range(spec.n_states * 20):
        _, left, _, _, _ = clipped.step(advance)
        _, right, _, _, _ = unclipped.step(advance)
        assert torch.equal(left, right)


@pytest.mark.parametrize('device', device_params())
def test_the_optimal_policy_earns_the_optimal_return(device):
    """The end-to-end sanity check: the oracle's claim, measured."""
    spec = diag.ChainSpec(n_states=6, success_prob=0.8)
    gamma = 0.9
    env = diag.DiagnosticVectorEnv(spec, num_envs=4096, device=device, seed=3)
    env.reset()
    advance = torch.zeros(4096, dtype=torch.long, device=device)
    discounted = torch.zeros(4096, device=device)
    for step in range(spec.n_states):
        _, rewards, terminated, _, _ = env.step(advance)
        discounted += gamma**step * rewards
    assert terminated.all()
    assert discounted.mean().item() == pytest.approx(
        diag.optimal_v(spec, gamma)[0], abs=0.02)


@pytest.mark.parametrize('device', device_params())
def test_the_stochastic_payout_matches_success_prob(device):
    spec = diag.ChainSpec(n_states=2, success_prob=0.8)
    env = diag.DiagnosticVectorEnv(spec, num_envs=8192, device=device, seed=11)
    env.reset()
    advance = torch.zeros(8192, dtype=torch.long, device=device)
    env.step(advance)
    _, rewards, terminated, _, _ = env.step(advance)
    assert terminated.all()
    assert rewards.mean().item() == pytest.approx(spec.mean_terminal_reward, abs=0.02)
    assert set(rewards.unique().tolist()) == {-1.0, 1.0}


@pytest.mark.parametrize('device', device_params())
@pytest.mark.parametrize('frame_stack', [1, 4])
def test_canonical_observations_are_what_the_env_produces(device, frame_stack):
    """The comparison in `verify_chain.py` reads Q at these exact tensors.

    If they drifted from the environment's real output the trainer would be
    scored on inputs it never saw, and a passing run would mean nothing.
    """
    spec = diag.ChainSpec(n_states=8)
    canonical = diag.canonical_observations(spec, frame_stack, device)
    env = diag.DiagnosticVectorEnv(spec, num_envs=1, device=device, frame_stack=frame_stack)
    observations, _ = env.reset()
    assert torch.equal(observations[0], canonical[0])
    advance = torch.zeros(1, dtype=torch.long, device=device)
    for state in range(1, spec.n_states):
        observations, _, _, _, _ = env.step(advance)
        assert torch.equal(observations[0], canonical[state])


@pytest.mark.parametrize('device', device_params())
def test_observations_separate_the_states(device):
    """A trivially learnable encoding, but it does have to be injective."""
    spec = diag.ChainSpec(n_states=diag.MAX_STATES)
    frames = diag._sprites(spec, torch.device(device))
    assert frames.shape == (spec.n_states, 84, 84)
    assert set(frames.unique().tolist()) == {0, 255}
    flattened = frames.reshape(spec.n_states, -1).float()
    overlaps = flattened @ flattened.T
    # Each sprite overlaps itself and nothing else.
    assert torch.equal(overlaps.nonzero()[:, 0], overlaps.nonzero()[:, 1])


# ---------------------------------------------------------------------------
# the env-id plumbing every trainer reaches this MDP through
# ---------------------------------------------------------------------------

def test_env_ids_route_and_parse():
    assert diag.is_diagnostic_env_id('Chain-v0')
    assert diag.is_diagnostic_env_id('Chain-v0:n_states=9')
    assert not diag.is_diagnostic_env_id('Breakout-v5')
    assert diag.parse_diagnostic_env_id('Chain-v0') == diag.ChainSpec()
    assert diag.parse_diagnostic_env_id('Chain-v0:n_states=9,success_prob=0.7') == \
        diag.ChainSpec(n_states=9, success_prob=0.7)
    with pytest.raises(ValueError):
        diag.parse_diagnostic_env_id('Breakout-v5')
    with pytest.raises(ValueError):
        diag.parse_diagnostic_env_id('Chain-v0:gamma=0.9')  # gamma is the agent's


def test_make_cule_env_dispatches_to_the_diagnostic_backend():
    """The one line that gives all 62 trainers this harness for free."""
    from cule_env import make_cule_env
    env = make_cule_env('Chain-v0', 4, 'cpu', seed=1)
    assert isinstance(env, diag.DiagnosticVectorEnv)
    assert env.is_cule and env.is_diagnostic
    assert env.single_observation_space.shape == (4, 84, 84)
    assert env.single_action_space.n == diag.ChainSpec().n_actions
