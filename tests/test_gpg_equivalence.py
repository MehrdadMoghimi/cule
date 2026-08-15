"""GPG: the paper's Algorithm 2, and the properties it proves.

No code was released for this paper, so nothing here is a diff against upstream
source. Instead:

  * `Algorithm 2` is transcribed literally as a slow, explicit, bin-dictionary
    reference and the vectorised implementation is diffed against it on random
    inputs;
  * the estimator properties the paper *proves* are checked directly — that the
    baseline is action-independent (Prop. 1, so the gradient stays unbiased),
    that `--binning zero` reduces to GRPO outcome supervision, and that the
    advantages within a bin sum to zero;
  * the two things this port does differently from the paper — a group of
    parallel environments rather than trajectories from a shared start state,
    and truncated Monte-Carlo returns — are pinned as *limitations*, so they
    cannot quietly turn into claims.
"""

import numpy as np
import pytest
import torch

from conftest import DiscreteEnvStub, load_trainer

TRAINER = 'gpg_atari'


@pytest.fixture(scope='module')
def gpg():
    return load_trainer(TRAINER)


# ---------------------------------------------------------------------------
# Algorithm 2, transcribed literally
# ---------------------------------------------------------------------------

def reference_algorithm_2(rewards, next_dones, gamma, binning, group_mode):
    """The paper's Algorithm 2, written as bins-and-dictionaries.

        1: compute returns R^n_t
        2: initialize empty bins B
        3-7: for each n, for each t, insert R^n_t into B[f(s^n_t, t)]
             if f(s^n_t) != f(s^n_i) for i = 1..t-1   (first visit)
        8: A^n_t = R^n_t - mean(B[f(s^n_t, t)])
    """
    num_steps, num_envs = rewards.shape

    returns = np.zeros((num_steps, num_envs))
    for n in range(num_envs):
        running = 0.0
        for t in reversed(range(num_steps)):
            running = rewards[t, n] + gamma * running * (1.0 - next_dones[t, n])
            returns[t, n] = running

    def bin_of(t):
        return 0 if binning == 'zero' else t

    bins = {}
    for n in range(num_envs):
        seen = set()
        for t in range(num_steps):
            key = bin_of(t)
            if key in seen:
                continue
            seen.add(key)
            scope = (key,) if group_mode == 'batch' else (key, n)
            bins.setdefault(scope, []).append(returns[t, n])

    advantages = np.zeros((num_steps, num_envs))
    for n in range(num_envs):
        for t in range(num_steps):
            key = bin_of(t)
            scope = (key,) if group_mode == 'batch' else (key, n)
            values = bins.get(scope, [])
            baseline = float(np.mean(values)) if values else 0.0
            advantages[t, n] = returns[t, n] - baseline
    return returns, advantages


@pytest.mark.parametrize('binning', ['zero', 'timestep'])
@pytest.mark.parametrize('group_mode', ['batch', 'env'])
@pytest.mark.parametrize('seed', range(3))
def test_advantages_match_algorithm_2(gpg, binning, group_mode, seed):
    generator = torch.Generator().manual_seed(seed)
    num_steps, num_envs, gamma = 24, 6, 0.99
    rewards = torch.randn(num_steps, num_envs, generator=generator, dtype=torch.float64)
    next_dones = (torch.rand(num_steps, num_envs, generator=generator) < 0.15).double()

    got, keep = gpg.gpg_advantages(rewards, next_dones, gamma, binning, group_mode)
    _, want = reference_algorithm_2(
        rewards.numpy(), next_dones.numpy(), gamma, binning, group_mode)

    assert torch.allclose(got, torch.as_tensor(want), rtol=0, atol=1e-12)
    assert keep.all()  # drop_truncated is off


def test_monte_carlo_returns_are_cut_at_episode_boundaries(gpg):
    """No bootstrap and no leakage across a `done`."""
    gamma = 0.5
    rewards = torch.tensor([[1.0], [1.0], [1.0], [1.0]], dtype=torch.float64)
    next_dones = torch.tensor([[0.0], [1.0], [0.0], [0.0]], dtype=torch.float64)

    returns, complete = gpg.monte_carlo_returns(rewards, next_dones, gamma)
    # t=3: 1 (rollout ends, no bootstrap); t=2: 1 + .5 = 1.5
    # t=1: done -> 1; t=0: 1 + .5*1 = 1.5
    assert np.allclose(returns.squeeze(-1).numpy(), [1.5, 1.0, 1.5, 1.0], atol=1e-12)
    # Steps 0 and 1 reached a real terminal; steps 2 and 3 were truncated.
    assert np.allclose(complete.squeeze(-1).numpy(), [1.0, 1.0, 0.0, 0.0], atol=1e-12)


def test_no_bootstrap_value_is_used(gpg):
    """The critic-free claim, made concrete: `monte_carlo_returns` takes no
    value estimate at all and its signature cannot accept one."""
    import inspect
    parameters = list(inspect.signature(gpg.monte_carlo_returns).parameters)
    assert parameters == ['rewards', 'next_dones', 'gamma']


def test_agent_has_no_critic(gpg):
    """The other half of critic-free: fewer parameters than PPO's agent, and no
    value head to find."""
    ppo = load_trainer('ppo_atari')
    envs = DiscreteEnvStub(6)
    ours = gpg.Agent(envs)
    theirs = ppo.Agent(envs)

    assert not hasattr(ours, 'critic')
    assert hasattr(theirs, 'critic')
    assert sum(p.numel() for p in ours.parameters()) < sum(p.numel() for p in theirs.parameters())

    obs = torch.randint(0, 255, (4, 4, 84, 84), dtype=torch.uint8).float()
    action, logprob, entropy = ours.get_action(obs)
    assert action.shape == (4,) and logprob.shape == (4,) and entropy.shape == (4,)


# ---------------------------------------------------------------------------
# the first-visit rule
# ---------------------------------------------------------------------------

def test_first_visit_mask_marks_only_first_occurrences(gpg):
    bins = torch.tensor([[0, 5], [1, 5], [0, 6], [1, 5], [2, 6]])
    mask = gpg.first_visit_mask(bins)
    assert mask[:, 0].tolist() == [True, True, False, False, True]
    assert mask[:, 1].tolist() == [True, False, True, False, False]


def test_first_visit_is_per_trajectory_not_global(gpg):
    """Every trajectory contributes once to a bin, so a shared bin gets N
    entries, not one."""
    bins = torch.zeros((4, 3), dtype=torch.long)
    mask = gpg.first_visit_mask(bins)
    assert mask.sum().item() == 3            # one per trajectory
    assert mask[0].tolist() == [True] * 3


def test_first_visit_changes_the_baseline_when_bins_repeat(gpg):
    """With `--binning zero` a trajectory visits the single bin many times.
    Counting all of them would let one trajectory dominate its own baseline."""
    num_steps, num_envs = 8, 4
    returns = torch.arange(num_steps * num_envs, dtype=torch.float64).reshape(num_steps, num_envs)
    bins = torch.zeros((num_steps, num_envs), dtype=torch.long)

    first_visit = gpg.first_visit_mask(bins)
    with_rule = gpg.group_advantages(returns, bins, 'batch', first_visit)
    without_rule = gpg.group_advantages(returns, bins, 'batch', torch.ones_like(bins, dtype=torch.bool))

    assert not torch.allclose(with_rule, without_rule, rtol=1e-6, atol=1e-6)
    # The rule uses only the t=0 row, whose mean is (0+1+2+3)/4 = 1.5.
    assert np.isclose((returns - with_rule).mean().item(), 1.5, atol=1e-12)


def test_first_visit_is_vacuous_for_timestep_binning(gpg):
    """`f(s, t) = t` visits each bin exactly once per trajectory."""
    bins = gpg.bin_indices('timestep', 10, 5, torch.device('cpu'))
    assert gpg.first_visit_mask(bins).all()


# ---------------------------------------------------------------------------
# the estimator properties the paper proves
# ---------------------------------------------------------------------------

def test_baseline_does_not_depend_on_the_action(gpg):
    """Prop. 1: the baseline may be any function of state, and then the gradient
    is unbiased. Nothing in the estimator's signature sees an action."""
    import inspect
    for function in (gpg.gpg_advantages, gpg.group_advantages, gpg.bin_indices):
        assert not any('action' in name for name in inspect.signature(function).parameters)


def test_advantages_within_a_bin_sum_to_zero(gpg):
    """A mean-subtracted baseline centres each bin, which is the variance
    reduction the estimator is buying."""
    num_steps, num_envs = 12, 5
    torch.manual_seed(0)
    returns = torch.randn(num_steps, num_envs, dtype=torch.float64)
    bins = gpg.bin_indices('timestep', num_steps, num_envs, torch.device('cpu'))
    advantages = gpg.group_advantages(returns, bins, 'batch', gpg.first_visit_mask(bins))
    # One bin per timestep, so each row must be centred.
    assert torch.allclose(advantages.sum(dim=1), torch.zeros(num_steps, dtype=torch.float64), atol=1e-12)


def test_zero_binning_is_grpo_outcome_supervision(gpg):
    """`f(s, t) = 0` subtracts one group mean from everything — GRPO's
    outcome-supervised baseline (and REINFORCE++ once normalised)."""
    num_steps, num_envs, gamma = 10, 8, 0.99
    torch.manual_seed(0)
    rewards = torch.randn(num_steps, num_envs, dtype=torch.float64)
    next_dones = torch.zeros(num_steps, num_envs, dtype=torch.float64)

    advantages, _ = gpg.gpg_advantages(rewards, next_dones, gamma, 'zero', 'batch')
    returns, _ = gpg.monte_carlo_returns(rewards, next_dones, gamma)
    # First visit means only t=0 contributes: the mean full-episode return.
    baseline = returns[0].mean()
    assert torch.allclose(advantages, returns - baseline, rtol=0, atol=1e-12)


def test_timestep_binning_differs_from_zero_binning(gpg):
    """Bin granularity is the paper's central knob; the two must not coincide."""
    torch.manual_seed(0)
    rewards = torch.randn(16, 6, dtype=torch.float64)
    next_dones = torch.zeros(16, 6, dtype=torch.float64)
    zero, _ = gpg.gpg_advantages(rewards, next_dones, 0.99, 'zero', 'batch')
    timestep, _ = gpg.gpg_advantages(rewards, next_dones, 0.99, 'timestep', 'batch')
    assert not torch.allclose(zero, timestep, rtol=1e-3, atol=1e-3)


def test_group_modes_scope_the_bins_differently(gpg):
    torch.manual_seed(0)
    rewards = torch.randn(12, 4, dtype=torch.float64)
    next_dones = torch.zeros(12, 4, dtype=torch.float64)
    batch, _ = gpg.gpg_advantages(rewards, next_dones, 0.99, 'timestep', 'batch')
    per_env, _ = gpg.gpg_advantages(rewards, next_dones, 0.99, 'timestep', 'env')

    assert not torch.allclose(batch, per_env, rtol=1e-3, atol=1e-3)
    # With one bin per (t, env), each bin holds a single return, so every
    # advantage is exactly zero -- the degeneracy that motivates `batch`.
    assert torch.allclose(per_env, torch.zeros_like(per_env), atol=1e-12)


def test_per_state_binning_would_be_degenerate_on_pixels(gpg):
    """Why `f(s, t) = s` is not offered. If every observation is unique, every
    bin holds one return and every advantage collapses to zero."""
    num_steps, num_envs = 10, 4
    torch.manual_seed(0)
    returns = torch.randn(num_steps, num_envs, dtype=torch.float64)
    unique_bins = torch.arange(num_steps * num_envs).reshape(num_steps, num_envs)
    advantages = gpg.group_advantages(returns, unique_bins, 'batch',
                                      gpg.first_visit_mask(unique_bins))
    assert torch.allclose(advantages, torch.zeros_like(advantages), atol=1e-12)


def test_unsupported_binning_and_group_mode_are_rejected(gpg):
    with pytest.raises(ValueError, match='unsupported binning'):
        gpg.bin_indices('state', 4, 2, torch.device('cpu'))
    with pytest.raises(ValueError, match='unsupported group_mode'):
        gpg.group_advantages(torch.zeros(2, 2), torch.zeros(2, 2, dtype=torch.long), 'cloned')


# ---------------------------------------------------------------------------
# the truncation caveat
# ---------------------------------------------------------------------------

def test_drop_truncated_keeps_only_completed_returns(gpg):
    gamma = 0.99
    rewards = torch.ones(6, 2, dtype=torch.float64)
    next_dones = torch.zeros(6, 2, dtype=torch.float64)
    next_dones[3, 0] = 1.0  # env 0 terminates at t=3; env 1 never does

    _, keep = gpg.gpg_advantages(rewards, next_dones, gamma, 'timestep', 'batch',
                                 drop_truncated=True)
    assert keep[:, 0].tolist() == [True, True, True, True, False, False]
    assert not keep[:, 1].any()


def test_truncated_returns_are_biased_downwards(gpg):
    """The reason the flag exists, demonstrated rather than asserted."""
    gamma = 0.99
    rewards = torch.ones(50, 1, dtype=torch.float64)
    next_dones = torch.zeros(50, 1, dtype=torch.float64)
    returns, complete = gpg.monte_carlo_returns(rewards, next_dones, gamma)
    assert not complete.any()
    # The last step sees one reward; the first sees fifty. Same true value.
    assert returns[-1].item() == pytest.approx(1.0)
    assert returns[0].item() > 39.0


def test_drop_truncated_also_excludes_those_steps_from_the_baseline(gpg):
    """A truncated return must not pollute the bin mean it is excluded from."""
    gamma = 0.99
    rewards = torch.ones(8, 3, dtype=torch.float64)
    next_dones = torch.zeros(8, 3, dtype=torch.float64)
    # Env 0 terminates early, so its return is genuinely shorter than the other
    # two -- a done on the *last* step would make all three returns identical
    # and the two baselines would coincide by accident.
    next_dones[3, 0] = 1.0

    kept, _ = gpg.gpg_advantages(rewards, next_dones, gamma, 'zero', 'batch',
                                 drop_truncated=True)
    all_steps, _ = gpg.gpg_advantages(rewards, next_dones, gamma, 'zero', 'batch',
                                      drop_truncated=False)
    assert not torch.allclose(kept, all_steps, rtol=1e-6, atol=1e-6)

    # With the flag, only env 0's completed t=0 return sets the baseline.
    returns, _ = gpg.monte_carlo_returns(rewards, next_dones, gamma)
    assert np.isclose((returns - kept).mean().item(), returns[0, 0].item(), atol=1e-12)
    assert np.isclose((returns - all_steps).mean().item(), returns[0].mean().item(), atol=1e-12)


def test_no_dones_leaves_everything_truncated(gpg):
    _, keep = gpg.gpg_advantages(
        torch.zeros(5, 3, dtype=torch.float64), torch.zeros(5, 3, dtype=torch.float64),
        0.99, 'timestep', 'batch', drop_truncated=True)
    assert not keep.any()


# ---------------------------------------------------------------------------
# limitations, pinned so they cannot become claims
# ---------------------------------------------------------------------------

def test_the_group_is_parallel_environments_not_a_cloned_state(gpg):
    """The paper's group is N trajectories from *one* start state. Nothing here
    clones environment state, and `bin_indices` never sees an observation — so
    the port cannot be claiming the paper's estimator, only a valid
    state-independent baseline in its place."""
    import inspect
    parameters = list(inspect.signature(gpg.bin_indices).parameters)
    assert parameters == ['binning', 'num_steps', 'num_envs', 'device']
    assert 'obs' not in parameters and 'states' not in parameters


def test_empty_bins_fall_back_to_a_zero_baseline(gpg):
    """A bin with no eligible entries has no mean; zero is the only sane value,
    and it must not become a NaN."""
    returns = torch.randn(4, 2, dtype=torch.float64)
    bins = torch.zeros((4, 2), dtype=torch.long)
    nothing_eligible = torch.zeros((4, 2), dtype=torch.bool)
    advantages = gpg.group_advantages(returns, bins, 'batch', nothing_eligible)
    assert torch.isfinite(advantages).all()
    assert torch.allclose(advantages, returns, rtol=0, atol=1e-15)
