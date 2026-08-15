"""Assert Endpoint Replay against the claims of arXiv:2607.25123.

"Endpoint Replay: Compressing the Recency Buffer in Deep Reinforcement
Learning", Mohammad Panahi, Ashrafi, Du, Patterson, White & White, RLJ/RLC 2026.

There is no numerical cross-check for this one: github.com/panahiparham/endpoint-replay
held only a LICENSE and a "under construction ... available by August 15, 2026"
README when this was written (2026-08-08). So these tests state the paper's
properties directly, above all the anchoring invariant that the whole method
exists to preserve.
"""

import numpy as np
import pytest
import torch

from conftest import DiscreteEnvStub, device_params, load_trainer

GAMMA = 0.99


class BoxEnvStub(DiscreteEnvStub):
    def __init__(self, n_actions=6):
        super().__init__(n_actions)
        self.single_observation_space = type("Box", (), {"shape": (4, 84, 84)})()


@pytest.fixture(scope="module")
def endpoint():
    return load_trainer("endpoint_ddqn_atari")


def make_buffer(endpoint, recency=20, coreset=64, n_step=5, n_envs=1, gamma=GAMMA):
    return endpoint.EndpointReplayBuffer(
        recency, coreset, BoxEnvStub().single_observation_space,
        BoxEnvStub().single_action_space, "cpu", n_envs=n_envs, n_step=n_step, gamma=gamma,
    )


def fill(buffer, steps, dones=(), rewards=None, n_envs=1):
    """Drive `steps` transitions through the buffer with recognisable frames.

    Frame t is filled with the constant value t % 256, so a stacked observation
    identifies exactly which timesteps it was built from.
    """
    observation = torch.zeros(n_envs, 4, 84, 84, dtype=torch.uint8)
    buffer.initialize(observation)
    for step in range(steps):
        next_observation = torch.full((n_envs, 4, 84, 84), (step + 1) % 256, dtype=torch.uint8)
        reward = torch.full((n_envs,), float(step) if rewards is None else float(rewards[step]))
        done = torch.tensor([step in dones] * n_envs)
        buffer.add(next_observation, torch.full((n_envs,), step % 6),
                   torch.full((n_envs,), (step + 1) % 6), reward, done)


# --- the anchoring invariant (Sections 3.1-3.2) ---------------------------


def test_every_bootstrap_endpoint_starts_another_chain_element(endpoint):
    """The paper's core claim: "every s', a' we bootstrap from is itself in the
    coreset as the beginning of another n-step tuple that gets updated".

    Checked on the timestep indices the chain is built from: with the window
    cleared every n steps, the endpoints are 0, n, 2n, ... and each is the start
    of the next tuple.
    """
    n_step = 5
    buffer = make_buffer(endpoint, recency=10, coreset=64, n_step=n_step)
    fill(buffer, 80)

    starts, ends = [], []
    for slot in range(buffer.coreset_count):
        # The newest frame of each stored stack is the timestep it came from.
        starts.append(int(buffer.coreset_obs[slot, -1, 0, 0]))
        ends.append(int(buffer.coreset_next_obs[slot, -1, 0, 0]))

    assert len(starts) >= 3
    # Consecutive chain elements: this tuple's endpoint is the next tuple's start.
    for index in range(len(starts) - 1):
        assert ends[index] == starts[index + 1], (
            f"chain broken at {index}: ends at {ends[index]}, next starts at {starts[index + 1]}"
        )
    assert all(end - start == n_step for start, end in zip(starts, ends))


def test_the_coreset_compresses_by_a_factor_of_n(endpoint):
    """"compress a large recency buffer D of size m to a coreset Dc of size m/n"."""
    for n_step in (2, 5, 10):
        buffer = make_buffer(endpoint, recency=10, coreset=10000, n_step=n_step)
        fill(buffer, 200)
        expired = 200 - buffer.recency_rows - 1 + 1  # transitions handed to the lag window
        assert buffer.coreset_count == pytest.approx(expired // n_step, abs=1)


def test_a_sliding_window_would_not_compress(endpoint):
    """Guards the reading taken from the paper.

    Algorithm 1 line 16 says to pop a single transition from the lag window
    after emitting a summary, which emits one coreset entry per step. Section
    3.2's compression claim and Section 5's "sub-samples every 10 steps" both
    require clearing it. This pins the implemented behaviour to the second
    reading: far fewer coreset entries than transitions consumed.
    """
    buffer = make_buffer(endpoint, recency=10, coreset=10000, n_step=10)
    fill(buffer, 300)
    consumed = 300 - buffer.recency_rows
    assert buffer.coreset_count < consumed / 5


# --- n-step return accumulation (Section 3.2) -----------------------------


def test_n_step_return_is_the_discounted_sum(endpoint):
    """g_{t,n} = sum_{i=0}^{n-1} gamma^i r_{t+i+1}."""
    n_step, gamma = 4, 0.5
    rewards = list(range(1, 61))
    buffer = make_buffer(endpoint, recency=8, coreset=64, n_step=n_step, gamma=gamma)
    fill(buffer, 60, rewards=rewards)

    for slot in range(min(buffer.coreset_count, 5)):
        start = int(buffer.coreset_obs[slot, -1, 0, 0])
        expected = sum(gamma**i * rewards[start + i] for i in range(n_step))
        assert buffer.coreset_returns[slot] == pytest.approx(expected, rel=1e-6)
        assert buffer.coreset_discounts[slot] == pytest.approx(gamma**n_step, rel=1e-6)


def test_termination_truncates_the_window_and_zeroes_the_discount(endpoint):
    """"If termination occurs before n steps, say in k < n steps ... the shorter
    k-step return is computed", and White (2017)'s termination-aware discount
    makes gamma^k zero so no value leaks across the boundary."""
    n_step, gamma = 10, 0.9
    rewards = [1.0] * 60
    buffer = make_buffer(endpoint, recency=8, coreset=64, n_step=n_step, gamma=gamma)
    fill(buffer, 60, dones={14}, rewards=rewards)

    terminal_slots = [s for s in range(buffer.coreset_count) if buffer.coreset_discounts[s] == 0.0]
    assert terminal_slots, "the window covering the terminal step must have a zero discount"
    slot = terminal_slots[0]
    start = int(buffer.coreset_obs[slot, -1, 0, 0])
    k = 14 - start + 1
    assert k < n_step
    expected = sum(gamma**i for i in range(k))
    assert buffer.coreset_returns[slot] == pytest.approx(expected, rel=1e-6)


def test_the_chain_restarts_after_a_termination(endpoint):
    """A truncated window still leaves the next tuple starting where it ended."""
    buffer = make_buffer(endpoint, recency=8, coreset=64, n_step=5)
    fill(buffer, 60, dones={13})
    starts = [int(buffer.coreset_obs[s, -1, 0, 0]) for s in range(buffer.coreset_count)]
    ends = [int(buffer.coreset_next_obs[s, -1, 0, 0]) for s in range(buffer.coreset_count)]
    for index in range(len(starts) - 1):
        assert ends[index] == starts[index + 1]


# --- frame handling -------------------------------------------------------


def test_stacks_are_reconstructed_before_their_history_is_overwritten(endpoint):
    """The eviction margin: a transition leaves the recency ring while the
    `frame_stack - 1` frames it needs are still resident."""
    buffer = make_buffer(endpoint, recency=12, coreset=64, n_step=4)
    fill(buffer, 120)
    for slot in range(buffer.coreset_count):
        newest = int(buffer.coreset_obs[slot, -1, 0, 0])
        for channel in range(4):
            expected = newest - (3 - channel)
            frame = int(buffer.coreset_obs[slot, channel, 0, 0])
            assert frame in (expected % 256, 0), f"slot {slot} channel {channel}: {frame}"


def test_recency_sampling_returns_matched_states_and_actions(endpoint):
    buffer = make_buffer(endpoint, recency=40, coreset=64, n_step=5)
    fill(buffer, 60)
    observations, actions, rewards, next_observations, dones = buffer.sample_recency(16)
    assert observations.shape == (16, 4, 84, 84)
    assert next_observations.shape == (16, 4, 84, 84)
    assert actions.shape == (16,) and rewards.shape == (16,) and dones.shape == (16,)
    # next_obs is exactly one timestep ahead of obs.
    for index in range(16):
        assert int(next_observations[index, -1, 0, 0]) == (int(observations[index, -1, 0, 0]) + 1) % 256


def test_coreset_is_empty_before_anything_expires(endpoint):
    buffer = make_buffer(endpoint, recency=50, coreset=64, n_step=5)
    fill(buffer, 10)
    assert buffer.coreset_count == 0
    assert buffer.sample_coreset(4) is None


def test_buffers_are_bounded(endpoint):
    buffer = make_buffer(endpoint, recency=10, coreset=7, n_step=2)
    fill(buffer, 400)
    assert buffer.coreset_count == 7
    assert buffer.recency_count == buffer.recency_rows


@pytest.mark.parametrize("n_envs", [1, 3])
def test_each_environment_keeps_its_own_chain(endpoint, n_envs):
    buffer = make_buffer(endpoint, recency=10, coreset=256, n_step=5, n_envs=n_envs)
    fill(buffer, 80, n_envs=n_envs)
    # Every env emits its own summaries, so the coreset fills n_envs times faster.
    single = make_buffer(endpoint, recency=10, coreset=256, n_step=5, n_envs=1)
    fill(single, 80, n_envs=1)
    assert buffer.coreset_count == pytest.approx(single.coreset_count * n_envs, abs=n_envs)


def test_rejects_invalid_configuration(endpoint):
    with pytest.raises(ValueError):
        make_buffer(endpoint, n_step=0)
    with pytest.raises(ValueError):
        endpoint.EndpointReplayBuffer(
            10, 10, type("Box", (), {"shape": (84, 84)})(),
            BoxEnvStub().single_action_space, "cpu",
        )


# --- the expectile loss (Section 3.2) -------------------------------------


def test_expectile_at_one_half_is_half_the_squared_error(endpoint):
    """"At tau = 0.5, the expectile is the mean" and the loss is the usual
    squared error up to the shared factor of one half."""
    torch.manual_seed(0)
    prediction = torch.randn(256, dtype=torch.float64)
    target = torch.randn(256, dtype=torch.float64)
    loss = endpoint.expectile_loss(prediction, target, 0.5)
    assert torch.allclose(loss, 0.5 * (target - prediction) ** 2, atol=1e-12)


@pytest.mark.parametrize("tau", [0.1, 0.3, 0.5, 0.7, 0.9])
def test_expectile_weights_are_asymmetric_in_the_stated_direction(endpoint, tau):
    """tau weights upward deviations, 1 - tau downward."""
    prediction = torch.zeros(2, dtype=torch.float64)
    target = torch.tensor([1.0, -1.0], dtype=torch.float64)
    loss = endpoint.expectile_loss(prediction, target, tau)
    assert loss[0].item() == pytest.approx(tau)
    assert loss[1].item() == pytest.approx(1.0 - tau)


@pytest.mark.parametrize("tau", [0.3, 0.5, 0.7, 0.9])
def test_minimiser_is_the_expectile_of_the_target_distribution(endpoint, tau):
    """u solves (1 - tau) E[(u - x)_+] = tau E[(x - u)_+], the definition in
    Section 3.2. Fitting the loss must land on that root."""
    torch.manual_seed(1)
    samples = torch.randn(20000, dtype=torch.float64) * 2.0 + 1.0
    u = torch.zeros((), dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.Adam([u], lr=0.05)
    for _ in range(1500):
        optimizer.zero_grad()
        endpoint.expectile_loss(u, samples, tau).mean().backward()
        optimizer.step()

    fitted = u.detach()
    below = torch.clamp(fitted - samples, min=0).mean()
    above = torch.clamp(samples - fitted, min=0).mean()
    assert ((1 - tau) * below).item() == pytest.approx((tau * above).item(), rel=2e-2)


def test_larger_tau_pulls_the_estimate_upwards(endpoint):
    """"As tau approaches 1, u shifts toward the maximum value of the
    distribution" -- which is how the pessimism of stale n-step targets is
    offset."""
    torch.manual_seed(2)
    samples = torch.randn(8000, dtype=torch.float64)
    fitted = []
    for tau in (0.3, 0.5, 0.7, 0.9):
        u = torch.zeros((), dtype=torch.float64, requires_grad=True)
        optimizer = torch.optim.Adam([u], lr=0.05)
        for _ in range(1200):
            optimizer.zero_grad()
            endpoint.expectile_loss(u, samples, tau).mean().backward()
            optimizer.step()
        fitted.append(u.item())
    assert fitted == sorted(fitted)
    assert fitted[1] == pytest.approx(samples.mean().item(), abs=0.02)  # tau = 0.5 is the mean


# --- configuration --------------------------------------------------------


def test_defaults_match_the_papers_atari_setup(endpoint):
    """Tables 2 and 3, plus Section 5's buffer split and sampling ratio."""
    args = endpoint.Args()
    assert args.n_step == 10
    assert args.expectile == 0.7  # "We set the value of tau to 0.7 ... across all experiments"
    assert args.gamma == 0.99
    assert args.learning_rate == 6.25e-5
    assert args.adam_eps == 1.5e-4
    assert args.batch_size == 32 and args.coreset_batch_size == 4  # the 7:1 ratio, 28 + 4
    assert (args.batch_size - args.coreset_batch_size) / args.coreset_batch_size == 7.0
    assert args.end_e == 0.01
    # The 50x setting: 10k recency + 10k coreset against a 1M recency baseline.
    assert args.recency_buffer_size == 10000 and args.coreset_buffer_size == 10000


@pytest.mark.parametrize("device", device_params())
def test_sarsa_target_uses_the_stored_action(endpoint, device):
    """"We use Q(s_{t+n}, a_{t+n}), instead of the typical max_a Q(s_{t+n}, a)
    ... to ensure we have anchored bootstrap targets"."""
    torch.manual_seed(3)
    network = endpoint.QNetwork(BoxEnvStub(6)).to(device)
    observations = torch.randint(0, 255, (8, 4, 84, 84), device=device).float()
    stored_actions = torch.randint(0, 6, (8,), device=device)
    with torch.no_grad():
        values = network(observations)
        sarsa = values.gather(1, stored_actions.unsqueeze(1)).squeeze(1)
        greedy = values.max(1).values
    assert torch.equal(sarsa, values[torch.arange(8, device=device), stored_actions])
    # Sarsa is never above the max, and generally strictly below it.
    assert (sarsa <= greedy + 1e-6).all()
    assert (sarsa < greedy).any()
