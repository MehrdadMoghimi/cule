"""A diagnostic MDP whose optimal value function is known in closed form.

The equivalence tests in `tests/` check that each trainer computes the right
*formulas*, and the cross-checks in `tests/crosscheck/` check that its
components match the authors' official code.  Neither checks the wiring around
those components: terminal masking, the discount exponent, the target-network
path, the order of the replay window.  A bug in any of those still trains on
Breakout, just worse, and "worse" is invisible against seed noise.

This module closes that gap.  It is a tiny MDP rendered into the same 84x84
uint8 four-frame observations Atari produces, exposed through the same vector
interface as `cule_env.CuLEVectorEnv`, so every trainer and every network in
this repository runs on it unmodified.  Because the MDP is trivial, its optimal
action-value function can be written down exactly, and a trained network's
Q-values can be compared against that number rather than against a noisy score.

The MDP -- "Chain"
------------------
States are `0 .. n_states - 1`; every episode starts in state 0.  Action 0
(ADVANCE) moves one state forward.  Every other action (STAY) leaves the state
alone -- they exist so the action set looks like an Atari action set and so the
argmax has somewhere wrong to go.  All rewards are zero except ADVANCE taken in
the last state, which terminates the episode and pays `reward` with probability
`success_prob` and `-reward` otherwise.

Rewards live in {-1, 0, +1} so that the sign-clipping every trainer applies is
the identity, and the observation is a white square whose position encodes the
state, which a Nature CNN separates in a few hundred updates.

The closed form
---------------
Write `mu` for the mean terminal reward, `reward * (2 * success_prob - 1)`.  The
agent's action never changes the reward distribution, only how many steps it
waits, so for `success_prob > 1/2` ADVANCE is optimal everywhere and

    V*(s)          = gamma ** (n_states - 1 - s) * mu
    Q*(s, ADVANCE) = gamma ** (n_states - 1 - s) * mu
    Q*(s, STAY)    = gamma * V*(s) = gamma ** (n_states - s) * mu

The last state is the interesting one: `Q*(n-1, ADVANCE) = mu` exactly, with no
bootstrap term, because the transition is terminal.  An implementation that
forgets to mask the bootstrap at a terminal state does not diverge and does not
change its greedy action -- it converges to a different, equally self-consistent
fixed point that is the correct one scaled by `1 / (1 - gamma ** n_states)`.
`q_without_terminal_mask` returns it, so a failed comparison can name the bug
instead of just reporting a distance.

`tests/test_diagnostic_env.py` re-derives all of these by tabular value
iteration, so the closed form itself is checked rather than assumed.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import gymnasium as gym
import numpy as np
import torch

# The observation is an 84x84 grid of 12x12 cells; a state lights exactly one.
GRID = 7
CELL = 12
MAX_STATES = GRID * GRID

ADVANCE = 0


@dataclass(frozen=True)
class ChainSpec:
    """The MDP's parameters.  Note that `gamma` is deliberately absent.

    The discount belongs to the agent, not to the environment, so every oracle
    below takes it as an argument.  A comparison is only meaningful when the
    gamma passed to the oracle is the gamma the trainer ran with.
    """

    n_states: int = 5
    n_actions: int = 4
    reward: float = 1.0
    success_prob: float = 1.0

    def __post_init__(self) -> None:
        if not 2 <= self.n_states <= MAX_STATES:
            raise ValueError(f"n_states must be in [2, {MAX_STATES}], got {self.n_states}")
        if self.n_actions < 2:
            raise ValueError("n_actions must be at least 2 so STAY exists")
        if not 0.5 < self.success_prob <= 1.0:
            # At success_prob <= 1/2 the mean terminal reward is non-positive and
            # waiting forever becomes optimal, which has no finite closed form.
            raise ValueError("success_prob must be in (0.5, 1.0]")
        if self.reward <= 0:
            raise ValueError("reward must be positive")

    @property
    def mean_terminal_reward(self) -> float:
        return self.reward * (2.0 * self.success_prob - 1.0)

    @property
    def is_deterministic(self) -> bool:
        return self.success_prob == 1.0


# ---------------------------------------------------------------------------
# the closed form
# ---------------------------------------------------------------------------

def optimal_q(spec: ChainSpec, gamma: float) -> np.ndarray:
    """`Q*` as an `(n_states, n_actions)` array, in float64."""
    if not 0.0 < gamma < 1.0:
        raise ValueError("gamma must be in (0, 1)")
    mu = spec.mean_terminal_reward
    steps_to_goal = np.arange(spec.n_states - 1, -1, -1, dtype=np.float64)
    q = np.empty((spec.n_states, spec.n_actions), dtype=np.float64)
    q[:, ADVANCE] = gamma**steps_to_goal * mu
    # STAY burns one step and lands back where it started, so it is worth
    # exactly one extra discount factor.
    q[:, ADVANCE + 1 :] = (gamma ** (steps_to_goal + 1.0) * mu)[:, None]
    return q


def optimal_v(spec: ChainSpec, gamma: float) -> np.ndarray:
    return optimal_q(spec, gamma).max(axis=1)


def q_without_terminal_mask(spec: ChainSpec, gamma: float) -> np.ndarray:
    """The fixed point reached when the terminal transition still bootstraps.

    With the mask missing, the terminal step pays `mu + gamma * V(0)` and the
    task becomes a continuing loop of period `n_states`.  Solving
    `V(s) = gamma ** (n - 1 - s) * (mu + gamma * V(0))` gives back `Q*` scaled
    by `1 / (1 - gamma ** n)` -- same greedy policy, larger magnitudes.
    """
    return optimal_q(spec, gamma) / (1.0 - gamma**spec.n_states)


def optimal_return_distribution(
    spec: ChainSpec, gamma: float, state: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Atoms and probabilities of the optimal return from `state`.

    C51, QR-DQN, IQN, FQF and DisCo predict this whole object, so they can be
    checked against it directly instead of against its mean.
    """
    if not 0 <= state < spec.n_states:
        raise ValueError(f"state must be in [0, {spec.n_states})")
    scale = gamma ** (spec.n_states - 1 - state) * spec.reward
    if spec.is_deterministic:
        return np.array([scale]), np.array([1.0])
    return (
        np.array([scale, -scale]),
        np.array([spec.success_prob, 1.0 - spec.success_prob]),
    )


def optimal_episode_length(spec: ChainSpec) -> int:
    """Steps the optimal policy needs to finish an episode."""
    return spec.n_states


def optimal_undiscounted_return(spec: ChainSpec) -> float:
    """What a `charts/episodic_return` curve converges to, for sanity checks."""
    return spec.mean_terminal_reward


def value_iteration(spec: ChainSpec, gamma: float, iterations: int = 20_000) -> np.ndarray:
    """Tabular `Q*`, used to verify the closed form rather than to replace it."""
    mu = spec.mean_terminal_reward
    q = np.zeros((spec.n_states, spec.n_actions), dtype=np.float64)
    for _ in range(iterations):
        v = q.max(axis=1)
        updated = np.empty_like(q)
        # ADVANCE from the last state is terminal: reward only, no bootstrap.
        updated[:-1, ADVANCE] = gamma * v[1:]
        updated[-1, ADVANCE] = mu
        updated[:, ADVANCE + 1 :] = (gamma * v)[:, None]
        if np.abs(updated - q).max() < 1e-15:
            return updated
        q = updated
    return q


# ---------------------------------------------------------------------------
# the environment
# ---------------------------------------------------------------------------

def _sprites(spec: ChainSpec, device: torch.device) -> torch.Tensor:
    """One 84x84 frame per state: a single lit cell, at 0 or 255."""
    frames = torch.zeros((spec.n_states, 84, 84), dtype=torch.uint8, device=device)
    for state in range(spec.n_states):
        row, column = divmod(state, GRID)
        frames[state, row * CELL : (row + 1) * CELL, column * CELL : (column + 1) * CELL] = 255
    return frames


class DiagnosticVectorEnv:
    """`cule_env.CuLEVectorEnv`'s interface, backed by `ChainSpec`.

    Observation layout, frame-stack ring, reward clipping, auto-reset, the
    zeroing of the stack on termination and the batched `final_info` flush all
    mirror `CuLEVectorEnv` deliberately: the point is to exercise the tensor
    path the trainers actually take, not a simplified stand-in for it.
    """

    is_cule = True
    is_diagnostic = True

    def __init__(
        self,
        spec: ChainSpec,
        num_envs: int,
        device: torch.device | str,
        seed: int = 1,
        frame_stack: int = 4,
        clip_rewards: bool = True,
        stats_interval: int = 1,
        **_: Any,
    ) -> None:
        self.spec = spec
        self.device = torch.device(device)
        self.num_envs = num_envs
        self.frame_stack = frame_stack
        self.clip_rewards = clip_rewards
        self.stats_interval = max(1, stats_interval)
        self._step_count = 0
        self._seed = seed
        self._generator = torch.Generator(device=self.device).manual_seed(seed)

        self.single_action_space = gym.spaces.Discrete(spec.n_actions)
        self.action_space = self.single_action_space
        self.single_action_space.seed(seed)
        self.single_observation_space = gym.spaces.Box(
            low=0, high=255, shape=(frame_stack, 84, 84), dtype=np.uint8
        )
        self.observation_space = self.single_observation_space

        self._sprite_table = _sprites(spec, self.device)
        self._states = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self._frames = torch.zeros(
            (num_envs, 2 * frame_stack, 84, 84), dtype=torch.uint8, device=self.device
        )
        self._ring_pos = frame_stack - 1
        self._lives = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self._episode_returns = torch.zeros(num_envs, device=self.device)
        self._episode_lengths = torch.zeros(num_envs, dtype=torch.int64, device=self.device)
        self._returned_returns = torch.zeros_like(self._episode_returns)
        self._returned_lengths = torch.zeros_like(self._episode_lengths)
        self._pending_episodes = torch.zeros(num_envs, dtype=torch.bool, device=self.device)

    def _write_frame(self) -> None:
        frame = self._sprite_table[self._states]
        self._frames[:, self._ring_pos].copy_(frame)
        self._frames[:, self._ring_pos + self.frame_stack].copy_(frame)

    def _obs_view(self) -> torch.Tensor:
        start = self._ring_pos + 1
        return self._frames[:, start : start + self.frame_stack]

    def reset(self, *, seed: int | None = None, **_: Any) -> tuple[torch.Tensor, dict[str, Any]]:
        if seed is not None:
            self._generator.manual_seed(seed)
        self._states.zero_()
        self._frames.zero_()
        self._ring_pos = self.frame_stack - 1
        self._write_frame()
        self._episode_returns.zero_()
        self._episode_lengths.zero_()
        self._pending_episodes.zero_()
        return self._obs_view(), {}

    def _flush_episode_infos(self) -> dict[str, Any]:
        if self._step_count % self.stats_interval:
            return {}
        pending = self._pending_episodes.detach().cpu().numpy()
        if not pending.any():
            return {}
        returns = self._returned_returns.detach().cpu().numpy()
        lengths = self._returned_lengths.detach().cpu().numpy()
        final_info: list[dict[str, Any] | None] = [None] * self.num_envs
        for index in np.flatnonzero(pending):
            final_info[index] = {
                "episode": {"r": float(returns[index]), "l": int(lengths[index])}
            }
        self._pending_episodes.zero_()
        return {"final_info": final_info}

    def step(
        self, actions: torch.Tensor | np.ndarray
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        actions = torch.as_tensor(actions, device=self.device).reshape(self.num_envs)
        advancing = actions.eq(ADVANCE)
        at_goal = self._states.eq(self.spec.n_states - 1)
        terminated = advancing & at_goal

        rewards = torch.zeros(self.num_envs, device=self.device)
        if terminated.any():
            if self.spec.is_deterministic:
                payout = torch.full_like(rewards, self.spec.reward)
            else:
                draw = torch.rand(
                    self.num_envs, generator=self._generator, device=self.device
                )
                payout = torch.where(
                    draw < self.spec.success_prob, self.spec.reward, -self.spec.reward
                )
            rewards = torch.where(terminated, payout, rewards)

        self._states = torch.where(advancing & ~at_goal, self._states + 1, self._states)
        self._states = torch.where(terminated, torch.zeros_like(self._states), self._states)

        self._episode_returns.add_(rewards)
        self._episode_lengths.add_(1)
        self._returned_returns.copy_(
            torch.where(terminated, self._episode_returns, self._returned_returns)
        )
        self._returned_lengths.copy_(
            torch.where(terminated, self._episode_lengths, self._returned_lengths)
        )
        self._pending_episodes.logical_or_(terminated)
        self._episode_returns.masked_fill_(terminated, 0)
        self._episode_lengths.masked_fill_(terminated, 0)

        self._ring_pos = (self._ring_pos + 1) % self.frame_stack
        self._frames.mul_((~terminated).view(-1, 1, 1, 1))
        self._write_frame()

        self._step_count += 1
        infos = self._flush_episode_infos()
        infos["lives"] = self._lives
        truncations = torch.zeros_like(terminated)
        training_rewards = rewards.sign() if self.clip_rewards else rewards
        return self._obs_view(), training_rewards, terminated, truncations, infos

    def close(self) -> None:
        return None


def canonical_observations(
    spec: ChainSpec, frame_stack: int = 4, device: torch.device | str = "cpu"
) -> torch.Tensor:
    """The observation the optimal policy sees in each state, `(n_states, k, 84, 84)`.

    Frame stacking means one MDP state has several possible observations -- the
    history in the older frames differs -- so "the network's Q at state s" is
    only well posed once a stack is named.  The stack along the optimal
    trajectory is the right one to name: it is the on-policy distribution, the
    only place Q* accuracy is required, and it is what the greedy rollout
    actually encounters.
    """
    device = torch.device(device)
    sprites = _sprites(spec, device)
    ring = torch.zeros((2 * frame_stack, 84, 84), dtype=torch.uint8, device=device)
    observations = torch.empty(
        (spec.n_states, frame_stack, 84, 84), dtype=torch.uint8, device=device
    )
    ring_pos = frame_stack - 1
    for state in range(spec.n_states):
        if state:
            ring_pos = (ring_pos + 1) % frame_stack
        ring[ring_pos] = sprites[state]
        ring[ring_pos + frame_stack] = sprites[state]
        observations[state] = ring[ring_pos + 1 : ring_pos + 1 + frame_stack]
    return observations


# ---------------------------------------------------------------------------
# env-id plumbing
# ---------------------------------------------------------------------------

REGISTRY: dict[str, ChainSpec] = {
    # The default is short and deterministic: with gamma=0.9 the four Q* values
    # span 0.656 to 1.0 and ADVANCE beats STAY by 10% everywhere, which is a
    # wide enough margin that a network cannot pass by being vaguely right.
    "Chain-v0": ChainSpec(),
    "ChainLong-v0": ChainSpec(n_states=12),
    # For the distributional family: a two-atom return with a known split.
    "ChainStochastic-v0": ChainSpec(success_prob=0.8),
}

_FIELDS = {"n_states": int, "n_actions": int, "reward": float, "success_prob": float}


def is_diagnostic_env_id(env_id: str) -> bool:
    return env_id.split(":", 1)[0] in REGISTRY


def parse_diagnostic_env_id(env_id: str) -> ChainSpec:
    """`Chain-v0` or `Chain-v0:n_states=8,success_prob=0.9`."""
    name, _, overrides = env_id.partition(":")
    if name not in REGISTRY:
        raise ValueError(f"unknown diagnostic env id {env_id!r}; known: {sorted(REGISTRY)}")
    spec = REGISTRY[name]
    if not overrides:
        return spec
    changes = {}
    for item in overrides.split(","):
        key, _, value = item.partition("=")
        key = key.strip()
        if key not in _FIELDS:
            raise ValueError(f"unknown ChainSpec field {key!r}; known: {sorted(_FIELDS)}")
        changes[key] = _FIELDS[key](value)
    return replace(spec, **changes)


def make_diagnostic_env(
    env_id: str,
    num_envs: int,
    device: torch.device | str,
    seed: int = 1,
    frame_stack: int = 4,
    **kwargs: Any,
) -> DiagnosticVectorEnv:
    return DiagnosticVectorEnv(
        parse_diagnostic_env_id(env_id), num_envs, device,
        seed=seed, frame_stack=frame_stack, **kwargs,
    )
