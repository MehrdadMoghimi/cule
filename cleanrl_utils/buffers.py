"""Small replay buffer subset required by the bundled CleanRL Atari scripts."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import torch


class ReplayBufferSamples(NamedTuple):
    observations: torch.Tensor
    actions: torch.Tensor
    next_observations: torch.Tensor
    dones: torch.Tensor
    rewards: torch.Tensor


class PrioritizedReplayBufferSamples(NamedTuple):
    observations: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_observations: torch.Tensor
    dones: torch.Tensor
    indices: np.ndarray
    weights: torch.Tensor


def _as_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


class ReplayBuffer:
    """NumPy replay storage compatible with CleanRL's SB3-style calls."""

    def __init__(
        self,
        buffer_size,
        observation_space,
        action_space,
        device,
        n_envs=1,
        optimize_memory_usage=False,
        handle_timeout_termination=True,
    ):
        self.buffer_size = max(int(buffer_size) // n_envs, 1)
        self.device = torch.device(device)
        self.n_envs = n_envs
        self.optimize_memory_usage = optimize_memory_usage
        self.handle_timeout_termination = handle_timeout_termination
        self.pos = 0
        self.full = False

        obs_shape = observation_space.shape
        self.observations = np.zeros(
            (self.buffer_size, n_envs, *obs_shape), dtype=observation_space.dtype
        )
        self.next_observations = None
        if not optimize_memory_usage:
            self.next_observations = np.zeros_like(self.observations)

        action_dim = int(np.prod(action_space.shape)) if action_space.shape else 1
        self.actions = np.zeros((self.buffer_size, n_envs, action_dim), dtype=np.int64)
        self.rewards = np.zeros((self.buffer_size, n_envs), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size, n_envs), dtype=np.float32)
        self.timeouts = np.zeros((self.buffer_size, n_envs), dtype=np.float32)

    def add(self, obs, next_obs, action, reward, done, infos):
        obs = _as_numpy(obs).reshape((self.n_envs, *self.observations.shape[2:]))
        next_obs = _as_numpy(next_obs).reshape((self.n_envs, *self.observations.shape[2:]))
        action = _as_numpy(action).reshape((self.n_envs, self.actions.shape[-1]))

        self.observations[self.pos] = obs
        if self.optimize_memory_usage:
            self.observations[(self.pos + 1) % self.buffer_size] = next_obs
        else:
            self.next_observations[self.pos] = next_obs
        self.actions[self.pos] = action
        self.rewards[self.pos] = _as_numpy(reward).reshape(self.n_envs)
        self.dones[self.pos] = _as_numpy(done).reshape(self.n_envs)

        if self.handle_timeout_termination:
            self.timeouts[self.pos] = np.asarray(
                [info.get("TimeLimit.truncated", False) for info in infos], dtype=np.float32
            )

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def sample(self, batch_size):
        if self.full:
            if self.optimize_memory_usage:
                batch_inds = (
                    np.random.randint(1, self.buffer_size, size=batch_size) + self.pos
                ) % self.buffer_size
            else:
                batch_inds = np.random.randint(0, self.buffer_size, size=batch_size)
        else:
            upper_bound = self.pos
            if upper_bound == 0:
                raise RuntimeError("cannot sample from an empty replay buffer")
            batch_inds = np.random.randint(0, upper_bound, size=batch_size)
        env_indices = np.random.randint(0, self.n_envs, size=batch_size)

        if self.optimize_memory_usage:
            next_obs = self.observations[(batch_inds + 1) % self.buffer_size, env_indices]
        else:
            next_obs = self.next_observations[batch_inds, env_indices]

        dones = self.dones[batch_inds, env_indices]
        if self.handle_timeout_termination:
            dones = dones * (1.0 - self.timeouts[batch_inds, env_indices])

        return ReplayBufferSamples(
            observations=self._to_torch(self.observations[batch_inds, env_indices]),
            actions=self._to_torch(self.actions[batch_inds, env_indices]),
            next_observations=self._to_torch(next_obs),
            dones=self._to_torch(dones.reshape(-1, 1)),
            rewards=self._to_torch(self.rewards[batch_inds, env_indices].reshape(-1, 1)),
        )

    def _to_torch(self, array):
        return torch.as_tensor(array, device=self.device)


class AtariReplayBuffer:
    """Frame-efficient replay storage for synchronous vector Atari environments.

    Only the newest 84x84 frame is copied from each stacked observation. Four
    frame states are reconstructed while sampling, respecting both episode
    boundaries and circular-buffer boundaries. ``buffer_size`` is measured in
    individual transitions rather than vector environment steps.
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space,
        action_space,
        device,
        n_envs: int,
        frame_stack: int = 4,
    ):
        if len(observation_space.shape) != 3 or observation_space.shape[0] != frame_stack:
            raise ValueError(
                f"expected ({frame_stack}, H, W) Atari observations, got {observation_space.shape}"
            )
        if not hasattr(action_space, "n"):
            raise ValueError("AtariReplayBuffer only supports discrete actions")

        self.device = torch.device(device)
        self.n_envs = int(n_envs)
        self.frame_stack = int(frame_stack)
        self.height, self.width = observation_space.shape[-2:]
        # One row is always the current observation and is not yet a transition.
        self.time_capacity = max(int(np.ceil(buffer_size / self.n_envs)) + 1, frame_stack + 2)
        self.buffer_size = (self.time_capacity - 1) * self.n_envs

        self.frames = np.zeros(
            (self.time_capacity, self.n_envs, self.height, self.width), dtype=np.uint8
        )
        self.actions = np.zeros((self.time_capacity, self.n_envs), dtype=np.int64)
        self.rewards = np.zeros((self.time_capacity, self.n_envs), dtype=np.float32)
        self.dones = np.zeros((self.time_capacity, self.n_envs), dtype=np.bool_)
        self.frame_ids = np.full(self.time_capacity, -1, dtype=np.int64)
        self.transition_ids = np.full(self.time_capacity, -1, dtype=np.int64)

        self.pos = 0
        self.steps = 0
        self.num_transitions = 0
        self.initialized = False

    @staticmethod
    def _numpy(value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
        return np.asarray(value)

    def _latest_frames(self, observations) -> np.ndarray:
        if isinstance(observations, torch.Tensor):
            observations = observations[:, -1].detach().cpu().numpy()
        else:
            observations = np.asarray(observations)[:, -1]
        return observations.reshape(self.n_envs, self.height, self.width).astype(np.uint8, copy=False)

    def initialize(self, observations) -> None:
        """Store the initial current frame before the first environment step."""
        self.frames[self.pos] = self._latest_frames(observations)
        self.frame_ids[self.pos] = self.steps
        self.initialized = True

    def add(self, next_observations, actions, rewards, dones) -> None:
        if not self.initialized:
            raise RuntimeError("call replay_buffer.initialize(initial_observations) before add")

        row = self.pos
        next_row = (row + 1) % self.time_capacity
        self.actions[row] = self._numpy(actions).reshape(self.n_envs)
        self.rewards[row] = self._numpy(rewards).reshape(self.n_envs)
        self.dones[row] = self._numpy(dones).reshape(self.n_envs).astype(bool, copy=False)
        self.transition_ids[row] = self.steps

        # The current frame was written by initialize() or the previous add().
        # Copy only the newest next frame, not all four stacked frames.
        self.frames[next_row] = self._latest_frames(next_observations)
        self.frame_ids[next_row] = self.steps + 1

        self.pos = next_row
        self.steps += 1
        self.num_transitions = min(self.num_transitions + self.n_envs, self.buffer_size)

    def __len__(self) -> int:
        return self.num_transitions

    @property
    def stored_rows(self) -> int:
        return min(self.steps, self.time_capacity - 1)

    def _sample_rows(self, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
        if self.num_transitions == 0:
            raise RuntimeError("cannot sample from an empty replay buffer")
        ages = np.random.randint(1, self.stored_rows + 1, size=batch_size)
        rows = (self.pos - ages) % self.time_capacity
        env_indices = np.random.randint(0, self.n_envs, size=batch_size)
        return rows, env_indices

    def _encode_stack(
        self, rows: np.ndarray, env_indices: np.ndarray, expected_ids: np.ndarray
    ) -> np.ndarray:
        offsets = np.arange(self.frame_stack - 1, -1, -1, dtype=np.int64)
        frame_rows = (rows[:, None] - offsets[None, :]) % self.time_capacity
        stacks = self.frames[frame_rows, env_indices[:, None]].copy()
        valid = self.frame_ids[frame_rows] == (expected_ids[:, None] - offsets[None, :])

        # A frame preceding a terminal transition belongs to an older episode.
        for channel, offset in enumerate(offsets):
            if offset == 0:
                continue
            transition_rows = (
                rows[:, None] - np.arange(offset, 0, -1, dtype=np.int64)[None, :]
            ) % self.time_capacity
            crossed_episode = self.dones[transition_rows, env_indices[:, None]].any(axis=1)
            valid[:, channel] &= ~crossed_episode
        stacks *= valid[:, :, None, None]
        return stacks

    def _encode_samples(
        self,
        rows: np.ndarray,
        env_indices: np.ndarray,
        n_step: int = 1,
        gamma: float = 1.0,
    ) -> ReplayBufferSamples:
        start_ids = self.transition_ids[rows]
        observations = self._encode_stack(rows, env_indices, start_ids)

        returns = np.zeros(len(rows), dtype=np.float32)
        dones = np.zeros(len(rows), dtype=np.bool_)
        alive = np.ones(len(rows), dtype=np.bool_)
        for offset in range(n_step):
            transition_rows = (rows + offset) % self.time_capacity
            expected = start_ids + offset
            contiguous = self.transition_ids[transition_rows] == expected
            if not contiguous.all():
                raise RuntimeError("sampled replay transition does not have enough contiguous successors")
            step_rewards = self.rewards[transition_rows, env_indices]
            step_dones = self.dones[transition_rows, env_indices]
            returns += (gamma**offset) * step_rewards * alive
            dones |= alive & step_dones
            alive &= ~step_dones

        next_rows = (rows + n_step) % self.time_capacity
        next_observations = self._encode_stack(next_rows, env_indices, start_ids + n_step)
        actions = self.actions[rows, env_indices].reshape(-1, 1)
        return ReplayBufferSamples(
            observations=self._to_torch(observations),
            actions=self._to_torch(actions),
            next_observations=self._to_torch(next_observations),
            dones=self._to_torch(dones.astype(np.float32).reshape(-1, 1)),
            rewards=self._to_torch(returns.reshape(-1, 1)),
        )

    def sample(self, batch_size: int) -> ReplayBufferSamples:
        rows, env_indices = self._sample_rows(batch_size)
        return self._encode_samples(rows, env_indices)

    def _to_torch(self, array: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(array, device=self.device)


class _BatchSumTree:
    """Power-of-two sum tree with vectorized batched updates and sampling."""

    def __init__(self, capacity: int):
        self.data_capacity = int(capacity)
        self.capacity = 1 << max(0, (self.data_capacity - 1).bit_length())
        self.tree = np.zeros(2 * self.capacity, dtype=np.float32)

    @property
    def total(self) -> float:
        return float(self.tree[1])

    def update(self, indices, values) -> None:
        indices = np.asarray(indices, dtype=np.int64).reshape(-1)
        values = np.asarray(values, dtype=np.float32).reshape(-1)
        tree_indices = indices + self.capacity
        self.tree[tree_indices] = values
        tree_indices = np.unique(tree_indices // 2)
        while tree_indices.size and tree_indices[0] >= 1:
            self.tree[tree_indices] = self.tree[2 * tree_indices] + self.tree[2 * tree_indices + 1]
            if tree_indices[0] == 1:
                break
            tree_indices = np.unique(tree_indices // 2)

    def values(self, indices) -> np.ndarray:
        return self.tree[np.asarray(indices, dtype=np.int64) + self.capacity]

    def sample(self, batch_size: int) -> np.ndarray:
        total = self.total
        if total <= 0:
            raise RuntimeError("cannot sample from an empty prioritized replay buffer")
        values = (np.arange(batch_size, dtype=np.float64) + np.random.random(batch_size)) * (
            total / batch_size
        )
        tree_indices = np.ones(batch_size, dtype=np.int64)
        while tree_indices[0] < self.capacity:
            left = tree_indices * 2
            left_values = self.tree[left]
            go_right = values >= left_values
            values = np.where(go_right, values - left_values, values)
            tree_indices = np.where(go_right, left + 1, left)
        indices = tree_indices - self.capacity
        if (indices >= self.data_capacity).any():
            raise RuntimeError("prioritized replay sampled outside its data capacity")
        return indices


class PrioritizedAtariReplayBuffer(AtariReplayBuffer):
    """Vectorized n-step prioritized replay using frame-efficient storage."""

    def __init__(
        self,
        buffer_size: int,
        observation_space,
        action_space,
        device,
        n_envs: int,
        n_step: int,
        gamma: float,
        alpha: float = 0.6,
        beta: float = 0.4,
        eps: float = 1e-6,
        frame_stack: int = 4,
    ):
        super().__init__(buffer_size, observation_space, action_space, device, n_envs, frame_stack)
        self.n_step = int(n_step)
        self.gamma = float(gamma)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.eps = float(eps)
        self.max_priority = 1.0
        self.sum_tree = _BatchSumTree(self.time_capacity * self.n_envs)

    def _flat_indices(self, rows, env_indices) -> np.ndarray:
        return np.asarray(rows, dtype=np.int64) * self.n_envs + np.asarray(env_indices, dtype=np.int64)

    def add(self, next_observations, actions, rewards, dones) -> None:
        super().add(next_observations, actions, rewards, dones)

        # self.pos is the current, not-yet-transition row. Any old priorities in
        # that row now refer to overwritten data and must be removed.
        env_indices = np.arange(self.n_envs, dtype=np.int64)
        overwritten = self._flat_indices(np.full(self.n_envs, self.pos), env_indices)
        self.sum_tree.update(overwritten, np.zeros(self.n_envs, dtype=np.float32))

        if self.steps < self.n_step:
            return
        candidate_row = (self.pos - self.n_step) % self.time_capacity
        candidate_id = self.steps - self.n_step
        if self.transition_ids[candidate_row] != candidate_id:
            return

        valid = np.ones(self.n_envs, dtype=np.bool_)
        # Match the usual deque implementation: do not create starts whose
        # n-step window crosses an earlier terminal transition.
        for offset in range(self.n_step - 1):
            row = (candidate_row + offset) % self.time_capacity
            valid &= ~self.dones[row]
        indices = self._flat_indices(np.full(self.n_envs, candidate_row), env_indices)
        priorities = np.where(valid, self.max_priority**self.alpha, 0.0).astype(np.float32)
        self.sum_tree.update(indices, priorities)

    def sample(self, batch_size: int) -> PrioritizedReplayBufferSamples:
        indices = self.sum_tree.sample(batch_size)
        rows = indices // self.n_envs
        env_indices = indices % self.n_envs
        samples = self._encode_samples(rows, env_indices, self.n_step, self.gamma)

        probabilities = self.sum_tree.values(indices) / self.sum_tree.total
        weights = (max(self.num_transitions, 1) * probabilities) ** (-self.beta)
        weights /= weights.max()
        return PrioritizedReplayBufferSamples(
            observations=samples.observations,
            actions=samples.actions,
            rewards=samples.rewards,
            next_observations=samples.next_observations,
            dones=samples.dones,
            indices=indices,
            weights=self._to_torch(weights.astype(np.float32).reshape(-1, 1)),
        )

    def update_priorities(self, indices, priorities) -> None:
        priorities = np.abs(np.asarray(priorities, dtype=np.float32).reshape(-1)) + self.eps
        self.max_priority = max(self.max_priority, float(priorities.max()))
        self.sum_tree.update(indices, priorities**self.alpha)
