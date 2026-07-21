"""Tensor-native CuLE vector environment helpers for the CleanRL Atari scripts.

Original to this fork.  It exposes torchcule through the subset of the
Gymnasium vector API that the bundled CleanRL-derived trainers call, so no
CleanRL or LeanRL code is reused here.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
import torch

from torchcule.atari import Env as AtariEnv


class CuLEVectorEnv:
    """Expose CuLE through the small Gymnasium vector API used by CleanRL.

    Observations stay on ``device`` and use the conventional ``NCHW`` four-frame
    Atari layout.  The returned observation tensor is reused by the next call to
    :meth:`step`; replay-based callers must copy an observation before stepping.
    """

    is_cule = True

    def __init__(
        self,
        env_id: str,
        num_envs: int,
        device: torch.device | str,
        seed: int = 1,
        episodic_life: bool = True,
        clip_rewards: bool = True,
        frame_stack: int = 4,
        stats_interval: int = 128,
        sync_step: bool = False,
    ) -> None:
        self.device = torch.device(device)
        self.num_envs = num_envs
        self.frame_stack = frame_stack
        self.clip_rewards = clip_rewards
        self.stats_interval = max(1, stats_interval)
        # sync_step=False steps the backend asynchronously; results are still
        # stream-ordered for GPU consumers, and the periodic episode-stat flush
        # (a .cpu() read) is the only host-side reader.
        self.sync_step = sync_step
        self._step_count = 0

        self.env = AtariEnv(
            env_id,
            num_envs=num_envs,
            color_mode="gray",
            device=self.device,
            rescale=True,
            frameskip=4,
            repeat_prob=0.0,
            episodic_life=episodic_life,
            max_noop_steps=30,
        )
        self.env.train(frameskip=4)

        self.single_action_space = self.env.action_space
        self.action_space = self.single_action_space
        self.single_action_space.seed(seed)
        self.single_observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(frame_stack, 84, 84),
            dtype=np.uint8,
        )
        self.observation_space = self.single_observation_space

        # Frame stacking uses a double-length ring: every new frame is written
        # at _ring_pos and _ring_pos + frame_stack, so the newest `frame_stack`
        # frames are always readable as one slice and no per-step shift copy is
        # needed.
        self._frames = torch.zeros(
            (num_envs, 2 * frame_stack, 84, 84), device=self.device, dtype=torch.uint8
        )
        self._ring_pos = frame_stack - 1
        self._episode_returns = torch.zeros(num_envs, device=self.device)
        self._episode_lengths = torch.zeros(num_envs, device=self.device, dtype=torch.int64)
        self._returned_returns = torch.zeros_like(self._episode_returns)
        self._returned_lengths = torch.zeros_like(self._episode_lengths)
        self._pending_episodes = torch.zeros(num_envs, device=self.device, dtype=torch.bool)
        self._seed = seed

    def _seeds(self, seed: int | None) -> torch.Tensor:
        base_seed = self._seed if seed is None else seed
        return torch.arange(
            base_seed,
            base_seed + self.num_envs,
            device=self.device,
            dtype=torch.int32,
        )

    @staticmethod
    def _gray_frame(observations: torch.Tensor) -> torch.Tensor:
        return observations.squeeze(-1)

    def _write_frame(self, frame: torch.Tensor) -> None:
        self._frames[:, self._ring_pos].copy_(frame)
        self._frames[:, self._ring_pos + self.frame_stack].copy_(frame)

    def _obs_view(self) -> torch.Tensor:
        start = self._ring_pos + 1
        return self._frames[:, start : start + self.frame_stack]

    def reset(self, *, seed: int | None = None, **_: Any) -> tuple[torch.Tensor, dict[str, Any]]:
        observations = self.env.reset(seeds=self._seeds(seed), initial_steps=0)
        self._frames.zero_()
        self._ring_pos = self.frame_stack - 1
        self._write_frame(self._gray_frame(observations))
        self._episode_returns.zero_()
        self._episode_lengths.zero_()
        self._pending_episodes.zero_()
        return self._obs_view(), {}

    def _flush_episode_infos(self) -> dict[str, Any]:
        # Avoid a GPU-to-CPU synchronization on every environment step. Episode
        # statistics are only logging data, so batching them does not affect RL.
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
        observations, rewards, terminated, raw_info = self.env.step(
            actions, asyn=not self.sync_step
        )
        frame = self._gray_frame(observations)

        self._episode_returns.add_(rewards)
        self._episode_lengths.add_(1)
        game_over = terminated & raw_info["ale.lives"].eq(0)
        self._returned_returns.copy_(
            torch.where(game_over, self._episode_returns, self._returned_returns)
        )
        self._returned_lengths.copy_(
            torch.where(game_over, self._episode_lengths, self._returned_lengths)
        )
        self._pending_episodes.logical_or_(game_over)
        self._episode_returns.masked_fill_(game_over, 0)
        self._episode_lengths.masked_fill_(game_over, 0)

        self._ring_pos = (self._ring_pos + 1) % self.frame_stack
        self._frames.mul_((~terminated).view(-1, 1, 1, 1))
        self._write_frame(frame)

        self._step_count += 1
        infos = self._flush_episode_infos()
        infos["lives"] = raw_info["ale.lives"]
        truncations = torch.zeros_like(terminated)
        training_rewards = rewards.sign() if self.clip_rewards else rewards
        return self._obs_view(), training_rewards, terminated, truncations, infos

    def close(self) -> None:
        # torchcule owns no Python-side closeable resource.
        return None


def make_cule_env(
    env_id: str,
    num_envs: int,
    device: torch.device | str,
    seed: int,
    capture_video: bool = False,
    frame_stack: int = 4,
) -> CuLEVectorEnv:
    if capture_video:
        raise ValueError("CuLE backend does not support CleanRL video capture")
    return CuLEVectorEnv(env_id, num_envs, device, seed=seed, frame_stack=frame_stack)


def resolve_cule_device(
    requested_device: str, training_device: torch.device, num_envs: int
) -> torch.device:
    """Choose CPU for tiny CuLE batches and the training GPU for vector batches."""
    if requested_device != "auto":
        return torch.device(requested_device)
    if training_device.type == "cuda" and num_envs >= 32:
        return training_device
    return torch.device("cpu")


def step_env(envs: Any, actions: torch.Tensor | np.ndarray):
    """Step either CuLE with tensors or a CPU vector env with NumPy actions."""
    if getattr(envs, "is_cule", False):
        return envs.step(actions)
    return envs.step(to_numpy(actions))


def to_tensor(value: Any, device: torch.device, dtype: torch.dtype | None = None) -> torch.Tensor:
    tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    return tensor.to(device=device, dtype=dtype) if dtype is not None else tensor.to(device=device)


def to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def copy_obs(value: Any):
    return value.clone() if isinstance(value, torch.Tensor) else value.copy()


def grayscale_observation(env: gym.Env):
    wrapper = getattr(gym.wrappers, "GrayscaleObservation", None)
    if wrapper is None:
        wrapper = gym.wrappers.GrayScaleObservation
    return wrapper(env)


def frame_stack_observation(env: gym.Env, stack_size: int):
    wrapper = getattr(gym.wrappers, "FrameStackObservation", None)
    if wrapper is not None:
        return wrapper(env, stack_size=stack_size)
    return gym.wrappers.FrameStack(env, stack_size)


def done_tensor(terminations: Any, truncations: Any, device: torch.device) -> torch.Tensor:
    terminations = to_tensor(terminations, device, torch.bool)
    truncations = to_tensor(truncations, device, torch.bool)
    return torch.logical_or(terminations, truncations).float()
