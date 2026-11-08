"""Tensor-native CuLE vector environment helpers for the CleanRL Atari scripts.

Original to this fork.  It exposes torchcule through the subset of the
Gymnasium vector API that the bundled CleanRL-derived trainers call, so no
CleanRL or LeanRL code is reused here.
"""

from __future__ import annotations

import os
from typing import Any

import gymnasium as gym
import numpy as np
import torch

from diagnostic_env import (
    is_diagnostic_env_id,
    make_diagnostic_env,
)
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
        # Episode statistics are logging only, so how often they are flushed is a
        # measurement choice rather than an algorithmic one: a short experiment
        # needs a denser learning curve than the default batching provides, and a
        # long one wants fewer host synchronizations.
        self.stats_interval = max(1, int(os.environ.get("CULE_STATS_INTERVAL", stats_interval)))
        # sync_step=False steps the backend asynchronously; results are still
        # stream-ordered for GPU consumers, and the periodic episode-stat flush
        # (a .cpu() read) is the only host-side reader.
        self.sync_step = sync_step
        self._step_count = 0

        # CuLE is asked for `episodic_life=False` whatever the caller wants, and
        # the life-loss terminal is reconstructed here instead.  The reason is
        # that CuLE's done buffer is `FLAG_ALE_TERMINAL || (episodic_life &&
        # lost_life)`, which conflates the two, and the game-over half is the
        # only one that can end an episode -- so with `episodic_life=True` there
        # is no signal left that says "the game actually ended".
        #
        # This used to be recovered as `terminated & lives.eq(0)`, which is a
        # fact about Breakout and not about Atari.  Measured under a random
        # policy: Breakout does read 0 lives at game over, but Qbert reads 254
        # (its life counter underflows), Seaquest and Krull read 1, and for
        # Seaquest that value is indistinguishable from an ordinary life loss.
        # The effect was that seven of eighteen games tested -- Qbert, Seaquest,
        # Krull, Asterix, MsPacman, BeamRider, Frostbite -- reported *no
        # episodes at all*, so every learning curve on them was empty and every
        # score derived from one was None.
        #
        # Reconstructing instead of fixing it up is safe because the console
        # reset does not depend on this flag: `reset_functor` in
        # `cule/atari/functors.hpp` resets a state only on FLAG_ALE_TERMINAL,
        # never on the done buffer.  So the emulator behaves identically either
        # way and only the reporting changes -- which keeps every score already
        # recorded in `benchmark_results` comparable.
        self.episodic_life = episodic_life
        self.env = AtariEnv(
            env_id,
            num_envs=num_envs,
            color_mode="gray",
            device=self.device,
            rescale=True,
            frameskip=4,
            repeat_prob=0.0,
            episodic_life=False,
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
        self._prev_lives = torch.zeros(num_envs, device=self.device, dtype=torch.int32)
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
        self._prev_lives.fill_(2 ** 31 - 1)   # first step can never look like a life loss
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
        observations, rewards, game_over, raw_info = self.env.step(
            actions, asyn=not self.sync_step
        )
        frame = self._gray_frame(observations)

        # `game_over` as returned is FLAG_ALE_TERMINAL alone; a life loss is a
        # decrease in the lives counter, which is well defined in every game
        # whatever absolute value the counter happens to take.
        lives = raw_info["ale.lives"]
        lost_life = lives.lt(self._prev_lives)
        self._prev_lives.copy_(lives)
        terminated = game_over | lost_life if self.episodic_life else game_over

        self._episode_returns.add_(rewards)
        self._episode_lengths.add_(1)
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
    **kwargs: Any,
):
    if capture_video:
        raise ValueError("CuLE backend does not support CleanRL video capture")
    # A diagnostic env id routes to the closed-form MDP of `diagnostic_env`.
    # It presents the identical interface, so every trainer gets the correctness
    # harness without a per-trainer flag.  Trainers that used to construct
    # `CuLEVectorEnv` directly now come through here too, which is why the extra
    # keywords (`clip_rewards` and friends) are forwarded rather than dropped.
    if is_diagnostic_env_id(env_id):
        return make_diagnostic_env(
            env_id, num_envs, device, seed=seed, frame_stack=frame_stack, **kwargs)
    return CuLEVectorEnv(env_id, num_envs, device, seed=seed, frame_stack=frame_stack, **kwargs)


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
