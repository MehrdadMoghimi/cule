"""Common full-game Atari evaluation for tensor-native CleanRL trainers."""

from __future__ import annotations

import os
import sys
from collections.abc import Callable

import numpy as np
import torch

from torchcule.atari import Env as AtariEnv

_CLEANRL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cleanrl")
if _CLEANRL_DIR not in sys.path:
    sys.path.insert(0, _CLEANRL_DIR)

from diagnostic_env import (  # isort:skip
    DiagnosticVectorEnv,
    is_diagnostic_env_id,
    optimal_episode_length,
    parse_diagnostic_env_id,
)


@torch.no_grad()
def evaluate_cule_policy(
    env_id: str,
    action_fn: Callable[[torch.Tensor], torch.Tensor],
    device: torch.device,
    num_episodes: int = 10,
    seed: int = 10_000,
    frame_stack: int = 4,
    max_episode_steps: int = 18_000,
) -> dict[str, float]:
    """Evaluate one complete, unclipped game per CuLE environment.

    Evaluation uses the CPU CuLE backend, no sticky actions, frame skip four,
    and no episodic-life termination.  The policy remains on ``device``.  A
    FIRE action is issued at the start and after life loss, matching the
    evaluation behavior used by the native CuLE examples.
    """

    if num_episodes < 1:
        raise ValueError("num_episodes must be positive")
    if max_episode_steps < 1:
        raise ValueError("max_episode_steps must be positive")

    # The closed-form diagnostic MDP borrows this entry point so that the
    # trainers' evaluation path works unchanged on it.  There are no lives and
    # no FIRE action, so the Atari-specific handling below does not apply.
    if is_diagnostic_env_id(env_id):
        return _evaluate_diagnostic_policy(
            env_id, action_fn, device, num_episodes, seed, frame_stack, max_episode_steps
        )

    env = AtariEnv(
        env_id,
        num_envs=num_episodes,
        color_mode="gray",
        device="cpu",
        rescale=True,
        frameskip=4,
        repeat_prob=0.0,
        episodic_life=False,
    )
    seeds = torch.arange(seed, seed + num_episodes, dtype=torch.int32)
    observations = env.reset(seeds=seeds, initial_steps=50).squeeze(-1)

    # Breakout needs FIRE to launch the ball.  This first action mirrors the
    # native example evaluator and is excluded from the reported episode data.
    initial_actions = torch.ones(num_episodes, dtype=torch.uint8)
    _, _, _, info = env.step(initial_actions)
    lives = info["ale.lives"].clone()

    states = torch.zeros(
        (num_episodes, frame_stack, 84, 84),
        device=device,
        dtype=torch.float32,
    )
    states[:, -1].copy_(observations.to(device=device, dtype=torch.float32))
    rewards = torch.zeros(num_episodes, dtype=torch.float32)
    lengths = torch.zeros(num_episodes, dtype=torch.int32)
    completed = torch.zeros(num_episodes, dtype=torch.bool)
    fire_reset = torch.zeros(num_episodes, dtype=torch.bool)

    while not completed.all():
        actions = action_fn(states).detach().to(device="cpu", dtype=torch.uint8).reshape(-1)
        actions[fire_reset] = 1
        observations, step_rewards, dones, info = env.step(actions)
        observations = observations.squeeze(-1)
        new_lives = info["ale.lives"].clone()
        fire_reset = new_lives < lives
        lives.copy_(new_lives)

        active = ~completed
        rewards.add_(step_rewards.cpu() * active.float())
        lengths.add_(active.int())
        completed.logical_or_(dones.cpu())
        completed.logical_or_(lengths >= max_episode_steps)

        observations = observations.to(device=device, dtype=torch.float32)
        states[:, :-1].copy_(states[:, 1:].clone())
        states.mul_((~dones.to(device=device)).view(-1, 1, 1, 1))
        states[:, -1].copy_(observations)

    rewards_np = rewards.numpy()
    lengths_np = lengths.numpy()
    return {
        "reward_mean": float(np.mean(rewards_np)),
        "reward_median": float(np.median(rewards_np)),
        "reward_min": float(np.min(rewards_np)),
        "reward_max": float(np.max(rewards_np)),
        "reward_std": float(np.std(rewards_np)),
        "length_mean": float(np.mean(lengths_np)),
        "length_median": float(np.median(lengths_np)),
        "length_min": int(np.min(lengths_np)),
        "length_max": int(np.max(lengths_np)),
        "length_std": float(np.std(lengths_np)),
    }


@torch.no_grad()
def _evaluate_diagnostic_policy(
    env_id: str,
    action_fn: Callable[[torch.Tensor], torch.Tensor],
    device: torch.device,
    num_episodes: int,
    seed: int,
    frame_stack: int,
    max_episode_steps: int,
) -> dict[str, float]:
    """Greedy rollouts on the diagnostic MDP, one complete episode per env.

    The optimal policy is known, so `length_mean` is as informative as
    `reward_mean` here: a correct greedy policy finishes in exactly
    `optimal_episode_length(spec)` steps, and any larger number counts the STAY
    actions the agent still takes.
    """
    spec = parse_diagnostic_env_id(env_id)
    env = DiagnosticVectorEnv(
        spec,
        num_envs=num_episodes,
        device=device,
        seed=seed,
        frame_stack=frame_stack,
        clip_rewards=False,
    )
    observations, _ = env.reset()

    rewards = torch.zeros(num_episodes, dtype=torch.float32)
    lengths = torch.zeros(num_episodes, dtype=torch.int32)
    completed = torch.zeros(num_episodes, dtype=torch.bool)

    while not completed.all():
        actions = action_fn(observations.to(dtype=torch.float32)).detach().reshape(-1)
        observations, step_rewards, dones, _, _ = env.step(actions)
        active = ~completed
        rewards.add_(step_rewards.cpu() * active.float())
        lengths.add_(active.int())
        completed.logical_or_(dones.cpu())
        completed.logical_or_(lengths >= max_episode_steps)

    rewards_np = rewards.numpy()
    lengths_np = lengths.numpy()
    return {
        "reward_mean": float(np.mean(rewards_np)),
        "reward_median": float(np.median(rewards_np)),
        "reward_min": float(np.min(rewards_np)),
        "reward_max": float(np.max(rewards_np)),
        "reward_std": float(np.std(rewards_np)),
        "length_mean": float(np.mean(lengths_np)),
        "length_median": float(np.median(lengths_np)),
        "length_min": int(np.min(lengths_np)),
        "length_max": int(np.max(lengths_np)),
        "length_std": float(np.std(lengths_np)),
        "optimal_reward_mean": float(spec.mean_terminal_reward),
        "optimal_length_mean": float(optimal_episode_length(spec)),
    }
