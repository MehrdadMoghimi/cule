# Follows the evaluation interface of CleanRL's cleanrl_utils/evals/dqn_eval.py
# (https://github.com/vwxyzjn/cleanrl, MIT; license in cleanrl/LICENSE.md).
"""Shared epsilon-greedy Atari evaluation for the distributional trainers.

QR-DQN, IQN, and FQF share the CleanRL ``get_action`` interface but differ in
network constructor keywords, so the caller names the saved-args keys that its
``Model`` needs.
"""

import random
from argparse import Namespace
from typing import Callable, Sequence

import gymnasium as gym
import numpy as np
import torch


def evaluate(
    model_path: str,
    make_env: Callable,
    env_id: str,
    eval_episodes: int,
    run_name: str,
    Model: torch.nn.Module,
    model_kwargs_keys: Sequence[str],
    device: torch.device = torch.device("cpu"),
    epsilon: float = 0.05,
    capture_video: bool = True,
):
    envs = gym.vector.SyncVectorEnv([make_env(env_id, 0, 0, capture_video, run_name)])
    model_data = torch.load(model_path, map_location="cpu")
    args = Namespace(**model_data["args"])
    model = Model(envs, **{key: getattr(args, key) for key in model_kwargs_keys})
    model.load_state_dict(model_data["model_weights"])
    model = model.to(device)
    model.eval()

    obs, _ = envs.reset()
    episodic_returns = []
    while len(episodic_returns) < eval_episodes:
        if random.random() < epsilon:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            with torch.no_grad():
                actions, _ = model.get_action(torch.Tensor(obs).to(device))
            actions = actions.cpu().numpy()
        next_obs, _, _, _, infos = envs.step(actions)
        for info in infos.get("final_info", ()):
            if info and "episode" in info:
                print(f"eval_episode={len(episodic_returns)}, episodic_return={info['episode']['r']}")
                episodic_returns.append(float(info["episode"]["r"]))
        obs = next_obs

    return episodic_returns
