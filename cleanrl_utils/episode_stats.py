from __future__ import annotations

from collections import deque
from typing import Any


class EpisodeStats:
    """Track a moving return without coupling trainers to a logging backend."""

    def __init__(self, window: int, solve_reward: float | None = None) -> None:
        if window < 1:
            raise ValueError("solve_window must be positive")
        self.window = window
        self.solve_reward = solve_reward
        self.returns: deque[float] = deque(maxlen=window)
        self.episodes = 0
        self.best_mean_return = float("-inf")
        self.solved_at_step: int | None = None

    @property
    def mean_return(self) -> float | None:
        if not self.returns:
            return None
        return sum(self.returns) / len(self.returns)

    def update(self, infos: dict[str, Any], global_step: int, writer: Any) -> bool:
        batch_returns: list[float] = []
        batch_lengths: list[int] = []
        for info in infos.get("final_info", ()):
            if not info or "episode" not in info:
                continue
            episode_return = float(info["episode"]["r"])
            episode_length = int(info["episode"]["l"])
            batch_returns.append(episode_return)
            batch_lengths.append(episode_length)
            self.returns.append(episode_return)
            self.episodes += 1
            mean_return = self.mean_return
            assert mean_return is not None
            self.best_mean_return = max(self.best_mean_return, mean_return)

            if (
                self.solved_at_step is None
                and self.solve_reward is not None
                and len(self.returns) == self.window
                and mean_return >= self.solve_reward
            ):
                self.solved_at_step = global_step
                print(
                    f"solved at global_step={global_step}: "
                    f"mean_return_{self.window}={mean_return}"
                )
        if batch_returns:
            batch_mean_return = sum(batch_returns) / len(batch_returns)
            batch_mean_length = sum(batch_lengths) / len(batch_lengths)
            mean_return = self.mean_return
            print(
                f"global_step={global_step}, completed_episodes={len(batch_returns)}, "
                f"batch_mean_return={batch_mean_return}, "
                f"mean_return_{len(self.returns)}={mean_return}"
            )
            if writer is not None:  # benchmark runs have no writer
                writer.add_scalar("charts/episodic_return", batch_mean_return, global_step)
                writer.add_scalar("charts/episodic_length", batch_mean_length, global_step)
                writer.add_scalar("charts/episodic_return_mean", mean_return, global_step)
        return self.solved_at_step is not None

    def print_summary(self) -> None:
        print("episodes:", self.episodes)
        print("recent mean return:", self.mean_return)
        print(
            "best moving mean return:",
            None if self.best_mean_return == float("-inf") else self.best_mean_return,
        )
        print("solved at step:", self.solved_at_step)
