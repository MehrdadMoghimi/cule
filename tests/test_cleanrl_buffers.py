import gymnasium as gym
import numpy as np
import torch

from cleanrl_utils.buffers import AtariReplayBuffer, PrioritizedAtariReplayBuffer


OBSERVATION_SPACE = gym.spaces.Box(0, 255, (4, 84, 84), dtype=np.uint8)
ACTION_SPACE = gym.spaces.Discrete(4)


def observations(values):
    result = np.zeros((len(values), 4, 84, 84), dtype=np.uint8)
    result[:, -1] = np.asarray(values, dtype=np.uint8)[:, None, None]
    return result


def test_vector_replay_reconstructs_stacks_and_episode_boundaries():
    replay = AtariReplayBuffer(20, OBSERVATION_SPACE, ACTION_SPACE, "cpu", n_envs=2)
    replay.initialize(observations([1, 10]))
    replay.add(observations([2, 11]), [0, 1], [1.0, 2.0], [False, False])
    replay.add(observations([3, 12]), [1, 2], [3.0, 4.0], [True, False])
    replay.add(observations([4, 13]), [2, 0], [5.0, 6.0], [False, False])

    samples = replay._encode_samples(np.array([2, 2]), np.array([0, 1]))
    assert samples.observations[:, :, 0, 0].tolist() == [
        [0, 0, 0, 3],
        [0, 10, 11, 12],
    ]
    assert samples.next_observations[:, :, 0, 0].tolist() == [
        [0, 0, 3, 4],
        [10, 11, 12, 13],
    ]
    assert samples.rewards[:, 0].tolist() == [5.0, 6.0]


def test_vector_replay_remains_contiguous_after_wraparound():
    replay = AtariReplayBuffer(16, OBSERVATION_SPACE, ACTION_SPACE, "cpu", n_envs=2)
    replay.initialize(observations([0, 0]))
    for step in range(30):
        replay.add(
            observations([step + 1, step + 1]),
            [0, 1],
            [1.0, 2.0],
            [step % 9 == 8, False],
        )

    for _ in range(20):
        samples = replay.sample(8)
        assert samples.observations.shape == (8, 4, 84, 84)
        assert samples.next_observations.shape == (8, 4, 84, 84)


def test_prioritized_replay_keeps_n_step_returns_separate_per_env():
    replay = PrioritizedAtariReplayBuffer(
        64,
        OBSERVATION_SPACE,
        ACTION_SPACE,
        "cpu",
        n_envs=2,
        n_step=3,
        gamma=1.0,
    )
    replay.initialize(observations([0, 0]))
    for step in range(8):
        replay.add(
            observations([step + 1, step + 1]),
            [0, 1],
            [1.0, 10.0],
            [False, False],
        )

    samples = replay.sample(64)
    assert set(samples.rewards[:, 0].tolist()) == {3.0, 30.0}
    replay.update_priorities(samples.indices, np.full(64, 2.0, dtype=np.float32))
    assert torch.isfinite(samples.weights).all()
